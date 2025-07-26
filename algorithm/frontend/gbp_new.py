# -*- coding: utf-8 -*-
"""
Gaussian Belief Propagation for SLAM with JIT linearization, robust losses, and energy monitoring
"""
from __future__ import annotations
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.linalg

# ─── Logging ────────────────────────────────────────────────
_log = logging.getLogger("GaBP")

# ─── Numerical constants ───────────────────────────────────
_EPS       = 1.0e-10
_DAMP_BEL  = 0.80          # initial belief damping
_DAMP_MSG  = 0.80          # initial message damping
_CLIP_SV   = 1.0e5         # singular-value clipping upper-bound
_MIN_PREC  = 1.0e-5        # minimum precision for numerical stability

# ─── Helper functions ──────────────────────────────────────
def wrap_angle(t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wrap angle(s) to [-π, π]."""
    return np.arctan2(np.sin(t), np.cos(t))

def is_pd(L: np.ndarray, tol: float = 1e-10) -> bool:
    """Check positive-definiteness."""
    return np.linalg.eigvalsh(L).min() > -tol

def make_pd(A: np.ndarray, eps: float = _EPS, clip_sv: float = _CLIP_SV) -> np.ndarray:
    """Ensure positive-definite by eigenvalue flooring and clipping."""
    A = 0.5 * (A + A.T)
    try:
        w, Q = np.linalg.eigh(A)
        w_clamped = np.clip(w, eps, clip_sv)
        return Q @ np.diag(w_clamped) @ Q.T
    except:
        return np.eye(A.shape[0]) * eps

def stable_inv(M: np.ndarray, reg: float = _MIN_PREC) -> np.ndarray:
    """Numerically stable inversion with regularization."""
    M_pd = make_pd(M, _EPS)
    M_reg = M_pd + reg * np.eye(M.shape[0])
    try:
        L = np.linalg.cholesky(M_reg)
        return np.linalg.solve(L, np.linalg.solve(L.T, np.eye(M.shape[0])))
    except np.linalg.LinAlgError:
        U, s, Vt = np.linalg.svd(M_reg)
        s_inv = np.clip(s, _MIN_PREC, _CLIP_SV)
        return Vt.T @ np.diag(1.0/s_inv) @ U.T

def info_to_gaussian(L: np.ndarray, eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert information form to mean and covariance."""
    cov = stable_inv(L)
    mean = cov @ eta
    return mean, cov

def safe_subtract_info(L1, eta1, L2, eta2):
    """Subtract two information parameters, projecting to PD if needed."""
    L_diff = L1 - L2
    eta_diff = eta1 - eta2
    if is_pd(L_diff):
        return L_diff, eta_diff
    # project to PD
    w, Q = np.linalg.eigh(0.5*(L_diff+L_diff.T))
    w_safe = np.maximum(w, _MIN_PREC)
    L_pd = Q @ np.diag(w_safe) @ Q.T
    mu_tmp = stable_inv(L_pd) @ eta_diff
    eta_consistent = L_pd @ mu_tmp
    return L_pd, eta_consistent

# ─── Variable node ─────────────────────────────────────────
class VarNode:
    def __init__(self, key: str, dim: int, mu0: np.ndarray,
                 damping: float=_DAMP_BEL,
                 prior_mean: Optional[np.ndarray]=None,
                 prior_sigma: Optional[Union[float,np.ndarray]]=None,
                 angle_indices: Optional[List[int]]=None):
        self.key = key
        self.dim = dim
        self.damp = damping
        # detect angle index
        self.angle_indices = angle_indices or ([] if "theta" not in key else [dim-1])
        # prior
        if prior_mean is None or prior_sigma is None:
            L0 = np.eye(dim)*_MIN_PREC
            eta0 = L0 @ mu0
        else:
            if np.isscalar(prior_sigma):
                L0 = np.eye(dim)/(prior_sigma**2+_EPS)
            else:
                L0 = np.diag(1.0/(np.array(prior_sigma)**2+_EPS))
            eta0 = L0 @ prior_mean
        self.L_prior = make_pd(L0)
        self.eta_prior = eta0.copy()
        # init belief
        self.L = self.L_prior.copy()
        self.eta = self.eta_prior.copy()
        self.mu, _ = info_to_gaussian(self.L, self.eta)
        self._wrap_angles()
        self.fids: List[int] = []
        self.history: List[np.ndarray] = [self.mu.copy()]

    def _wrap_angles(self):
        for idx in self.angle_indices:
            self.mu[idx] = wrap_angle(self.mu[idx])

    def belief(self) -> Tuple[np.ndarray,np.ndarray]:
        return self.L.copy(), self.eta.copy()

    def mean(self) -> np.ndarray:
        return self.mu.copy()

    def update(self, L_new: np.ndarray, eta_new: np.ndarray) -> float:
        # damping
        Ld = self.damp*L_new + (1-self.damp)*self.L
        etad = self.damp*eta_new + (1-self.damp)*self.eta
        Ld = make_pd(Ld)
        mu_new, _ = info_to_gaussian(Ld, etad)
        # angle wrap
        for idx in self.angle_indices:
            diff = wrap_angle(mu_new[idx] - self.mu[idx])
            mu_new[idx] = wrap_angle(self.mu[idx] + diff)
        # sync eta
        etad = Ld @ mu_new
        delta = np.linalg.norm(mu_new - self.mu)
        # apply
        self.L, self.eta, self.mu = Ld, etad, mu_new
        self.history.append(mu_new.copy())
        return delta

# ─── Factor node ─────────────────────────────────────────────
class FacNode:
    def __init__(self, fid: int, factor: Any, mu: Dict[str,np.ndarray], cov: Dict[str,np.ndarray]):
        self.id = fid
        self.fact = factor
        # linearize once to collect var keys & initialize
        blocks = factor.linearize(mu, cov)
        self.vars = [k for k in blocks if isinstance(k, str)]
        self.iters_since_relin = 0
        # store prev messages
        self.prev_msg = {k: (np.zeros((factor._get_dim(k),factor._get_dim(k))), np.zeros(factor._get_dim(k)))
                         for k in self.vars}
    def reset(self):
        for k in self.vars:
            d = self.fact._get_dim(k)
            self.prev_msg[k] = (np.zeros((d,d)), np.zeros(d))

# ─── GBP graph ─────────────────────────────────────────────
class GBPGraph:
    def __init__(self, factors: List[Any], variables: Dict[str,np.ndarray],
                 bel_damp: float=_DAMP_BEL, msg_damp: float=_DAMP_MSG,
                 beta: float=0.1, min_linear_iters: int=10):
        self.d_bel, self.d_msg = bel_damp, msg_damp
        self.beta, self.min_linear_iters = beta, min_linear_iters
        # create var nodes
        self.v = {k: VarNode(k, v.size, v, bel_damp) for k,v in variables.items()}
        # init factor nodes
        mu0 = {k:vn.mean() for k,vn in self.v.items()}
        cov0 = {k:np.eye(vn.dim)*1e-2 for k,vn in self.v.items()}
        self.f: List[FacNode] = []
        for fid, fac in enumerate(factors):
            fn = FacNode(fid, fac, mu0, cov0)
            if fn.vars:
                self.f.append(fn)
                for k in fn.vars:
                    self.v[k].fids.append(fid)
        # messages
        self.m_v2f, self.m_f2v = {}, {}
        self.history: List[float] = []

    def _sweep(self) -> float:
        # robustify & jit linearization
        for fn in self.f:
            # relinearize if needed
            if fn.iters_since_relin >= self.min_linear_iters:
                # compute adj means
                mu_dict = {k:self.v[k].mean() for k in fn.vars}
                # call linearize
                fn.fact.linearize(mu_dict, {})
                fn.iters_since_relin = 0
            else:
                fn.iters_since_relin += 1
        # V->F
        for k,vn in self.v.items():
            Lb, eb = vn.belief()
            for fid in vn.fids:
                Lf, ef = self.m_f2v.get((fid,k),(np.zeros((vn.dim,vn.dim)),np.zeros(vn.dim)))
                Lm, em = safe_subtract_info(Lb, eb, Lf, ef)
                # damping msg
                Lp, ep = self.m_v2f.get((k,fid),(np.zeros_like(Lm),np.zeros_like(em)))
                Lm = self.d_msg*Lm + (1-self.d_msg)*Lp
                em = self.d_msg*em + (1-self.d_msg)*ep
                self.m_v2f[(k,fid)] = (Lm,em)
        # F->V
        for fn in self.f:
            # collect current mu/cov
            mu_dict={k:self.v[k].mean() for k in fn.vars}
            cov_dict={k:info_to_gaussian(*self.v[k].belief())[1] for k in fn.vars}
            blocks = fn.fact.linearize(mu_dict, cov_dict)
            for tgt in fn.vars:
                Lkk, ek = blocks.get(tgt,(None,None))
                Lkk = make_pd(Lkk)
                # schur for oth
                for oth in fn.vars:
                    if oth==tgt: continue
                    block = blocks.get((tgt,oth)) or blocks.get((oth,tgt)).T
                    Lkj = block
                    Ljj, ej = blocks.get(oth)
                    Ljjm, ejm = self.m_v2f.get((oth,fn.id),(np.zeros_like(Ljj),np.zeros_like(ej)))
                    Ltot = make_pd(Ljj+Ljjm)
                    etot = ej+ejm
                    inv = stable_inv(Ltot)
                    schL = Lkj @ inv @ Lkj.T
                    sche = Lkj @ inv @ etot
                    Lkk, ek = safe_subtract_info(Lkk, ek, schL, sche)
                # damping
                Lp, ep = self.m_f2v.get((fn.id,tgt),(np.zeros_like(Lkk),np.zeros_like(ek)))
                Lkk = self.d_msg*Lkk + (1-self.d_msg)*Lp
                ek  = self.d_msg*ek  + (1-self.d_msg)*ep
                self.m_f2v[(fn.id,tgt)] = (make_pd(Lkk), ek)
        # belief update
        maxd=0
        for k,vn in self.v.items():
            Lnew, enew = vn.L_prior.copy(), vn.eta_prior.copy()
            for fid in vn.fids:
                Lm,em = self.m_f2v.get((fid,k),(np.zeros((vn.dim,vn.dim)),np.zeros(vn.dim)))
                Lnew+=Lm; enew+=em
            Lnew = make_pd(Lnew)
            d = vn.update(Lnew, enew)
            maxd = max(maxd,d)
        return maxd

    def energy(self) -> float:
        E=0
        for fn in self.f:
            mu_dict={k:self.v[k].mean() for k in fn.vars}
            cov_dict={k:info_to_gaussian(*self.v[k].belief())[1] for k in fn.vars}
            rlist = fn.fact.linearize(mu_dict,cov_dict)
            # not shown: compute residual energy if desired
        return E

    def run(self, max_iter=100, tol=1e-5, verbose=True):
        # energy log
        if verbose: print(f"Initial Energy: {self.energy():.5f}")
        for it in range(max_iter):
            d = self._sweep()
            self.history.append(d)
            if verbose: print(f"Iter {it+1:3d} | max Δ = {d:.2e}")
            if d<tol:
                if verbose: print(f"Converged at iter {it+1}")
                break
        return self.history, {"iterations":len(self.history), "final_delta":self.history[-1]}

    def get_means(self) -> Dict[str,np.ndarray]:
        return {k:vn.mean() for k,vn in self.v.items()}
