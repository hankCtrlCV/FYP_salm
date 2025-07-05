# -------------------------------------------------------------
# Gaussian Belief Propagation – Improved version for SLAM
# -------------------------------------------------------------
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.linalg

# ─── Logging ────────────────────────────────────────────────
_log = logging.getLogger("GaBP")

# ─── Numerical constants ───────────────────────────────────
_EPS       = 1.0e-8
_DAMP_BEL  = 0.50          # belief-update damping
_DAMP_MSG  = 0.30          # message-update damping
_CLIP_SV   = 1.0e6         # singular-value clipping upper-bound
_MIN_PREC  = 1.0e-12       # minimum precision for numerical stability

# ─── Helper functions ──────────────────────────────────────
def wrap_angle(t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """ Wrap angle(s) to [-π, π]. """
    return np.arctan2(np.sin(t), np.cos(t)) if isinstance(t, np.ndarray) \
           else math.atan2(math.sin(t), math.cos(t))

def angle_diff(a1: float, a2: float) -> float:
    """Compute angular difference a1 - a2 wrapped to [-π, π]."""
    return wrap_angle(a1 - a2)

def make_pd(A: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """Ensure positive-definite by eigenvalue flooring with better conditioning."""
    A = 0.5 * (A + A.T)
    w, Q = np.linalg.eigh(A)
    # More aggressive regularization for numerical stability
    w = np.maximum(w, eps)
    # Add small diagonal regularization
    w += eps * np.max(w)
    return Q @ np.diag(w) @ Q.T

def stable_inv(M: np.ndarray, reg: float = _MIN_PREC) -> np.ndarray:
    """Numerically stable matrix inversion with regularization."""
    M = make_pd(M, _EPS)
    # Add diagonal regularization for better conditioning
    M_reg = M + reg * np.eye(M.shape[0])
    
    try:
        # Try Cholesky first (fastest for PD matrices)
        L = np.linalg.cholesky(M_reg)
        return np.linalg.solve(L, np.linalg.solve(L.T, np.eye(M.shape[0])))
    except np.linalg.LinAlgError:
        # Fallback to SVD
        U, s, Vt = np.linalg.svd(M_reg, full_matrices=False)
        s_inv = 1.0 / np.maximum(s, _EPS)
        return Vt.T @ np.diag(s_inv) @ U.T

def info_to_gaussian(L: np.ndarray, eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert information form to moment form."""
    Σ = stable_inv(L)
    μ = Σ @ eta
    return μ, Σ


def _clip_pair(L: np.ndarray, eta: np.ndarray,
               s_max_lim: float = _CLIP_SV) -> Tuple[np.ndarray, np.ndarray]:
    """
    If the greatest singular value of L exceeds s_max_lim, uniformly scale
    (L, η) so that the new max-SV equals s_max_lim while keeping μ unchanged.
    Since μ = L⁻¹η, scaling both by the same factor keeps μ invariant.
    """
    if L.size == 0:
        return L, eta

    try:
        s_max = np.linalg.svd(L, compute_uv=False).max()
        if s_max > s_max_lim:
            scale = s_max_lim / s_max          # 直接压缩到阈值
            return L * scale, eta * scale      # 同一次幂保持 μ 不变
    except Exception:      # SVD 失败直接跳过
        pass
    return L, eta


def safe_subtract_info(L1: np.ndarray, eta1: np.ndarray, 
                      L2: np.ndarray, eta2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Safely subtract information matrices while maintaining PD property.
    
    When L_diff = L1 - L2 becomes indefinite, we project it to PD space.
    To maintain consistency, we have two options:
    1. Keep η_diff = η1 - η2 (simpler, may be inconsistent)
    2. Re-project η to maintain μ consistency (more complex, numerically stable)
    """
    L_diff = L1 - L2
    eta_diff = eta1 - eta2
    
    # Check if result is already positive definite
    try:
        w_min = np.linalg.eigvals(L_diff).min()
        if w_min >= _MIN_PREC:
            return L_diff, eta_diff  # Already PD, no projection needed
    except:
        pass
    
    # Project to PD space
    w, Q = np.linalg.eigh(L_diff)
    w_clipped = np.maximum(w, _MIN_PREC)
    L_pd = Q @ np.diag(w_clipped) @ Q.T
    
    # Option 1: Simple approach (keeps original eta_diff)
    # This is faster but may introduce slight inconsistency
    if True:  # Use simple approach by default
        return L_pd, eta_diff
    
    # Option 2: Consistency-preserving approach
    # Re-project η onto new precision to maintain self-consistency
    try:
        mu_tmp = stable_inv(L_pd) @ eta_diff
        eta_consistent = L_pd @ mu_tmp
        return L_pd, eta_consistent
    except:
        # Fallback to simple approach if projection fails
        return L_pd, eta_diff

# ─── Variable node ─────────────────────────────────────────
class VarNode:
    def __init__(self, key: str, dim: int, mu0: np.ndarray,
                 damping: float,
                 prior_mean: Optional[np.ndarray] = None,
                 prior_sigma: Optional[Union[float, np.ndarray]] = None,
                 angle_indices: Optional[List[int]] = None):
        self.key   = key
        self.dim   = dim
        self.damp  = float(damping)
        
        # Improved angle detection
        self.angle_indices = angle_indices or []
        if not self.angle_indices:
            # Default angle detection for SLAM
            if "theta" in key.lower() or "yaw" in key.lower():
                self.angle_indices = [dim - 1]  # Assume last dimension is angle
            elif key.startswith("x") and dim >= 3:
                self.angle_indices = [2]  # SE(2) pose: [x, y, theta]
        
        # Setup prior
        if prior_mean is None or prior_sigma is None:
            # Weak prior
            L0 = np.eye(dim) * _MIN_PREC
            eta0 = L0 @ mu0
        else:
            if np.isscalar(prior_sigma):
                L0 = np.eye(dim) / (prior_sigma**2 + _EPS)
            else:
                sigma_arr = np.array(prior_sigma)
                L0 = np.diag(1.0 / (sigma_arr**2 + _EPS))
            eta0 = L0 @ prior_mean
        
        L0 = make_pd(L0)
        self.L_prior, self.eta_prior = L0, eta0
        self.L, self.eta             = L0.copy(), eta0.copy()
        
        # Initialize mean
        self.mu, _ = info_to_gaussian(self.L, self.eta)
        self._wrap_angles()
        
        self.mu_prev = self.mu.copy()
        self.history = [self.mu.copy()]
        self.fids: List[int] = []

    def _wrap_angles(self):
        """Wrap angle components to [-π, π]."""
        for idx in self.angle_indices:
            if 0 <= idx < self.dim:
                self.mu[idx] = wrap_angle(self.mu[idx])

    def belief(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.L.copy(), self.eta.copy()
    
    def mean(self) -> np.ndarray:
        return self.mu.copy()

    def update(self, L_new: np.ndarray, eta_new: np.ndarray) -> float:
        """Update belief with damping and proper angle handling."""
        # Damped update
        L_damped = self.damp * L_new + (1 - self.damp) * self.L
        eta_damped = self.damp * eta_new + (1 - self.damp) * self.eta
        
        # Ensure PD
        L_damped = make_pd(L_damped)
        
        # Compute new mean
        mu_new, _ = info_to_gaussian(L_damped, eta_damped)
        
        # Handle angle updates properly
        if self.angle_indices:
            for idx in self.angle_indices:
                if 0 <= idx < self.dim:
                    # Use angular difference for proper angle updates
                    angle_diff_val = angle_diff(mu_new[idx], self.mu[idx])
                    mu_new[idx] = wrap_angle(self.mu[idx] + angle_diff_val)

        
        # Compute convergence metric
        delta = float(np.linalg.norm(mu_new - self.mu))
        
        # Update state
        self.mu_prev = self.mu.copy()
        self.mu = mu_new
        self.L = L_damped
        self.eta = eta_damped
        self.history.append(self.mu.copy())
        
        return delta

# ─── Factor node ─────────────────────────────────────────────
class FacNode:
    def __init__(self, fid: int, factor: Any, vkeys: List[str]):
        self.id   = fid
        self.fact = factor
        self.vars = [k for k in vkeys if factor._get_dim(k) > 0]
        self.prev_msg = {k: (np.zeros((factor._get_dim(k), factor._get_dim(k))),
                             np.zeros(factor._get_dim(k))) for k in self.vars}
        self.robot_set = {  # 记录与此因子相连的所有机器人 ID
            int(v.split('_')[0][1:])            # 从 'x0_3' 提取 0
            for v in self.vars if v.startswith('x')
        }

    def reset(self):
        """Reset all previous messages."""
        for k in self.vars:
            dim = self.fact._get_dim(k)
            self.prev_msg[k] = (np.zeros((dim, dim)), np.zeros(dim))

# ─── GBP graph ─────────────────────────────────────────────
class GBPGraph:
    def __init__(self, factors: List[Any], variables: Dict[str, np.ndarray],
                 priors: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
                 angle_vars: Optional[Dict[str, List[int]]] = None,
                 bel_damp: float = _DAMP_BEL, msg_damp: float = _DAMP_MSG):
        
        self.d_bel, self.d_msg = float(bel_damp), float(msg_damp)
        
        # Variable nodes
        self.v: Dict[str, VarNode] = {}
        for k, μ0 in variables.items():
            pm, ps = priors.get(k, (None, None)) if priors else (None, None)
            angle_idx = angle_vars.get(k) if angle_vars else None
            self.v[k] = VarNode(k, μ0.size, μ0, self.d_bel, pm, ps, angle_idx)
        
        # Factor nodes
        self.f: List[FacNode] = []
        for fid, fac in enumerate(factors):
            keys = [k for k in self.v if fac._get_dim(k) > 0]
            if keys:
                fn = FacNode(fid, fac, keys)
                self.f.append(fn)
                for k in keys:
                    self.v[k].fids.append(fid)
        
        # Message storage
        self.m_v2f: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}
        self.m_f2v: Dict[Tuple[int, str], Tuple[np.ndarray, np.ndarray]] = {}
        self.max_delta_history: List[float] = []
        
        #communication statistics
        self.comm_bytes = 0
        self.robot_vars = self._identify_robot_variables()

    def _identify_robot_variables(self) -> Dict[str, int]:
        """识别每个变量属于哪个机器人"""
        robot_vars = {}
        for var_name in self.v:
            robot_id = self._get_robot_id(var_name)
            if robot_id is not None:
                robot_vars[var_name] = robot_id
        return robot_vars

    def _get_robot_id(self, var_name: str) -> Optional[int]:
        """从变量名提取机器人ID"""
        if var_name.startswith('x'):
            parts = var_name.split('_')
            if len(parts) >= 2:
                # 格式如 'x0_1' 表示机器人0的第1个位姿
                try:
                    return int(parts[0][1:])  # 提取'x'后的数字
                except ValueError:
                    pass
        return None

    
    def _is_cross_robot(self, var_name: str, fn: FacNode) -> bool:
        if var_name not in self.robot_vars:           # 变量未归属任何机器人
            return False
        return self.robot_vars[var_name] not in fn.robot_set
    
    def _sweep(self) -> float:
        """One complete GBP iteration."""
        
        # 1) Variable to Factor messages
        for k, vn in self.v.items():
            L_belief, eta_belief = vn.belief()
            
            for fid in vn.fids:
                fn_obj = self.f[fid]
                # Get previous F→V message
                L_f2v, eta_f2v = self.m_f2v.get((fid, k), 
                                                (np.zeros((vn.dim, vn.dim)),
                                                 np.zeros(vn.dim)))
                
                # Compute V→F message (belief minus incoming)
                L_msg, eta_msg = safe_subtract_info(L_belief, eta_belief, 
                                                   L_f2v, eta_f2v)
                
                # Damping
                L_prev, eta_prev = self.m_v2f.get((k, fid),
                                                 (np.zeros_like(L_msg), 
                                                  np.zeros_like(eta_msg)))
                L_msg = self.d_msg * L_msg + (1 - self.d_msg) * L_prev
                eta_msg = self.d_msg * eta_msg + (1 - self.d_msg) * eta_prev
                
                # Clip for numerical stability
                L_msg, eta_msg = _clip_pair(L_msg, eta_msg)
                
                self.m_v2f[(k, fid)] = (L_msg, eta_msg)
                
                if self._is_cross_robot(k, fn_obj): 
                    # 对称信息矩阵只需传递上/下三角+信息向量
                    # 每个浮点数8字节
                    matrix_elements = vn.dim * (vn.dim + 1) // 2  # 上三角元素数量
                    vector_elements = vn.dim                      # 向量元素数量
                    self.comm_bytes += 8 * (matrix_elements + vector_elements)

        # 2) Factor to Variable messages  
        for fn in self.f:
            # Collect current estimates
            μ_dict, Σ_dict = {}, {}
            for k in fn.vars:
                μ_dict[k] = self.v[k].mean()
                # Wrap angles for linearization
                for idx in self.v[k].angle_indices:
                    if 0 <= idx < len(μ_dict[k]):
                        μ_dict[k][idx] = wrap_angle(μ_dict[k][idx])
                
                _, Σ = info_to_gaussian(*self.v[k].belief())
                # Add small regularization for numerical stability
                Σ_dict[k] = make_pd(Σ, _MIN_PREC)

            # Linearize factor
            try:
                blocks = fn.fact.linearize(μ_dict, Σ_dict)
            except Exception as e:
                _log.error(f"Factor {fn.id} linearization failed: {e}")
                continue

            # Compute messages for each connected variable
            for tgt in fn.vars:
                vn = self.v[tgt]
                dim = vn.dim
                
                # Get blocks related to target variable
                Lkk, ηk = blocks.get(tgt, (np.zeros((dim, dim)), np.zeros(dim)))
                Lkk = make_pd(Lkk, _MIN_PREC)
                
                # Schur complement for other variables
                for oth in fn.vars:
                    if oth == tgt:
                        continue
                    
                    # Get cross terms
                    if (tgt, oth) in blocks:
                        Lkj = blocks[(tgt, oth)]
                    elif (oth, tgt) in blocks:
                        Lkj = blocks[(oth, tgt)].T
                    else:
                        continue
                    
                    # Get other variable's message and block
                    Lj_msg, ηj_msg = self.m_v2f.get((oth, fn.id),
                                                   (np.zeros((self.v[oth].dim,
                                                            self.v[oth].dim)),
                                                    np.zeros(self.v[oth].dim)))
                    
                    Ljj, ηj = blocks.get(oth, (np.zeros((self.v[oth].dim,
                                                        self.v[oth].dim)),
                                              np.zeros(self.v[oth].dim)))
                    
                    # Compute Schur complement
                    Ljj_total = make_pd(Ljj + Lj_msg, _MIN_PREC)
                    ηj_total = ηj + ηj_msg
                    
                    try:
                        Ljj_inv = stable_inv(Ljj_total)
                        # Update target blocks
                        Lkk -= Lkj @ Ljj_inv @ Lkj.T
                        ηk -= Lkj @ Ljj_inv @ ηj_total
                    except Exception as e:
                        _log.warning(f"Schur complement failed for factor {fn.id}: {e}")
                        continue
                
                # Ensure positive definite
                Lkk = make_pd(Lkk, _MIN_PREC)
                
                # Damping
                L_prev, η_prev = self.m_f2v.get((fn.id, tgt),
                                               (np.zeros_like(Lkk), np.zeros_like(ηk)))
                Lkk = self.d_msg * Lkk + (1 - self.d_msg) * L_prev
                ηk = self.d_msg * ηk + (1 - self.d_msg) * η_prev
                
                # Clip and store
                Lkk, ηk = _clip_pair(Lkk, ηk)
                self.m_f2v[(fn.id, tgt)] = (Lkk, ηk)
                
                if self._is_cross_robot(tgt, fn): 
                    dim = self.v[tgt].dim
                    matrix_elements = dim * (dim + 1) // 2
                    vector_elements = dim
                    self.comm_bytes += 8 * (matrix_elements + vector_elements)

        # 3) Belief updates
        max_delta = 0.0
        for k, vn in self.v.items():
            # Combine prior with all incoming messages
            L_new = vn.L_prior.copy()
            eta_new = vn.eta_prior.copy()
            
            for fid in vn.fids:
                L_msg, eta_msg = self.m_f2v.get((fid, k), 
                                               (np.zeros((vn.dim, vn.dim)),
                                                np.zeros(vn.dim)))
                L_new += L_msg
                eta_new += eta_msg
            
            # Small regularization for numerical stability
            L_new = make_pd(L_new, _MIN_PREC)
            
            # Update and track convergence
            delta = vn.update(L_new, eta_new)
            max_delta = max(max_delta, delta)
        
        self.max_delta_history.append(max_delta)
        return max_delta

    def run(self, max_iter: int = 100, tol: float = 1e-5, 
            verbose: bool = True) -> Tuple[List[float], Dict[str, Any]]:
            
        """Run GBP until convergence."""
        
        # Reset state
        self.m_v2f.clear()
        self.m_f2v.clear()
        for fn in self.f:
            fn.reset()
        self.max_delta_history.clear()
        self.comm_bytes = 0
        
        if verbose:
            _log.info(f"Starting GBP with {len(self.v)} variables, {len(self.f)} factors")
        
        for iteration in range(max_iter):
            delta = self._sweep()
            
            if verbose and (iteration % 10 == 0 or delta < tol):
                _log.info(f"Iter {iteration:3d} | max Δ = {delta:.2e}")
            
            # Check for numerical issues
            if not np.isfinite(delta) or delta > 1e8:
                _log.error(f"Numerical instability detected at iteration {iteration}")
                break
            
            # Check convergence
            if delta < tol:
                if verbose:
                    _log.info(f"Converged at iteration {iteration} (Δ = {delta:.2e})")
                break
        else:
            if verbose:
                _log.warning(f"Reached max iterations ({max_iter}) without full convergence")
        
        stats = {
            "iterations": len(self.max_delta_history),
            "converged": len(self.max_delta_history) > 0 and self.max_delta_history[-1] < tol,
            "comm_bytes": self.comm_bytes,
            "final_delta": self.max_delta_history[-1] if self.max_delta_history else float('inf')
        }
        return self.max_delta_history, stats
    
    # Backward compatibility alias
    def solve(self, *args, **kwargs):
        """Backward compatibility alias for run()."""
        return self.run(*args, **kwargs)

    # ─── Utility methods ───────────────────────────────────────
    def get_means(self) -> Dict[str, np.ndarray]:
        """Get current mean estimates for all variables."""
        return {k: vn.mean() for k, vn in self.v.items()}

    def get_marginals(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get marginal distributions (mean, covariance) for all variables."""
        return {k: info_to_gaussian(*vn.belief()) for k, vn in self.v.items()}
    
    def get_covariances(self) -> Dict[str, np.ndarray]:
        """Get covariance matrices for all variables."""
        return {k: info_to_gaussian(*vn.belief())[1] for k, vn in self.v.items()}
    
    def get_information_matrices(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get information form (precision matrix, information vector) for all variables."""
        return {k: vn.belief() for k, vn in self.v.items()}
    
    def set_prior(self, var_key: str, mean: np.ndarray, 
                  sigma: Union[float, np.ndarray]):
        """Update prior for a specific variable."""
        if var_key not in self.v:
            raise ValueError(f"Variable {var_key} not found")
        
        vn = self.v[var_key]
        if np.isscalar(sigma):
            L_prior = np.eye(vn.dim) / (sigma**2 + _EPS)
        else:
            sigma_arr = np.array(sigma)
            L_prior = np.diag(1.0 / (sigma_arr**2 + _EPS))
        
        eta_prior = L_prior @ mean
        vn.L_prior = make_pd(L_prior)
        vn.eta_prior = eta_prior
    
    def reset_messages(self):
        """Reset all messages (useful for re-running optimization)."""
        self.m_v2f.clear()
        self.m_f2v.clear()
        for fn in self.f:
            fn.reset()
    
    def get_variable_history(self, var_key: str) -> List[np.ndarray]:
        """Get optimization history for a specific variable."""
        if var_key not in self.v:
            raise ValueError(f"Variable {var_key} not found")
        return self.v[var_key].history.copy()
    
    def get_convergence_history(self) -> List[float]:
        """Get convergence history (max delta per iteration)."""
        return self.max_delta_history.copy()
    
    def get_comm_bytes(self) -> int:
        """获取累计通信字节数"""
        return self.comm_bytes

    def reset_comm_stats(self):
        """重置通信统计"""
        self.comm_bytes = 0