# -------------------------------------------------------------
# Gaussian Belief Propagation – Improved version for SLAM
# -------------------------------------------------------------
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from algorithm.frontend.factor_ut import ensure_positive_definite

import numpy as np
import scipy.linalg

# ─── Logging ────────────────────────────────────────────────
_log = logging.getLogger("GaBP")

# ─── Numerical constants ───────────────────────────────────
_EPS       = 1.0e-10
_DAMP_BEL  = 0.80          # belief-update damping
_DAMP_MSG  = 0.80          # message-update damping
_CLIP_SV   = 1.0e5         # singular-value clipping upper-bound
_MIN_PREC  = 1.0e-5       # minimum precision for numerical stability

# ─── Helper functions ──────────────────────────────────────
def wrap_angle(t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """ Wrap angle(s) to [-π, π]. """
    return np.arctan2(np.sin(t), np.cos(t)) if isinstance(t, np.ndarray) \
           else math.atan2(math.sin(t), math.cos(t))

def angle_diff(a1: float, a2: float) -> float:
    """Compute angular difference a1 - a2 wrapped to [-π, π]."""
    return wrap_angle(a1 - a2)

def is_pd(L: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if matrix is positive definite within a numerical tolerance."""
    min_eig = np.linalg.eigvalsh(L).min()
    return min_eig > -tol

def make_pd(A: np.ndarray, eps: float = 1e-8, clip_sv: float = 1e5) -> np.ndarray:
    """Ensure positive-definite by eigenvalue flooring and upper bound."""
    A = 0.5 * (A + A.T)
    try:
        w, Q = np.linalg.eigh(A)
        # 先上界clip，再下界floor
        w_clipped = np.clip(w, -clip_sv, clip_sv)
        w_safe = np.maximum(w_clipped, eps)
        # 检查有无修正
        if np.any(w < eps):
            # print(f"[make_pd] Info: eigenvalue corrected: {w} => {w_safe}")
            pass
        return Q @ np.diag(w_safe) @ Q.T
    except Exception:
        # 数值异常兜底
        dim = A.shape[0]
        return np.eye(dim) * eps


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
        U, s, Vt = np.linalg.svd(M_reg, full_matrices=False)
        s = np.clip(s, _MIN_PREC, _CLIP_SV)
        return Vt.T @ np.diag(1.0 / s) @ U.T
    
def info_to_gaussian(L: np.ndarray, eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert information form to moment form."""
    Σ = stable_inv(L)
    μ = Σ @ eta
    return μ, Σ


def _clip_pair(L: np.ndarray, eta: np.ndarray,
               s_max_lim: float = _CLIP_SV) -> Tuple[np.ndarray, np.ndarray]:
    if L.size == 0:
        return L, eta

    try:
        s_max = np.abs(np.linalg.eigvalsh(L)).max()
        if s_max <= s_max_lim:
            return L, eta

        w, Q = np.linalg.eigh(L)

        if np.abs(w).max() > s_max_lim:
            w_clipped = np.clip(w, -s_max_lim, s_max_lim)
            # 先 clip，再直接做 eigen-floor 保 PD
            w_safe = np.maximum(w_clipped, _EPS)
            L_clipped = Q @ np.diag(w_safe) @ Q.T            # 已 PD

            # μ = L⁻¹η, 用同一 (Q, w_safe)
            mu = Q @ ((Q.T @ eta) / w_safe)
            eta_new = L_clipped @ mu
            return L_clipped, eta_new
    except Exception:
        pass
    return L, eta


def safe_subtract_info(L1, eta1, L2, eta2):
    L_diff = L1 - L2
    eta_diff = eta1 - eta2

    try:
        if is_pd(L_diff, tol=1e-10):
            return L_diff, eta_diff  # Already PD
    except:
        pass

    # Project to PD
    w, Q = np.linalg.eigh(L_diff)
    w_clipped = np.maximum(w, _MIN_PREC)
    L_pd = Q @ np.diag(w_clipped) @ Q.T

    # Option 2: 重新同步η，保证 mean 不变
    # 先计算 mean: mu = L_diff^-1 @ eta_diff
    try:
        mu_tmp = stable_inv(L_pd) @ eta_diff
        eta_consistent = L_pd @ mu_tmp
        return L_pd, eta_consistent
    except Exception as e:
        # 兜底：数值出错时退回 Option 1
        print(f"[WARN] Option 2 η修正失败, fallback to Option 1: {e}")
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
        
        # Handle angle updates properly - 修正版本
        if self.angle_indices:
            for idx in self.angle_indices:
                if 0 <= idx < self.dim:
                    # 计算最短角度路径更新
                    raw_diff = mu_new[idx] - self.mu[idx]
                    wrapped_diff = wrap_angle(raw_diff)
                    mu_new[idx] = wrap_angle(self.mu[idx] + wrapped_diff)
            
            # 🔥 关键修复：角度wrap后重新同步η
            eta_damped = L_damped @ mu_new
        
        # Compute convergence metric
        delta = float(np.linalg.norm(mu_new - self.mu))
        
        # Update state
        self.mu_prev = self.mu.copy()
        self.mu = mu_new
        self.L = L_damped
        self.eta = eta_damped  # 现在η与wrapped μ保持一致
        self.history.append(self.mu.copy())
        
        return delta

# ─── Factor node ─────────────────────────────────────────────
class FacNode:
    def __init__(self, fid: int, factor: Any, mu: Dict[str, np.ndarray], cov: Dict[str, np.ndarray]):
        self.id   = fid
        self.fact = factor

        # 关键：直接用当前mu/cov调用linearize获取所有涉及变量的key
        blocks = factor.linearize(mu, cov)
        # 只保留str类型key，即所有单变量相关（支持tuple可扩展）
        self.vars = [k for k in blocks if isinstance(k, str)]
        self.prev_msg = {k: (np.zeros((factor._get_dim(k), factor._get_dim(k))),
                             np.zeros(factor._get_dim(k))) for k in self.vars}
        self.robot_set = {
            int(v.split('_')[0][1:]) for v in self.vars if v.startswith('x')
        }
        # print(f"Adapter type: {type(self.fact)}, raw type: {type(self.fact._f)}")


    def reset(self):
        for k in self.vars:
            dim = self.fact._get_dim(k)
            self.prev_msg[k] = (np.zeros((dim, dim)), np.zeros(dim))


def _extract_block_tuple(block, dim):
    if isinstance(block, tuple) and len(block) == 2:
        return block
    elif isinstance(block, np.ndarray):
        print(f"[WARN] Factor block 只返回矩阵，无eta: {block.shape}")
        return block, np.zeros(dim)
    else:
        print(f"[WARN] Factor block 类型异常: {type(block)}, 内容: {block}")
        return np.zeros((dim, dim)), np.zeros(dim)


# ─── GBP graph ─────────────────────────────────────────────
class GBPGraph:
    def __init__(self, factors: List[Any], variables: Dict[str, np.ndarray],
                 priors: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
                 angle_vars: Optional[Dict[str, List[int]]] = None,
                 bel_damp: float = _DAMP_BEL, msg_damp: float = _DAMP_MSG):

        self.d_bel, self.d_msg = float(bel_damp), float(msg_damp)
        self.v: Dict[str, VarNode] = {}
        for k, μ0 in variables.items():
            pm, ps = priors.get(k, (None, None)) if priors else (None, None)
            angle_idx = angle_vars.get(k) if angle_vars else None
            self.v[k] = VarNode(k, μ0.size, μ0, self.d_bel, pm, ps, angle_idx)

        # -- Patch here: FacNode正确收集变量 --
        self.f: List[FacNode] = []
        mu_init = {k: v.mean() for k, v in self.v.items()}
        cov_init = {k: np.eye(v.dim)*0.01 for k, v in self.v.items()}
        for fid, fac in enumerate(factors):
            fn = FacNode(fid, fac, mu_init, cov_init)
            if fn.vars:
                self.f.append(fn)
                for k in fn.vars:
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
        """从变量名提取机器人ID - 更鲁棒的版本"""
        # 支持多种命名约定
        patterns = [
            r'^x(\d+)_',    # x0_1, x1_2 等
            r'^robot(\d+)_', # robot0_pose 等
            r'^r(\d+)_',    # r0_1 等
        ]
        
        import re
        for pattern in patterns:
            match = re.match(pattern, var_name)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
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

                # ---- PATCH: 安全解包对角块 ----
                Lkk, ηk = _extract_block_tuple(blocks.get(tgt, None), dim)
                Lkk = make_pd(Lkk, _MIN_PREC)

                
                # Schur complement for other variables
                for oth in fn.vars:
                    if oth == tgt:
                        continue
                    
                    Lkj = None
                    if (tgt, oth) in blocks:
                        Lkj = blocks[(tgt, oth)]
                    elif (oth, tgt) in blocks:
                        Lkj = blocks[(oth, tgt)].T
                    if Lkj is None or not isinstance(Lkj, np.ndarray):
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
                        # 计算Schur complement项
                        schur_term = Lkj @ Ljj_inv @ Lkj.T
                        schur_eta = Lkj @ Ljj_inv @ ηj_total
                        
                       # —— 统一用 safe_subtract_info 做 “信息减法 + 投影” ——
                        Lkk, ηk = safe_subtract_info(Lkk, ηk, schur_term, schur_eta)

                    except Exception as e:
                        _log.warning(f"Schur complement failed for factor {fn.id}: {e}")
                        pass
                
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
        
        comm_bytes_this_run = 0
        
        if verbose:
            _log.info(f"Starting GBP with {len(self.v)} variables, {len(self.f)} factors")
        
        prev_delta = float('inf')
        for iteration in range(max_iter):
            old_comm = self.comm_bytes
            delta = self._sweep()
            comm_bytes_this_run += (self.comm_bytes - old_comm)
            
            if verbose and (iteration % 10 == 0 or delta < tol):
                _log.info(f"Iter {iteration:3d} | max Δ = {delta:.2e}")
                
            # ── ③-A 自适应阻尼 ──────────────────────────────────
            # 目标：如果震荡 / 发散⇒减小 damping；快速收敛⇒适度增大
            if delta > prev_delta * 1.2:       # 明显反弹
                self.d_msg *= 0.5
                self.d_bel *= 0.5
            elif delta < prev_delta * 0.5:     # 收敛速度很快
                self.d_msg = min(self.d_msg * 1.1, 0.8)
                self.d_bel = min(self.d_bel * 1.1, 0.8)
            prev_delta = delta
            # ────────────────────────────────────────────────
            
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
            "comm_bytes": comm_bytes_this_run,  # 只报告本次运行的通信量
            "total_comm_bytes": self.comm_bytes,  # 总累积通信量
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
        # 🔥 修复：同步重置通信统计
        self.comm_bytes = 0
    
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