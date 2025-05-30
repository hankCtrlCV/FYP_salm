"""
Sigma-Point Belief Propagation utilities
---------------------------------------
• SigmaGenerator        : 生成 2 n+1 σ-点及权重（增强 SPD 保障）
• unscented_transform   : 返回 μ_y, Σ_y, 以及 cross-cov P_xy
                           （SPBP 需要横协方差 Σ_xy）
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Tuple

# =========================================================
# 1. Sigma-point 生成器  ——  数值稳健版
# =========================================================
class SigmaGenerator:
    """
    参数
    ----
    alpha, beta, kappa : UKF 常用参数
    jitter_init        : Cholesky 首次尝试的抖动
    jitter_factor      : 每失败一次 *factor
    jitter_max         : 允许的最大 jitter
    eig_floor          : SPD 修正时的最小特征值
    """

    def __init__(
        self,
        alpha: float = 1e-3,
        beta : float = 2.0,
        kappa: float = 0.0,
        *,
        jitter_init : float = 1e-10,
        jitter_factor: float = 10.0,
        jitter_max  : float = 1e-1,
        eig_floor   : float = 1e-10,
    ):
        self.alpha  = alpha
        self.beta   = beta
        self.kappa  = kappa
        self.j0     = jitter_init
        self.j_fac  = jitter_factor
        self.j_max  = jitter_max
        self.eig_floor = eig_floor

        self._last_jitter = jitter_init          # 记忆上次成功的 jitter

    # ----------  SPD 修复 ----------
    def _make_spd(self, M: np.ndarray) -> np.ndarray:
        """对称化 + 提升最小特征值到 eig_floor"""
        M = 0.5 * (M + M.T)
        eigval, eigvec = np.linalg.eigh(M)
        if eigval.min() < self.eig_floor:
            eigval = np.clip(eigval, self.eig_floor, None)
            M = eigvec @ np.diag(eigval) @ eigvec.T
            M = 0.5 * (M + M.T)                 # 再次对称
        return M

    # ----------  σ-point 生成 ----------
    def generate(
        self, mean: np.ndarray, cov: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        n = mean.size
        lam = self.alpha**2 * (n + self.kappa) - n
        scale = np.sqrt(n + lam)

        # ---- ① SPD 防护 ----
        cov = self._make_spd(cov)

        # ---- ② Cholesky（带自适应 jitter）----
        eye = np.eye(n)
        jitter = self._last_jitter
        for _ in range(12):                     # 最多 12 次
            try:
                sqrt_mat = np.linalg.cholesky(cov + jitter * eye)
                self._last_jitter = jitter
                break
            except np.linalg.LinAlgError:
                jitter *= self.j_fac
                if jitter > self.j_max:
                    raise RuntimeError(
                        f"Cholesky failed: jitter>{self.j_max:.1e}"
                    ) from None

        # ---- ③ σ-points ----
        pts = [mean]
        for i in range(n):
            col = scale * sqrt_mat[:, i]
            pts += [mean + col, mean - col]

        wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
        wc = wm.copy()
        wm[0] = lam / (n + lam)
        wc[0] = wm[0] + (1 - self.alpha**2 + self.beta)

        return np.stack(pts), wm, wc


# =========================================================
# 2. Unscented Transform  (带 cross-cov)
# =========================================================
def unscented_transform(
    f: Callable[[np.ndarray], np.ndarray],
    mu: np.ndarray,
    P:  np.ndarray,
    R:  np.ndarray,
    *,
    alpha: float = 1e-3,
    beta : float = 2.0,
    kappa: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    f   : 非线性函数  R^n → R^m
    mu  : (n,)
    P   : (n,n)  输入协方差
    R   : (m,m)  观测噪声协方差
    返回
    ----
    μ_y  : (m,)
    Σ_y  : (m,m)
    P_xy : (n,m)   state-to-measurement cross-cov
    """
    gen = SigmaGenerator(alpha, beta, kappa)
    sigma, wm, wc = gen.generate(mu, P)

    # ---- σ-points through f ----
    y_sigma = np.asarray([f(pt) for pt in sigma])      # (2n+1, m)

    # ---- 输出均值 ----
    mu_y = wm @ y_sigma

    # ---- 协方差 & cross-cov ----
    diff_x = sigma - mu            # (2n+1, n)
    diff_y = y_sigma - mu_y        # (2n+1, m)

    Σ_y  = diff_y.T @ (wc[:, None] * diff_y) + R
    P_xy = diff_x.T @ (wc[:, None] * diff_y)           # (n,m)

    # 轻微对称化 & 正则
    Σ_y = 0.5 * (Σ_y + Σ_y.T) + 1e-12 * np.eye(Σ_y.shape[0])

    return mu_y, Σ_y, P_xy
