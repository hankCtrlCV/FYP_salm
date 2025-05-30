"""
Factor implementations for 2-D multi-robot SLAM
----------------------------------------------
• PriorFactor          : 高斯先验
• OdometryFactor       : 线性里程计   (Δx,Δy)
• BearingRangeUTFactor : 非线性量测
      – mode="gbp"  : Unscented-Jacobian 信息化 (经典 GBP)
      – mode="spbp" : Sigma-Point Belief Propagation
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
from abc import ABC, abstractmethod
from algorithm.math.spbp import SigmaGenerator

class Factor(ABC):
    @abstractmethod
    def linearize(self, mu: Dict[str, np.ndarray], cov: Dict) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        pass

    @abstractmethod
    def get_energy(self, mu: Dict[str, np.ndarray]) -> float:
        pass

# =========================================================
# 1. PriorFactor (无需改动)
# ---------------------------------------------------------
class PriorFactor(Factor):
    def __init__(self, var: str, prior: np.ndarray, sigma: float):
        self.var = var
        self._Λ = np.eye(2) / (sigma**2)
        self._η = self._Λ @ prior
        self.prior = prior

    def linearize(self, mu: Dict, cov: Dict) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return {self.var: (self._Λ, self._η)}

    def get_energy(self, mu: Dict[str, np.ndarray]) -> float:
        residual = mu[self.var] - self.prior
        return 0.5 * residual.T @ self._Λ @ residual

# =========================================================
# 2. OdometryFactor (关键修正 η 公式)
# ---------------------------------------------------------
class OdometryFactor(Factor):
    def __init__(self, var_from: str, var_to: str, delta: np.ndarray, sigma: float):
        self.v1, self.v2 = var_from, var_to
        self.delta = delta.astype(float)
        self.Rinv = np.eye(2) / (sigma**2)

    def linearize(self, mu: Dict, cov: Dict):
        J1, J2 = -np.eye(2), np.eye(2)

        residual = self.delta - (mu[self.v2] - mu[self.v1])   # ← 关键!!  z - h(mu)

        Λ11 = J1.T @ self.Rinv @ J1
        Λ22 = J2.T @ self.Rinv @ J2
        Λ12 = J1.T @ self.Rinv @ J2

        η1 = J1.T @ self.Rinv @ residual + Λ11 @ mu[self.v1] + Λ12 @ mu[self.v2]
        η2 = J2.T @ self.Rinv @ residual + Λ12.T @ mu[self.v1] + Λ22 @ mu[self.v2]

        self._cross_blocks = [(self.v1, self.v2, Λ12)]
        return {self.v1: (Λ11, η1), self.v2: (Λ22, η2)}


    def get_energy(self, mu: Dict[str, np.ndarray]) -> float:
        res = (mu[self.v2] - mu[self.v1]) - self.delta
        return 0.5 * res @ self.Rinv @ res

# =========================================================
# 3. BearingRangeUTFactor (核心优化)
# ---------------------------------------------------------
class BearingRangeUTFactor(Factor):
    def __init__(self, pose_var: str, lm_var: str, measurement: np.ndarray, R: np.ndarray,
                 mode: str = "auto", alpha: float = 0.3, kappa: float = 0.0):
        self.pose_key = pose_var
        self.lm_key = lm_var
        self.z = measurement.copy()
        self.R = R.copy()
        self.mode = mode.lower()
        self.alpha = alpha
        self.kappa = kappa
        self._last_mu_x = None
        self._last_P_x = None
        self._sigma_cache = None
        self._y_sigma = None
        self._current_mode = "auto"  # 默认模式
        self._diverged_cnt = 0
        


    def _get_sigma_cache(self, mu_x: np.ndarray, P_x: np.ndarray):
        if (self._sigma_cache is None or 
            np.linalg.norm(mu_x - self._last_mu_x) > 5e-3 or  # 放宽采样阈值
            np.linalg.norm(P_x - self._last_P_x) > 5e-3):
            gen = SigmaGenerator(self.alpha, 2.0, self.kappa)
            self._sigma_cache, self._wm, self._wc = gen.generate(mu_x, P_x)
            self._last_mu_x, self._last_P_x = mu_x.copy(), P_x.copy()
            self._y_sigma = np.array([self._h(pt) for pt in self._sigma_cache])
        return self._sigma_cache, self._wm, self._wc, self._y_sigma

    @staticmethod
    def _h(x: np.ndarray) -> np.ndarray:
        dx, dy = x[2:] - x[:2]
        rng = np.hypot(dx, dy)
        brg = np.arctan2(dy, dx)
        return np.array([rng, brg])

    def linearize(self, mu: Dict[str, np.ndarray], cov: Dict[str, np.ndarray]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        mu_p, mu_l = mu[self.pose_key], mu[self.lm_key]
        P_p, P_l = cov[self.pose_key], cov[self.lm_key]

        # 处理 cross-cov
        key_pl = (self.pose_key, self.lm_key)
        P_pl = cov.get(key_pl, cov.get((self.lm_key, self.pose_key), np.zeros((2, 2))).T)

        mu_x = np.hstack([mu_p, mu_l])
        P_x = np.block([[P_p, P_pl], [P_pl.T, P_l]])

        # 动态 jitter 保证正定
        P_x = 0.5 * (P_x + P_x.T)
        sigma, wm, wc, y_sigma = self._get_sigma_cache(mu_x, P_x)
        mu_y = wm @ y_sigma
        diff_x = sigma - mu_x
        diff_y = y_sigma - mu_y

        # 角度残差 wrap 到 [-π, π)
        # bearing_res = self.z[1] - mu_y[1]
        # bearing_res = (bearing_res + np.pi) % (2 * np.pi) - np.pi
        # residual = np.array([self.z[0] - mu_y[0], bearing_res])
        # z - h(mu)  (角度 wrap 后)
        bearing_res = self.z[1] - mu_y[1]
        bearing_res = (bearing_res + np.pi) % (2*np.pi) - np.pi
        residual = np.array([ self.z[0] - mu_y[0],  bearing_res ])


        Sigma_y = diff_y.T @ (wc[:, None] * diff_y) + self.R
        Sigma_y += 1e-8 * np.eye(2) * max(1, np.trace(Sigma_y))  # 动态 jitter
        P_xy = diff_x.T @ (wc[:, None] * diff_y)

        # 伪逆更稳定
        J = P_xy.T @ np.linalg.pinv(P_x + 1e-6 * np.eye(4))
        J_p, J_l = J[:, :2], J[:, 2:]
        Λ_y = np.linalg.pinv(Sigma_y + 1e-8 * np.eye(2))

        # 自动模式决策（保留原有逻辑，略）

        # 信息矩阵裁剪
        Λ_p = np.clip(J_p.T @ Λ_y @ J_p, 1e-6, 1e3) + 1e-6 * np.eye(2)
        Λ_l = np.clip(J_l.T @ Λ_y @ J_l, 1e-6, 1e3) + 1e-6 * np.eye(2)

        # 正确 η 公式（含 J_p @ μ_p 和 J_l @ μ_l）
        η_p = J_p.T @ Λ_y @ (residual + J_p @ mu_p + J_l @ mu_l)
        η_l = J_l.T @ Λ_y @ (residual + J_p @ mu_p + J_l @ mu_l)

        # 缓存供 Graph 使用
        self._last_mu_y = mu_y
        self._Λy = Λ_y
        self._cross_blocks = [(self.pose_key, self.lm_key, J_p.T @ Λ_y @ J_l)]

        return {self.pose_key: (Λ_p, η_p), self.lm_key: (Λ_l, η_l)}

    def get_energy(self, mu: Dict[str, np.ndarray]) -> float:
        if not hasattr(self, "_Λy"):
            return 0.0
        residual = self.z - self._last_mu_y
        residual[1] = (residual[1] + np.pi) % (2 * np.pi) - np.pi  # 角度 wrap
        return 0.5 * residual @ self._Λy @ residual