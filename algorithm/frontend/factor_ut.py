"""Factor implementations for 3-D multi-robot SLAM (x, y, θ)"""

from __future__ import annotations
import numpy as np
import math
import threading
from typing import Dict, Tuple, Union, Any, Optional
from abc import ABC, abstractmethod

# 共用常量
_EPS = 1e-12
_SMALL_NUMBER = 1e-9

def wrap_angle(theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """角度归一化到[-π, π)，返回类型与输入保持一致"""
    if isinstance(theta, np.ndarray):
        return np.arctan2(np.sin(theta), np.cos(theta))
    else:
        # 标量时返回 Python float 类型
        return float(np.arctan2(np.sin(theta), np.cos(theta)))

# =====================================================================
# Sigma点生成器 (UT算法核心)
# =====================================================================
class SigmaGenerator:
    """Generate sigma points for Unscented Transform"""
    def __init__(self, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
    def generate(self, mu: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = mu.size
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        gamma = math.sqrt(n + lambda_)
        
        # 协方差矩阵平方根
        try:
            S = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            cov_reg = cov + np.eye(n) * _SMALL_NUMBER
            S = np.linalg.cholesky(cov_reg)
        
        # 生成sigma点
        sigma_points = np.zeros((2*n + 1, n))
        sigma_points[0] = mu
        
        for i in range(n):
            sigma_points[i+1] = mu + gamma * S[:, i]
            sigma_points[n+i+1] = mu - gamma * S[:, i]
        
        # 计算权重
        wm = np.full(2*n+1, 0.5/(n+lambda_))
        wc = wm.copy()
        
        wm[0] = lambda_/(n+lambda_)
        # 按照要求修正权重计算
        wc[0] = wm[0] + (1 - self.alpha**2 + self.beta)
        
        # 权重归一化
        wm /= wm.sum()
        wc /= wc.sum()
        
        return sigma_points, wm, wc

# =====================================================================
# 基类
# =====================================================================
class Factor(ABC):
    @abstractmethod
    def linearize(self, mu: Dict[str, np.ndarray], cov: Dict[str, np.ndarray]) -> Dict: ...
    
    @abstractmethod
    def get_energy(self, mu: Dict[str, np.ndarray]) -> float: ...

# =====================================================================
# 1. Prior factor
# =====================================================================
class PriorFactor(Factor):
    def __init__(self, var: str, prior: np.ndarray, sigmas: np.ndarray):
        self.var = var
        self.prior = prior.astype(float)
        sigma_array = np.ones_like(prior) * sigmas if np.isscalar(sigmas) else sigmas
        self._Λ = np.diag(1.0 / (sigma_array ** 2))
        self._η = self._Λ @ self.prior

    def linearize(self, *_) -> Dict:
        return {self.var: (self._Λ, self._η)}

    def get_energy(self, mu: Dict[str, np.ndarray]) -> float:
        r = mu[self.var] - self.prior
        if r.size == 3: 
            r[2] = wrap_angle(r[2])
        return 0.5 * r @ self._Λ @ r
        
    def _get_dim(self, key: str) -> int:
        return self.prior.shape[0] if key == self.var else 0

# =====================================================================
# 2. Odometry factor (关键修正版)
# =====================================================================
class OdometryFactor(Factor):
    def __init__(self, v_from: str, v_to: str, delta: np.ndarray, sigmas: np.ndarray):
        self.v1, self.v2 = v_from, v_to
        self.delta = delta.astype(float)
        
        # 预计算逆变换的Adjoint矩阵
        dx, dy, th = self.delta
        inv_adjoint = self._adjoint(dx, dy, th, inverse=True)
        self.Ji_adjust = -inv_adjoint  # J_i = -Adjoint(Δ⁻¹)
        
        sigma_array = np.ones_like(delta) * sigmas if np.isscalar(sigmas) else sigmas
        self.Rinv = np.diag(1.0 / (sigma_array ** 2))
    
    @staticmethod
    def _se2_minus(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """SE(2)上的右乘误差: Log(b ∘ a⁻¹)"""
        # 提取位姿分量
        x_a, y_a, θ_a = a
        x_b, y_b, θ_b = b
        
        # 计算相对旋转
        dθ = wrap_angle(θ_b - θ_a)
        
        # 计算旋转矩阵的转置 (即逆旋转)
        cθ, sθ = math.cos(θ_a), math.sin(θ_a)
        R_T = np.array([[cθ, sθ], [-sθ, cθ]])
        
        # 计算在a坐标系下的平移
        trans_global = np.array([x_b - x_a, y_b - y_a])
        trans_body = R_T @ trans_global
        
        return np.array([trans_body[0], trans_body[1], dθ])
    
    @staticmethod
    def _adjoint(x: float, y: float, th: float, inverse=False) -> np.ndarray:
        """完整SE(2) Adjoint矩阵(含平移分量)"""
        ct, st = math.cos(th), math.sin(th)
        if inverse:
            # Adjoint(Δ⁻¹)的精确计算
            return np.array([
                [ct, st, -y*ct + x*st],
                [-st, ct, -x*ct - y*st],
                [0, 0, 1]
            ])
        else:
            # 标准Adjoint
            return np.array([
                [ct, st, -st*y + ct*x],
                [-st, ct, -ct*x - st*y],
                [0, 0, 1]
            ])

    def linearize(self, mu: Dict, cov: Dict) -> Dict:
        Ti, Tj = mu[self.v1], mu[self.v2]
        
        # 计算误差: e = Log(Δ⁻¹ ∘ (T_i⁻¹ ∘ T_j))
        Tij = self._se2_minus(Ti, Tj)
        err = self._se2_minus(self.delta, Tij)
        err[2] = wrap_angle(err[2])

        # 左雅可比矩阵
        J_i = self.Ji_adjust
        J_j = np.eye(3)

        # 信息矩阵部分
        Λii = J_i.T @ self.Rinv @ J_i
        Λjj = J_j.T @ self.Rinv @ J_j
        Λij = J_i.T @ self.Rinv @ J_j

        # 信息向量部分
        ηi = -J_i.T @ self.Rinv @ err
        ηj = -J_j.T @ self.Rinv @ err

        # 返回结果并添加对称块
        result = {
            self.v1: (Λii, ηi),
            self.v2: (Λjj, ηj),
            (self.v1, self.v2): Λij
        }
        # 确保对称性
        result[(self.v2, self.v1)] = Λij.T
        return result

    def get_energy(self, mu: Dict[str, np.ndarray]) -> float:
        """保持与linearize一致的误差计算"""
        Ti, Tj = mu[self.v1], mu[self.v2]
        Tij = self._se2_minus(Ti, Tj)
        err = self._se2_minus(self.delta, Tij)
        err[2] = wrap_angle(err[2])
        return 0.5 * err @ self.Rinv @ err
        
    def _get_dim(self, key: str) -> int:
        return 3 if key in (self.v1, self.v2) else 0

# =====================================================================
# 3. Bearing-Range UT 因子 (关键修正版)
# =====================================================================
class BearingRangeUTFactor(Factor):
    def __init__(self, pose_var: str, lm_var: str, measurement: np.ndarray, R: np.ndarray,
                 mode: str = "auto", alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0,
                 linear_threshold: float = 5.0, residual_sigma_thresh: float = 3.0):
        self.pose_key = pose_var
        self.lm_key = lm_var
        self.z = measurement.astype(float)
        self.R = R.copy()
        self.Rinv = np.linalg.inv(self.R)
        self.mode = mode.lower()
        self.alpha, self.beta, self.kappa = alpha, beta, kappa
        self.linear_th = linear_threshold
        self.res_th_mul = residual_sigma_thresh
        self._diverged_cnt = 0
        self._Λy: Optional[np.ndarray] = None
        self._last_Λ_cross: Optional[np.ndarray] = None
        self._lock = threading.RLock()

    def linearize(self, mu: Dict, cov: Dict) -> Dict:
        with self._lock:
            μp = mu[self.pose_key]
            μl = mu[self.lm_key]
            Pp = cov[self.pose_key]
            Pl = cov[self.lm_key]
            Ppl = cov.get((self.pose_key, self.lm_key), np.zeros((3, 2)))
            μx = np.hstack([μp, μl])
            Px = np.block([[Pp, Ppl], [Ppl.T, Pl]]) + np.eye(5) * _SMALL_NUMBER

            # 计算残差（带角度归一化）
            y_pred = self._h(μx)
            r = self.z - y_pred
            r[1] = wrap_angle(r[1])

            # 计算距离（带安全阈值）
            dx, dy = μl[0] - μp[0], μl[1] - μp[1]
            dist = math.hypot(dx, dy)
            safe_dist = max(dist, _SMALL_NUMBER)

            # 模式选择
            mode_now = self._decide_mode(safe_dist, r)
            
            # 根据模式选择线性化方法
            if mode_now == "spbp":
                try:
                    return self._spbp_linearize(μx, Px, r)
                except np.linalg.LinAlgError:
                    # 奇异矩阵时回落到GBP
                    return self._gbp_linearize(μp, μl, r, safe_dist, dx, dy)
            else:
                return self._gbp_linearize(μp, μl, r, safe_dist, dx, dy)

    def get_energy(self, mu: Dict[str, np.ndarray]) -> float:
        with self._lock:
            if self._Λy is None:
                return 0.0
            μx = np.hstack([mu[self.pose_key], mu[self.lm_key]])
            pred = self._h(μx)
            r = self.z - pred
            r[1] = wrap_angle(r[1])
            return 0.5 * r @ self._Λy @ r

    def cross_block(self, var_i: str, var_j: str) -> np.ndarray:
        with self._lock:
            # 处理非本因子变量的安全访问
            if not (self._is_related_var(var_i) and self._is_related_var(var_j)):
                return np.zeros((self._get_dim(var_i), self._get_dim(var_j)))
                
            if self._last_Λ_cross is None:
                return np.zeros((self._get_dim(var_i), self._get_dim(var_j)))
                
            if (var_i, var_j) == (self.pose_key, self.lm_key):
                return self._last_Λ_cross
            elif (var_i, var_j) == (self.lm_key, self.pose_key):
                return self._last_Λ_cross.T
            
            return np.zeros((self._get_dim(var_i), self._get_dim(var_j)))
            
    def _is_related_var(self, var: str) -> bool:
        """检查变量是否与本因子相关"""
        return var in (self.pose_key, self.lm_key)

    # ----------- 内部方法 -----------
    @staticmethod
    def _h(x: np.ndarray) -> np.ndarray:
        px, py, th, lx, ly = x
        dx, dy = lx - px, ly - py
        rng = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx) - th
        return np.array([rng, wrap_angle(bearing)])

    def _decide_mode(self, dist: float, residual: np.ndarray) -> str:
        if self.mode in ("gbp", "spbp"):
            return self.mode
            
        σ_r, σ_b = math.sqrt(self.R[0, 0]), math.sqrt(self.R[1, 1])
        big_r = abs(residual[0]) > self.res_th_mul * σ_r
        big_b = abs(residual[1]) > self.res_th_mul * σ_b
        
        if dist > self.linear_th or big_r or big_b or self._diverged_cnt > 3:
            self._diverged_cnt = 0
            return "spbp"
            
        self._diverged_cnt = (self._diverged_cnt + 1) if (big_r or big_b) else 0
        return "gbp"

    def _gbp_linearize(self, μp: np.ndarray, μl: np.ndarray, residual: np.ndarray, 
                       dist: float, dx: float, dy: float) -> Dict:
        # 安全处理距离和增量
        safe_dist = max(dist, _SMALL_NUMBER)
        q = safe_dist ** 2
        
        # 处理小值问题
        safe_dx = dx if abs(dx) > _EPS else _EPS * (1 if dx >= 0 else -1)
        safe_dy = dy if abs(dy) > _EPS else _EPS * (1 if dy >= 0 else -1)

        # Jacobian矩阵 (关键修正：姿态角对bearing残差梯度位置)
        Jp = np.array([
            [-safe_dx/safe_dist, -safe_dy/safe_dist, 0.0],
            [safe_dy/q, -safe_dx/q, -1.0]  # 第三列设为-1
        ])
        Jl = np.array([
            [safe_dx/safe_dist, safe_dy/safe_dist],
            [-safe_dy/q, safe_dx/q]
        ])

        Λp = Jp.T @ self.Rinv @ Jp
        Λl = Jl.T @ self.Rinv @ Jl
        Λcross = Jp.T @ self.Rinv @ Jl

        # 信息向量
        r_col = residual.reshape(2, 1)
        ηp = Jp.T @ self.Rinv @ r_col - (Λp @ μp.reshape(3, 1) + Λcross @ μl.reshape(2, 1))
        ηl = Jl.T @ self.Rinv @ r_col - (Λl @ μl.reshape(2, 1) + Λcross.T @ μp.reshape(3, 1))

        # 缓存结果
        self._Λy = self.Rinv
        self._last_Λ_cross = Λcross.copy()
        
        # 返回结果并添加对称块
        result = {
            self.pose_key: (Λp, ηp.flatten()),
            self.lm_key: (Λl, ηl.flatten()),
            (self.pose_key, self.lm_key): Λcross
        }
        result[(self.lm_key, self.pose_key)] = Λcross.T
        return result

    def _spbp_linearize(self, μx: np.ndarray, Px: np.ndarray, residual: np.ndarray) -> Dict:
        gen = SigmaGenerator(self.alpha, self.beta, self.kappa)
        χ, wm, wc = gen.generate(μx, Px)
        
        # 计算sigma点的观测
        yσ = np.zeros((len(χ), 2))
        for i, p in enumerate(χ):
            yσ[i] = self._h(p)
        
        # 计算均值 - 特殊处理角度
        μy_range = np.sum(wm * yσ[:, 0])
        sin_sum = np.sum(wm * np.sin(yσ[:, 1]))
        cos_sum = np.sum(wm * np.cos(yσ[:, 1]))
        μy_bearing = math.atan2(sin_sum, cos_sum)
        μy = np.array([μy_range, μy_bearing])
        
        # 计算残差 (注意角度归一化)
        r = self.z - μy
        r[1] = wrap_angle(r[1])
        
        # 计算协方差
        diff_x = χ - μx
        diff_y = yσ - μy
        diff_y[:, 1] = wrap_angle(diff_y[:, 1])
        
        Sy = np.zeros((2, 2))
        Pxy = np.zeros((5, 2))
        for k in range(len(χ)):
            Sy += wc[k] * np.outer(diff_y[k], diff_y[k])
            Pxy += wc[k] * np.outer(diff_x[k], diff_y[k])
        Sy += self.R  # 添加观测噪声
        
        # 正则化Sy
        Sy_reg = Sy + np.eye(2) * _SMALL_NUMBER
        try:
            Sinv = np.linalg.inv(Sy_reg)
        except np.linalg.LinAlgError:
            Sy_reg += np.eye(2) * _SMALL_NUMBER
            Sinv = np.linalg.inv(Sy_reg)
        
        # 卡尔曼增益
        K = Pxy @ Sinv
        
        # 后验协方差
        Σ_post = Px - K @ Sy @ K.T
        Σ_post = 0.5 * (Σ_post + Σ_post.T) + np.eye(5) * _SMALL_NUMBER
        
        # 信息矩阵
        Λ = np.linalg.inv(Σ_post)
        Λ = 0.5 * (Λ + Λ.T)  # 确保对称
        
        # 信息向量
        η_joint = Λ @ (μx + K @ r)  # 后验均值
        
        # 分割结果
        ηp = η_joint[:3]
        ηl = η_joint[3:]
        Λp = Λ[:3, :3]
        Λl = Λ[3:, 3:]
        Λcross = Λ[:3, 3:]
        
        # 线程安全缓存
        with self._lock:
            self._last_μx = μx.copy()
            self._last_Λy = Sinv.copy()
            self._last_Λ_cross = Λcross.copy()
        
        # 返回结果
        return {
            self.pose_key: (Λp, ηp),
            self.lm_key: (Λl, ηl),
            (self.pose_key, self.lm_key): Λcross,
            (self.lm_key, self.pose_key): Λcross.T
        }
    
    def get_energy(self, mu: Dict[str, np.ndarray]) -> float:
        with self._lock:
            if self._last_μx is None or self._last_Λy is None:
                return 0.0
            
            # 使用缓存的μx和Λy计算能量
            μx_current = np.hstack([mu[self.pose_key], mu[self.lm_key]])
            δx = μx_current - self._last_μx
            
            # 计算预测观测
            pred = self._h(μx_current)
            r = self.z - pred
            r[1] = wrap_angle(r[1])
            
            # 使用缓存的Λy
            return 0.5 * r @ self._last_Λy @ r

    def _get_dim(self, key: str) -> int:
        """安全处理变量维度，避免reshape(0,0)问题"""
        if key == self.pose_key:
            return 3
        elif key == self.lm_key:
            return 2
        else:
            return 0  # 非本因子变量返回0