"""Factor implementations for 3-D multi-robot SLAM (x, y, θ)"""

from __future__ import annotations
import numpy as np
import math
import threading
from typing import Dict, Tuple, Union, Any, Optional
from abc import ABC, abstractmethod
import scipy.linalg

# 共用常量
_EPS = 1e-12
_SMALL_NUMBER = 1e-9

def wrap_angle(theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    if isinstance(theta, np.ndarray):
        # 使用模运算确保精确的[-π, π)范围
        return np.mod(theta + np.pi, 2 * np.pi) - np.pi
    else:
        # 标量版本处理π边界
        wrapped = (theta + math.pi) % (2 * math.pi) - math.pi
        return wrapped if wrapped != math.pi else -math.pi  # 处理π的边界情况

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
        
        # 移除一次性计算的雅可比矩阵
        sigma_array = np.ones_like(delta) * sigmas if np.isscalar(sigmas) else sigmas
        self.Rinv = np.diag(1.0 / (sigma_array ** 2))
    
    @staticmethod
    def _se2_minus(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """SE(2)上的右乘误差: Log(b ∘ a⁻¹)"""
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
    def _adjoint(x: float, y: float, th: float) -> np.ndarray:
        """完整SE(2) Adjoint矩阵(含平移分量) - 修正版"""
        ct, st = math.cos(th), math.sin(th)
        # 修正：根据图片建议的正确形式
        return np.array([
            [ct, -st, y],
            [st,  ct, -x],
            [0,   0,  1]
        ])

    def linearize(self, mu: Dict, cov: Dict) -> Dict:
        Ti, Tj = mu[self.v1], mu[self.v2]
        
        # 修正1：正确计算相对位姿误差
        Tij = self._se2_minus(Ti, Tj)  # 计算 Tj ∘ Ti⁻¹
        # 修正2：交换参数顺序 Log(Δ⁻¹ ∘ (T_i⁻¹ ∘ T_j))
        err = self._se2_minus(self.delta, Tij)  # 修正顺序 
        err[2] = wrap_angle(err[2])
        
        # 修正3：使用相对位姿计算雅可比
        J_i = -self._adjoint(*Tij)  # Adj(Tij)
        J_j = np.eye(3)
        
        # 信息矩阵部分
        Λii = J_i.T @ self.Rinv @ J_i
        Λjj = J_j.T @ self.Rinv @ J_j
        Λij = J_i.T @ self.Rinv @ J_j
        
        # 信息向量部分
        ηi = J_i.T @ self.Rinv @ err
        ηj = J_j.T @ self.Rinv @ err

        # 返回结果并添加对称块
        result = {
            self.v1: (Λii, ηi),
            self.v2: (Λjj, ηj),
            (self.v1, self.v2): Λij
        }
        result[(self.v2, self.v1)] = Λij.T
        return result

    def get_energy(self, mu: Dict[str, np.ndarray]) -> float:
        Ti, Tj = mu[self.v1], mu[self.v2]
        Tij = self._se2_minus(Ti, Tj)
        # 修正：与linearize保持一致的误差计算顺序
        err = self._se2_minus(self.delta, Tij)
        err[2] = wrap_angle(err[2])
        return 0.5 * err @ self.Rinv @ err
        
    def _get_dim(self, key: str) -> int:
        return 3 if key in (self.v1, self.v2) else 0

# =====================================================================
# 3. Bearing-Range UT 因子 
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
        self._lock = threading.RLock()
        self._last_μx = None
        self._last_Λ_cross = None  # 删除_last_Λy变量
        self._spbp_cache = {}
        self._current_mode = "gbp"
        # 维度映射表 - 确保返回零矩阵时维度正确
        self._dim_map = {
            self.pose_key: 3,
            self.lm_key: 2
        }

    def linearize(self, mu: Dict, cov: Dict) -> Dict:
        with self._lock:
            μp = mu[self.pose_key]
            μl = mu[self.lm_key]
            Pp = cov[self.pose_key]
            Pl = cov[self.lm_key]
            Ppl = cov.get((self.pose_key, self.lm_key), np.zeros((3, 2)))
            μx = np.hstack([μp, μl])
            Px = np.block([[Pp, Ppl], [Ppl.T, Pl]]) + np.eye(5) * _SMALL_NUMBER

            # 计算残差
            y_pred = self._h(μx)
            r = self.z - y_pred
            r[0] = wrap_angle(r[0])

            # 计算距离
            dx, dy = μl[0] - μp[0], μl[1] - μp[1]
            dist = math.hypot(dx, dy)
            safe_dist = max(dist, _SMALL_NUMBER)

            # 模式选择
            mode_now = self._decide_mode(safe_dist, r)
            
            # 设置通用属性
            self._last_μx = μx.copy()
            self._last_Λ_cross = None
            
            # 根据模式选择线性化方法
            if mode_now == "spbp":
                try:
                    # 生成缓存键 - 包含所有必要信息
                    cache_key = self._generate_cache_key(μx, Px)
                    result = self._spbp_linearize(μx, Px, r, cache_key)
                    return result
                except np.linalg.LinAlgError:
                    # 奇异矩阵时回落到GBP
                    result = self._gbp_linearize(μp, μl, r, safe_dist, dx, dy)
                    return result
            else:
                result = self._gbp_linearize(μp, μl, r, safe_dist, dx, dy)
                return result

    def get_energy(self, mu: Dict[str, np.ndarray]) -> float:
        with self._lock:
            try:
                μx_current = np.hstack([mu[self.pose_key], mu[self.lm_key]])
            except KeyError:
                return 0.0
                
            # 计算预测观测
            pred = self._h(μx_current)
            r = self.z - pred
            r[0] = wrap_angle(r[0])
            
            # 修正7：始终使用观测信息矩阵 R⁻¹
            return 0.5 * r @ self.Rinv @ r
        
    def cross_block(self, var_i: str, var_j: str) -> np.ndarray:
        with self._lock:
            # 获取安全维度（使用_dim_map确保维度一致）
            dim_i = self._dim_map.get(var_i, 0)
            dim_j = self._dim_map.get(var_j, 0)
            
            # 修正6：对任一非相关变量都返回零矩阵
            if not self._is_related_var(var_i) or not self._is_related_var(var_j):
                return np.zeros((dim_i, dim_j))
            
            if self._last_Λ_cross is None:
                return np.zeros((dim_i, dim_j))
                
            if (var_i, var_j) == (self.pose_key, self.lm_key):
                return self._last_Λ_cross
            elif (var_i, var_j) == (self.lm_key, self.pose_key):
                return self._last_Λ_cross.T
            
            # 修正6：确保对(var_i, var_j)不匹配也返回正确维度
            return np.zeros((dim_i, dim_j))
            
    def _is_related_var(self, var: str) -> bool:
        return var in (self.pose_key, self.lm_key)

    # ----------- 内部方法 -----------
    @staticmethod
    def _h(x: np.ndarray) -> np.ndarray:
        px, py, th, lx, ly = x
        dx, dy = lx - px, ly - py
        rng = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx) - th
        return np.array([wrap_angle(bearing), rng]) 

    def _decide_mode(self, dist: float, residual: np.ndarray) -> str:
        if self.mode in ("gbp", "spbp"):
            return self.mode
            
        # 极近距离强制使用GBP
        if dist < _SMALL_NUMBER:
            return "gbp"
            
        # 残差分析
        σ_bearing = math.sqrt(self.R[0, 0])
        σ_range = math.sqrt(self.R[1, 1])
        big_bearing = abs(residual[0]) > self.res_th_mul * σ_bearing
        big_range = abs(residual[1]) > self.res_th_mul * σ_range
        
        # 模式切换逻辑
        if self._current_mode == "spbp":
            if not (big_bearing or big_range) and dist < self.linear_th:
                self._diverged_cnt = max(0, self._diverged_cnt - 1)
                if self._diverged_cnt == 0:
                    self._current_mode = "gbp"
            else:
                return "spbp"
        else:
            if dist > self.linear_th or big_bearing or big_range:
                self._diverged_cnt += 1
                if self._diverged_cnt > 3:
                    self._current_mode = "spbp"
        
        return self._current_mode

    def _gbp_linearize(self, μp: np.ndarray, μl: np.ndarray, residual: np.ndarray, 
                       dist: float, dx: float, dy: float) -> Dict:
        # 安全处理距离和增量
        safe_dist = max(dist, _SMALL_NUMBER)
        q = safe_dist ** 2
        
        # 处理小值问题
        safe_dx = dx if abs(dx) > _EPS else _EPS * (1 if dx >= 0 else -1)
        safe_dy = dy if abs(dy) > _EPS else _EPS * (1 if dy >= 0 else -1)

        # Jacobian矩阵
        Jp_bearing = np.array([
            safe_dy/q, -safe_dx/q, -1.0
        ])
        Jl_bearing = np.array([
            -safe_dy/q, safe_dx/q
        ])
        Jp_range = np.array([
            -safe_dx/safe_dist, -safe_dy/safe_dist, 0.0
        ])
        Jl_range = np.array([
            safe_dx/safe_dist, safe_dy/safe_dist
        ])
        
        Jp = np.vstack([Jp_bearing, Jp_range])
        Jl = np.vstack([Jl_bearing, Jl_range])

        Λp = Jp.T @ self.Rinv @ Jp
        Λl = Jl.T @ self.Rinv @ Jl
        Λcross = Jp.T @ self.Rinv @ Jl

        # 信息向量 
        r_col = residual.reshape(2, 1)
        ηp = Jp.T @ self.Rinv @ r_col
        ηl = Jl.T @ self.Rinv @ r_col

        # 更新缓存
        self._last_Λ_cross = Λcross.copy()
        
        # 返回结果
        result = {
            self.pose_key: (Λp, ηp.flatten()),
            self.lm_key: (Λl, ηl.flatten()),
            (self.pose_key, self.lm_key): Λcross
        }
        result[(self.lm_key, self.pose_key)] = Λcross.T
        return result

    def _generate_cache_key(self, μx, Px):
        """生成稳定的缓存键"""
        return (
            tuple(np.round(μx, 8)),  # 四舍五入提高稳定性
            tuple(np.round(Px.flatten(), 8)),
            tuple(self.z)  # 包含量测值
        )

    def _spbp_linearize(self, μx: np.ndarray, Px: np.ndarray, residual: np.ndarray, 
                    cache_key: int) -> Dict:
        # 检查缓存
        if cache_key in self._spbp_cache:
            cache_data = self._spbp_cache[cache_key]
            with self._lock:
                self._last_Λ_cross = cache_data.get('Λcross', None)
            return cache_data['result']
        
        # 生成sigma点
        gen = SigmaGenerator(self.alpha, self.beta, self.kappa)
        χ, wm, wc = gen.generate(μx, Px)
        
        # 计算sigma点的观测
        yσ = np.zeros((len(χ), 2))
        for i, p in enumerate(χ):
            yσ[i] = self._h(p)
        
        # 计算均值 - 特殊处理角度
        μy_range = np.sum(wm * yσ[:, 1])
        sin_sum = np.sum(wm * np.sin(yσ[:, 0]))
        cos_sum = np.sum(wm * np.cos(yσ[:, 0]))
        μy_bearing = math.atan2(sin_sum, cos_sum)
        μy = np.array([μy_bearing, μy_range])
        
        # 计算残差
        r = self.z - μy
        r[0] = wrap_angle(r[0])
        
        # 计算协方差
        diff_x = χ - μx
        diff_x[:, 2] = wrap_angle(diff_x[:, 2])  # 角度分量wrap处理
        
        diff_y = np.zeros_like(yσ)
        
        # 正确的角度差分计算
        for k in range(len(χ)):
            diff_y[k, 0] = wrap_angle(yσ[k, 0] - μy_bearing)
            diff_y[k, 1] = yσ[k, 1] - μy_range
        
        Sy = np.zeros((2, 2))
        Pxy = np.zeros((5, 2))
        for k in range(len(χ)):
            Sy += wc[k] * np.outer(diff_y[k], diff_y[k])
            Pxy += wc[k] * np.outer(diff_x[k], diff_y[k])
        Sy += self.R  # 添加观测噪声
        
        # 正则化Sy
        Sy_reg = Sy + np.eye(2) * _SMALL_NUMBER
        try:
            c, low = scipy.linalg.cho_factor(Sy_reg)
            Sinv = scipy.linalg.cho_solve((c, low), np.eye(2))
        except np.linalg.LinAlgError:
            Sy_reg += np.eye(2) * _SMALL_NUMBER
            Sinv = np.linalg.inv(Sy_reg)
        
        # 计算雅可比矩阵 (关键步骤)
        J = Pxy @ Sinv  # (5,2)
        
        # 关键修正：正确信息矩阵
        Λz = J.T @ self.Rinv @ J  # J^T R^{-1} J (5x5)
        
        # 关键修正：正确信息向量
        ηz = (J.T @ self.Rinv @ r).reshape(-1, 1)  # J^T R^{-1} r (5,1)
        
        # 分解信息矩阵和向量
        ηp = ηz[:3]     # (3,1)
        ηl = ηz[3:]     # (2,1)
        Λp = Λz[:3, :3]  # (3,3)
        Λl = Λz[3:, 3:]  # (2,2)
        Λcross = Λz[:3, 3:]  # (3,2)
        
        # 构建结果
        result = {
            self.pose_key: (Λp, ηp),
            self.lm_key: (Λl, ηl),
            (self.pose_key, self.lm_key): Λcross,
            (self.lm_key, self.pose_key): Λcross.T
        }
        
        # 更新共享状态
        with self._lock:
            self._last_μx = μx.copy()
            self._last_Λ_cross = Λcross.copy()
            self._spbp_cache[cache_key] = {
                'result': result,
                'Λcross': Λcross
            }
        
        return result

    def _get_dim(self, key: str) -> int:
        """返回变量维度 - 使用dim_map确保一致性"""
        return self._dim_map.get(key, 0)