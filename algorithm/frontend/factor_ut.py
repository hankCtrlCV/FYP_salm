"""Factor implementations for 3-D multi-robot SLAM (x, y, θ)"""

from __future__ import annotations
import numpy as np
import math
from typing import Dict, Tuple, Union
from abc import ABC, abstractmethod
from algorithm.math.spbp import SigmaGenerator

# ---------------------------------------------------------------------
# 共用常量
_EPS = 1e-12
_SMALL_NUMBER = 1e-9

# 角度归一化函数 - 修正为更高效且JIT友好的版本
def wrap_angle(theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """支持标量或ndarray，结果归一化到[-π, π)"""
    return (theta + np.pi) % (2 * np.pi) - np.pi

# =====================================================================
# 基类
# =====================================================================
class Factor(ABC):
    @abstractmethod
    def linearize(self,
                  mu:  Dict[str, np.ndarray],
                  cov: Dict[str, np.ndarray]
                  ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """线性化因子，返回信息矩阵和信息向量"""
        ...

    @abstractmethod
    def get_energy(self,
                   mu: Dict[str, np.ndarray]
                   ) -> float:
        """计算当前状态的能量"""
        ...


# =====================================================================
# 1. Prior factor (3D支持)
# =====================================================================
class PriorFactor(Factor):
    def __init__(self, var: str, prior: np.ndarray, sigmas: np.ndarray):
        """
        3D位姿先验因子
        
        参数:
            var: 变量名
            prior: 先验均值 (3,)
            sigmas: 标准差向量 (3,)
        """
        self.var = var
        self.prior = prior.astype(float)
        
        # 使用向量sigmas构造对角信息矩阵
        if np.isscalar(sigmas):
            sigma_array = np.ones(prior.shape[0]) * sigmas
        else:
            sigma_array = sigmas
            
        # 处理3维位姿
        if prior.shape[0] == 3:
            # 位置和角度分开处理
            sigma_x, sigma_y, sigma_theta = sigma_array
            self._Λ = np.diag([
                1.0/(sigma_x**2),
                1.0/(sigma_y**2),
                1.0/(sigma_theta**2)
            ])
        else:
            self._Λ = np.diag(1.0 / (sigma_array ** 2))
        
        self._η = self._Λ @ self.prior

    def linearize(self, *_):
        """返回先验因子的信息矩阵和信息向量"""
        return {self.var: (self._Λ, self._η)}

    def get_energy(self, mu: Dict[str, np.ndarray]) -> float:
        """计算当前状态的能量"""
        r = mu[self.var] - self.prior
        # 对于3维位姿，角度残差需要wrap
        if r.shape[0] == 3:
            r[2] = wrap_angle(r[2])
        return 0.5 * r @ self._Λ @ r


# =====================================================================
# 2. Odometry factor (3D支持) - 修正η的符号
# =====================================================================
class OdometryFactor(Factor):
    """3D里程计因子: x_j = x_i + [Δx, Δy, Δθ] + noise(σ)"""

    def __init__(self, v_from: str, v_to: str,
                 delta: np.ndarray, sigmas: np.ndarray):
        """
        初始化里程计因子
        
        参数:
            v_from: 起始位姿变量名
            v_to: 目标位姿变量名
            delta: 位姿变化量 (Δx, Δy, Δθ)
            sigmas: 标准差向量 (3,)
        """
        self.v1, self.v2 = v_from, v_to
        self.delta = delta.astype(float)
        
        # 使用向量sigmas构造对角信息矩阵
        if np.isscalar(sigmas):
            sigma_array = np.ones(delta.shape[0]) * sigmas
        else:
            sigma_array = sigmas
            
        if delta.shape[0] == 3:
            sigma_x, sigma_y, sigma_theta = sigma_array
            self.Rinv = np.diag([
                1.0/(sigma_x**2),
                1.0/(sigma_y**2),
                1.0/(sigma_theta**2)
            ])
        else:
            self.Rinv = np.diag(1.0 / (sigma_array ** 2))

    def linearize(self, mu: Dict, cov: Dict):
        """
        线性化里程计因子
        
        返回:
            Dict: {
                v1: (Λ11, η1),
                v2: (Λ22, η2),
                (v1, v2): (Λ12, None)  # 交叉信息块
            }
        """
        mu1, mu2 = mu[self.v1], mu[self.v2]
        
        # 计算残差（角度需要wrap）
        residual = self.delta - (mu2 - mu1)
        if residual.shape[0] == 3:
            residual[2] = wrap_angle(residual[2])
            
        # Jacobian矩阵 - 统一形式
        dim = residual.shape[0]
        J1 = -np.eye(dim)
        J2 = np.eye(dim)
        
        # 信息矩阵计算
        Λ11 = J1.T @ self.Rinv @ J1
        Λ22 = J2.T @ self.Rinv @ J2
        Λ12 = J1.T @ self.Rinv @ J2
        
        # η计算公式 - 符号修正
        η1 = J1.T @ self.Rinv @ residual - (Λ11 @ mu1 + Λ12 @ mu2)
        η2 = J2.T @ self.Rinv @ residual - (Λ22 @ mu2 + Λ12.T @ mu1)
        
        # 返回结果
        return {
            self.v1: (Λ11, η1),
            self.v2: (Λ22, η2),
            (self.v1, self.v2): Λ12,
            (self.v2, self.v1): Λ12.T  # 同时返回对称块
        }

    def get_energy(self, mu: Dict[str, np.ndarray]) -> float:
        """计算当前状态的能量"""
        r = (mu[self.v2] - mu[self.v1]) - self.delta
        
        # 对于3维位姿，角度残差需要wrap
        if r.shape[0] == 3:
            r[2] = wrap_angle(r[2])
        return 0.5 * r @ self.Rinv @ r


# =====================================================================
# 3. Bearing-Range factor (GBP/SPBP/AUTO) - 修正η和SPBP实现
# =====================================================================
class BearingRangeUTFactor(Factor):
    """距离-方位角观测因子 (支持GBP与SPBP)"""

    def __init__(self,
                 pose_var: str, lm_var: str,
                 measurement: np.ndarray, R: np.ndarray,
                 *,
                 mode: str = "auto",
                 alpha: float = 1e-3, 
                 beta: float = 2.0, 
                 kappa: float = 0.0,
                 linear_threshold: float = 5.0,
                 residual_sigma_thresh: float = 3.0):
        """
        初始化距离-方位角观测因子
        
        参数:
            pose_var: 位姿变量名
            lm_var: 地标变量名
            measurement: 量测值 [距离, 方位角]
            R: 2×2 观测噪声协方差矩阵
            mode: 线性化方法 ("gbp", "spbp", "auto")
            alpha, beta, kappa: UT参数
            linear_threshold: 触发SPBP的距离阈值 (米)
            residual_sigma_thresh: 触发SPBP的残差阈值 (标准差倍数)
        """
        self.pose_key = pose_var
        self.lm_key = lm_var
        self.z = measurement.astype(float)
        self.R = R.copy()
        self.Rinv = np.linalg.inv(self.R)
        
        # 线性化模式管理
        self._current_mode = "init"
        self._diverged_cnt = 0
        
        # 参数配置
        self.mode = mode.lower()
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.linear_th = linear_threshold
        self.res_th_mul = residual_sigma_thresh
        
        # 缓存
        self._Λy = None

    def linearize(self, 
                 mu: Dict[str, np.ndarray],
                 cov: Dict[str, np.ndarray]) -> Dict[Union[str, Tuple], Union[Tuple, np.ndarray]]:
        """
        线性化因子
        """
        mu_p = mu[self.pose_key]
        mu_l = mu[self.lm_key]
        
        # 获取协方差信息
        P_p = cov[self.pose_key]
        P_l = cov[self.lm_key]
        P_pl = cov.get((self.pose_key, self.lm_key), np.zeros((mu_p.size, mu_l.size)))
        
        # 构建联合状态
        mu_x = np.hstack([mu_p, mu_l])
        P_x = np.block([[P_p, P_pl], [P_pl.T, P_l]])
        dim_p = mu_p.size
        
        # 数值稳定性：添加小对角扰动
        P_x += np.eye(P_x.shape[0]) * _SMALL_NUMBER
        
        # 计算预测量测和残差
        pred_y = self._h(mu_x)
        residual = self.z - pred_y
        residual[1] = wrap_angle(residual[1])  # 角度归一化
        
        # 计算机器人到地标的距离
        dx = mu_l[0] - mu_p[0]
        dy = mu_l[1] - mu_p[1]
        distance = np.linalg.norm([dx, dy])
        
        # 决策线性化模式
        mode = self._decide_mode(distance, residual)
        if mode == "spbp":
            return self._spbp_linearize(mu_x, P_x, residual, dim_p)
        else:
            result = self._gbp_linearize(mu_p, mu_l, residual, distance)
            
            # AUTO模式下根据残差重置或增加发散计数 ✔️补丁3
            if self.mode == "auto":
                sigma_range = np.sqrt(self.R[0, 0])
                sigma_bearing = np.sqrt(self.R[1, 1])
                if (abs(residual[0]) < 5 * sigma_range and 
                    abs(residual[1]) < 5 * sigma_bearing):
                    self._diverged_cnt = 0
                else:
                    self._diverged_cnt += 1
            return result

    def get_energy(self, 
                   mu: Dict[str, np.ndarray]) -> float:
        """计算当前状态的能量"""
        if self._Λy is None:
            return 0.0
            
        pred_y = self._h(np.hstack([mu[self.pose_key], mu[self.lm_key]]))
        residual = self.z - pred_y
        residual[1] = wrap_angle(residual[1])
        
        return 0.5 * residual @ self._Λy @ residual

    # -----------------------------------------------------------------
    # 内部工具方法
    # -----------------------------------------------------------------
    @staticmethod
    def _h(x: np.ndarray) -> np.ndarray:
        """观测模型: [距离, 方位角]"""
        pose_x, pose_y = x[0], x[1]  # 位姿位置
        lm_x, lm_y = x[-2], x[-1]    # 地标位置
        
        dx = lm_x - pose_x
        dy = lm_y - pose_y
        
        rng = np.hypot(dx, dy)
        # ✔️补丁2a: 对bearing立即wrap
        bearing = wrap_angle(np.arctan2(dy, dx))
        return np.array([rng, bearing])

    def _decide_mode(self, distance: float, residual: np.ndarray) -> str:
        """决策线性化模式"""
        if self.mode in ("gbp", "spbp"):
            return self.mode
        
        # 基于距离和残差决策
        sigma_range = np.sqrt(self.R[0, 0])
        sigma_bearing = np.sqrt(self.R[1, 1])
        
        if (distance > self.linear_th or
            abs(residual[0]) > self.res_th_mul * sigma_range or
            abs(residual[1]) > self.res_th_mul * sigma_bearing or
            self._diverged_cnt > 3):
            return "spbp"
        
        return "gbp"

    def _gbp_linearize(self,
                       mu_p: np.ndarray,
                       mu_l: np.ndarray,
                       residual: np.ndarray,
                       distance: float) -> Dict:
        """GBP线性化实现"""
        dx = mu_l[0] - mu_p[0]
        dy = mu_l[1] - mu_p[1]
        r = max(distance, _SMALL_NUMBER)  # 防止距离为0
        q = r**2
        
        # Jacobian矩阵
        J_p = np.array([
            [-dx/r, -dy/r, 0],
            [dy/q, -dx/q, -1]
        ])
        
        J_l = np.array([
            [dx/r, dy/r],
            [-dy/q, dx/q]
        ])
        
        # 信息矩阵计算
        Λ_p = J_p.T @ self.Rinv @ J_p
        Λ_l = J_l.T @ self.Rinv @ J_l
        Λ_cross = J_p.T @ self.Rinv @ J_l
        
        # 信息向量计算 - 符号修正
        η_p = J_p.T @ self.Rinv @ residual - (Λ_p @ mu_p + Λ_cross @ mu_l)
        η_l = J_l.T @ self.Rinv @ residual - (Λ_l @ mu_l + Λ_cross.T @ mu_p)
        
        # 更新状态
        self._current_mode = "gbp"
        self._Λy = self.Rinv
        
        return {
            self.pose_key: (Λ_p, η_p),
            self.lm_key: (Λ_l, η_l),
            (self.pose_key, self.lm_key): Λ_cross,
            (self.lm_key, self.pose_key): Λ_cross.T  # 对称交叉块
        }

    def _spbp_linearize(self,
                        mu_x: np.ndarray,
                        P_x: np.ndarray,
                        residual: np.ndarray,
                        dim_p: int) -> Dict:
        """SPBP线性化实现 - 修正η计算和Pxy计算"""
        # 生成Sigma点
        gen = SigmaGenerator(self.alpha, self.beta, self.kappa)
        sigma_pts, wm, wc = gen.generate(mu_x, P_x)
        n_sigma = sigma_pts.shape[0]
        
        # 计算所有Sigma点的预测观测
        y_sigma = np.array([self._h(pt) for pt in sigma_pts])
        
        # ✔️补丁2b: 对bearing进行wrap
        y_sigma[:, 1] = wrap_angle(y_sigma[:, 1])
        
        # 计算统计量
        mu_y = np.average(y_sigma, axis=0, weights=wm)
        # ✔️补丁2b: 对mu_y中的bearing进行wrap
        mu_y[1] = wrap_angle(mu_y[1])
        
        # 计算残差
        diff_x = sigma_pts - mu_x
        diff_y = y_sigma - mu_y
        # ✔️补丁2b: 对diff_y中的bearing进行wrap
        diff_y[:, 1] = wrap_angle(diff_y[:, 1])
        
        # 计算协方差
        Sy = np.zeros((2, 2))
        Pxy = np.zeros((mu_x.size, 2))
        
        for k in range(n_sigma):
            Sy += wc[k] * np.outer(diff_y[k], diff_y[k])
            Pxy += wc[k] * np.outer(diff_x[k], diff_y[k])
            
        Sy += self.R
        
        # 信息矩阵计算
        Sinv = np.linalg.inv(Sy)
        Λ = Pxy @ Sinv @ Pxy.T
        
        # 分块拆解
        Λ_p = Λ[:dim_p, :dim_p]
        Λ_l = Λ[dim_p:, dim_p:]
        Λ_cross = Λ[:dim_p, dim_p:]
        
        # 信息向量计算 - 修正η公式
        η_joint = Pxy @ Sinv @ residual - Λ @ mu_x
        η_p = η_joint[:dim_p]
        η_l = η_joint[dim_p:]
        
        # 更新状态
        self._current_mode = "spbp"
        self._Λy = Sinv
        
        return {
            self.pose_key: (Λ_p, η_p),
            self.lm_key: (Λ_l, η_l),
            (self.pose_key, self.lm_key): Λ_cross,
            (self.lm_key, self.pose_key): Λ_cross.T  # 对称交叉块
        }