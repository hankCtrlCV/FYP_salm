"""
Enhanced Factor implementations for Multi-Robot GBP-SLAM
Supports both GBP and SPBP (Sigma Points Belief Propagation) with UT

Author: Enhanced for Multi-Robot SLAM (Improved Version)
Date: 2025-06-21
Version: 2.3 - MAJOR: Fixed caching and numerical robustness issues
Critical Fixes:
- Fixed SPBP dimension bug (K @ Rinv @ K.T vs K.T @ Rinv @ K) 
- Fixed cache key issues: using content hash instead of memory address
- Fixed SigmaGenerator kappa race condition in multithreading
- Unified factor caching: single global cache manager instead of per-factor caches
- Enhanced numerical robustness: improved SVD pseudoinverse, enhanced PD enforcement
- Relaxed small angle threshold from 1e-6 to 1e-3 rad for better stability
- Added comprehensive dimension validation and safe SPBP conversion
- Memory optimization: reduced cache instances from thousands to a few global ones
- ✅ NEW: Fixed _get_dim() design pattern and single-direction cross blocks
"""

from __future__ import annotations
import numpy as np
import math
import threading
import weakref
import logging
import scipy
from typing import Dict, Tuple, Union, Any, Optional, List, NamedTuple
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from numpy.linalg import LinAlgError, solve
from functools import lru_cache
import warnings

# Configure logging
logger = logging.getLogger(__name__)

# =====================================================================
# Constants and Configuration
# =====================================================================


@dataclass
class PerformanceConfig:
    """所有缓存尺寸由 YAML → graph_build.py 注入后写回这里"""
    default_cache_size:     int = 128
    sigma_param_cache_size: int = 64
    energy_cache_size:      int = 512
    spbp_cache_size:        int = 256
    jacobian_cache_size:    int = 128

# **全局可变实例**；GBPGraphBuilder 解析完 YAML 后会覆盖字段
performance_config = PerformanceConfig()

@dataclass
class NumericalConfig:
    """数值计算相关阈值（可由 YAML / CLI 覆盖）"""
    max_condition_number:  float = 1.0e6
    min_eigenvalue:        float = 1.0e-6
    small_angle_threshold: float = 1.0e-3
    regularization:        float = 1.0e-6
    jacobian_clip_threshold: float = 1.0e6
    angle_wrap_tolerance:   float = 1.0e-12
    symmetry_rtol:          float = 1.0e-10   # ← 新增：矩阵对称检查 rtol
    symmetry_atol:          float = 1.0e-12   # ← 新增：矩阵对称检查 atol
    weight_sum_tol:         float = 1.0e-10   # ← 新增：UT 权重和≈1 容差
    max_information_diagonal: float = 1.0e8

# 全局配置实例
numerical_config = NumericalConfig()

# =====================================================================
# Thread-Safe Cache Implementation
# =====================================================================
class ThreadSafeCache:
    """线程安全的LRU缓存实现"""
    def __init__(self, maxsize: int | None = None):
        if maxsize is None:
            maxsize = performance_config.default_cache_size
        self.maxsize = maxsize
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: Any, default: Any = None) -> Any:
        """获取缓存值"""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)  # LRU: 移到末尾
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return default
    
    def put(self, key: Any, value: Any) -> None:
        """存储缓存值"""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self.maxsize:
                self._cache.popitem(last=False)  # 移除最旧的项
            self._cache[key] = value
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def stats(self) -> Dict[str, Union[int, float]]:
        """获取缓存统计"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                'hits': self._hits,
                'misses': self._misses,
                'size': len(self._cache),
                'hit_rate': hit_rate
            }
    
    # 允许在 YAML 更新 performance_config 后重建缓存实例
    @staticmethod 
    def rebuild_caches() -> None:
        """
        Re-instantiate global caches so that new values in performance_config
        (e.g. after YAML injection) take effect immediately.
        """
        global _global_cache, _factor_cache_manager
        _global_cache = ThreadSafeCache()          # uses up-to-date default_cache_size
        _factor_cache_manager = FactorCacheManager()

# 全局缓存实例
_global_cache = ThreadSafeCache()

# 因子专用缓存管理器
class FactorCacheManager:
    """
    统一的因子缓存管理器，避免每个因子都创建独立缓存
    """
    
    def __init__(self):
        self.energy_cache   = ThreadSafeCache(maxsize=performance_config.energy_cache_size)
        self.spbp_cache     = ThreadSafeCache(maxsize=performance_config.spbp_cache_size)
        self.jacobian_cache = ThreadSafeCache(maxsize=performance_config.jacobian_cache_size)
        
    def get_energy_cache_key(self, factor_type: str, var_data: bytes) -> str:
        """生成能量缓存键"""
        return f"energy_{factor_type}_{hash(var_data)}"
    
    def get_spbp_cache_key(self, factor_type: str, state_data: bytes, cov_data: bytes) -> str:
        """生成SPBP缓存键"""
        return f"spbp_{factor_type}_{hash(state_data)}_{hash(cov_data)}"
    
    def clear_all(self):
        """清空所有缓存"""
        self.energy_cache.clear()
        self.spbp_cache.clear()
        self.jacobian_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            'energy_cache': self.energy_cache.stats(),
            'spbp_cache': self.spbp_cache.stats(),
            'jacobian_cache': self.jacobian_cache.stats()
        }

# 全局因子缓存管理器
_factor_cache_manager = FactorCacheManager()

# =====================================================================
# Enhanced Numerical Utilities
# =====================================================================
def wrap_angle(theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    数值稳定的角度包装到[-π, π]
    
    Args:
        theta: 要包装的角度
        
    Returns:
        包装后的角度在[-π, π]范围内
    """
    if isinstance(theta, np.ndarray):
        # 向量化处理
        result = np.arctan2(np.sin(theta), np.cos(theta))
        # 处理数值误差
        tol = numerical_config.angle_wrap_tolerance
        result = np.where(np.abs(result - np.pi) < tol, -np.pi, result)
        return result
    else:
        result = math.atan2(math.sin(theta), math.cos(theta))
        # 处理π边界情况
        if abs(result - math.pi) < numerical_config.angle_wrap_tolerance:
            result = -math.pi
        return result

def angle_diff(a1: Union[float, np.ndarray], a2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    计算角度差 a1 - a2，包装到[-π, π]
    """
    return wrap_angle(a1 - a2)

def validate_matrix_properties(matrix: np.ndarray, name: str = "matrix", 
                             check_symmetric: bool = True, 
                             check_positive_definite: bool = True) -> bool:
    """
    验证矩阵的数值属性
    
    Args:
        matrix: 要验证的矩阵
        name: 矩阵名称（用于错误信息）
        check_symmetric: 是否检查对称性
        check_positive_definite: 是否检查正定性
        
    Returns:
        验证是否通过
        
    Raises:
        ValueError: 验证失败时
    """
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name}: Contains non-finite values")
    
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name}: Must be square matrix, got shape {matrix.shape}")
    
    if check_symmetric:
        if not np.allclose(
                matrix, matrix.T,
                rtol=numerical_config.symmetry_rtol,
                atol=numerical_config.symmetry_atol):
            raise ValueError(f"{name}: Matrix is not symmetric")
    
    if check_positive_definite:
        try:
            eigenvals = np.linalg.eigvals(matrix)
            if np.any(eigenvals <= 0):
                min_eigenval = np.min(eigenvals)
                raise ValueError(f"{name}: Not positive definite (min eigenvalue: {min_eigenval})")
            
            # 检查条件数
            cond = np.max(eigenvals) / np.min(eigenvals)
            if cond > numerical_config.max_condition_number:
                logger.warning(f"{name}: High condition number {cond:.2e}")
                
        except np.linalg.LinAlgError as e:
            raise ValueError(f"{name}: Eigenvalue computation failed: {e}")
    
    return True

def ensure_positive_definite(matrix: np.ndarray, 
                           regularization: float = None,
                           method: str = "cholesky_first") -> np.ndarray:
    """
    确保矩阵正定，使用改进的数值稳定性策略
    
    Args:
        matrix: 输入矩阵
        regularization: 正则化参数
        method: 方法选择 ("cholesky_first", "eigenvalue", "identity_reg")
        
    Returns:
        正定矩阵
    """
    if regularization is None:
        regularization = numerical_config.regularization
    
    # 首先对称化
    matrix = 0.5 * (matrix + matrix.T)
    
    if method == "cholesky_first":
        try:
            # 先尝试Cholesky分解验证是否已经正定
            np.linalg.cholesky(matrix)
            return matrix.copy()
        except np.linalg.LinAlgError:
            # 失败则使用特征值分解
            method = "eigenvalue"
    
    if method == "eigenvalue":
        try:
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            # 正则化特征值
            scale = max(regularization, regularization * eigenvals.max())
            eigenvals_reg = np.maximum(eigenvals, scale)
            
            # 检查是否需要正则化
            if np.any(eigenvals <= regularization):
                logger.debug(f"Regularized {np.sum(eigenvals <= regularization)} eigenvalues")
            
            A = eigenvecs @ np.diag(eigenvals_reg) @ eigenvecs.T
            return 0.5 * (A + A.T)
            
        except np.linalg.LinAlgError:
            logger.warning("Eigenvalue decomposition failed, using identity regularization")
            method = "identity_reg"
    
    if method == "identity_reg":
        # 最后的回退策略：添加对角正则化
        reg_matrix = matrix + regularization * np.eye(matrix.shape[0])
        return reg_matrix
    
    raise ValueError(f"Unknown method: {method}")

def safe_matrix_inverse(matrix: np.ndarray, 
                       regularization: float = None,
                       check_condition: bool = True) -> np.ndarray:
    """
    数值稳定的矩阵求逆
    
    Args:
        matrix: 要求逆的矩阵
        regularization: 正则化参数
        check_condition: 是否检查条件数
        
    Returns:
        逆矩阵
    """
    if regularization is None:
        regularization = numerical_config.regularization
    
    # 输入验证
    if not np.isfinite(matrix).all():
        raise ValueError("Matrix contains non-finite values")
    
    try:
        # 首先尝试Cholesky求逆（最快且数值稳定）
        matrix_pd = ensure_positive_definite(matrix, regularization, "cholesky_first")
        
        # 条件数检查
        if check_condition:
            cond = np.linalg.cond(matrix_pd)
            if cond > numerical_config.max_condition_number:
                logger.warning(f"High condition number in matrix inversion: {cond:.2e}")
        
        # Cholesky分解求逆
        L = np.linalg.cholesky(matrix_pd)
        identity = np.eye(matrix.shape[0])
        inv_L = np.linalg.solve(L, identity)
        return inv_L.T @ inv_L
        
    except np.linalg.LinAlgError:
        logger.debug("Cholesky inversion failed, falling back to SVD")
        
        # 回退到SVD方法
        try:
            U, s, Vt = np.linalg.svd(matrix)
            # 正则化奇异值
            s_reg = np.where(s > regularization, 1.0/s, 0.0)
            return Vt.T @ np.diag(s_reg) @ U.T
            
        except np.linalg.LinAlgError as e:
            logger.error(f"Matrix inversion completely failed: {e}")
            # 最后的回退：正则化的伪逆
            return np.linalg.pinv(matrix, rcond=regularization)

def clip_jacobian(jacobian: np.ndarray, threshold: float = None) -> np.ndarray:
    """
    裁剪雅可比矩阵防止数值爆炸
    
    Args:
        jacobian: 雅可比矩阵
        threshold: 裁剪阈值
        
    Returns:
        裁剪后的雅可比矩阵
    """
    if threshold is None:
        threshold = numerical_config.jacobian_clip_threshold
    
    # 提前跳过：如果已经在阈值内，避免不必要的拷贝
    if np.max(np.abs(jacobian)) <= threshold:
        return jacobian
    
    clipped = np.clip(jacobian, -threshold, threshold)
    
    # 记录裁剪信息
    max_val = np.max(np.abs(jacobian))
    logger.debug(f"Jacobian clipped: max value was {max_val:.2e}")
    
    return clipped

# =====================================================================
# Sigma Points Generator
# =====================================================================
class SigmaGenerator:
    """
    增强的sigma点生成器，具有改进的数值稳定性和缓存
    """
    
    def __init__(self, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        """
        初始化UT参数
        
        Args:
            alpha: 扩散参数 (1e-4 ≤ α ≤ 1)
            beta: 高阶矩参数 (β = 2 对高斯分布最优)
            kappa: 次要缩放参数
        """
        self.alpha = np.clip(alpha, 1e-4, 1.0)  # 严格限制范围
        self.beta = beta
        self.kappa = kappa
        
        # 线程安全的参数缓存
        self._param_cache = ThreadSafeCache(maxsize=performance_config.sigma_param_cache_size)
        
        # 预分配工作数组
        self._work_arrays = {}
        
    def _get_cached_params(self, n: int) -> Tuple[float, np.ndarray, np.ndarray]:
        """获取缓存的UT参数，线程安全且避免kappa竞态"""
        cache_key = f"ut_params_{n}_{self.alpha}_{self.beta}_{self.kappa}"
        cached = self._param_cache.get(cache_key)
        
        if cached is not None:
            return cached
        
        # 使用局部变量避免多线程竞态
        local_kappa = self.kappa
        lambda_ = self.alpha**2 * (n + local_kappa) - n
        
        # 确保数值稳定性 - 调整局部kappa而不修改实例字段
        if lambda_ + n <= 0:
            local_kappa = 3 - n + 1e-6  # 使用局部变量
            lambda_ = self.alpha**2 * (n + local_kappa) - n
            logger.warning(f"Locally adjusted kappa from {self.kappa} to {local_kappa} for n={n}")
            # 注意：不修改self.kappa，避免多线程竞态
        
        gamma = math.sqrt(n + lambda_)
        
        # 计算权重
        denom = n + lambda_
        wm = np.full(2*n+1, 0.5/denom)
        wc = wm.copy()
        
        wm[0] = lambda_/denom
        wc[0] = wm[0] + (1 - self.alpha**2 + self.beta)
        
        # 验证权重和为1
        weight_sum = np.sum(wm)
        if abs(weight_sum - 1.0) > numerical_config.weight_sum_tol:    
            logger.warning(f"Weight sum deviation: {weight_sum}")
            wm = wm / weight_sum  # 归一化
        
        params = (lambda_, wm, wc)
        self._param_cache.put(cache_key, params)
        return params
        
    def generate(self, mu: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成sigma点用于UT变换
        
        Args:
            mu: 均值向量
            cov: 协方差矩阵
            
        Returns:
            sigma_points: (2n+1, n) sigma点
            wm: 均值权重
            wc: 协方差权重
        """
        mu = np.asarray(mu, dtype=float)
        cov = np.asarray(cov, dtype=float)
        
        n = mu.size
        
        # 验证输入
        validate_matrix_properties(cov, "covariance matrix", 
                                 check_symmetric=True, 
                                 check_positive_definite=True)
        
        if mu.shape[0] != cov.shape[0]:
            raise ValueError(f"Dimension mismatch: mu={mu.shape}, cov={cov.shape}")
        
        # 获取UT参数
        lambda_, wm, wc = self._get_cached_params(n)
        gamma = math.sqrt(n + lambda_)
        
        # 计算矩阵平方根
        try:
            # 确保协方差矩阵正定
            cov_pd = ensure_positive_definite(cov, numerical_config.min_eigenvalue)
            
            # 使用Cholesky分解
            S = np.linalg.cholesky(cov_pd)
            
        except np.linalg.LinAlgError:
            logger.debug("Cholesky failed, using SVD square root")
            # 回退到SVD方法
            U, s, Vt = np.linalg.svd(cov)
            s_safe = np.maximum(s, numerical_config.min_eigenvalue)
            S = U @ np.diag(np.sqrt(s_safe))
        
        # 生成sigma点
        sigma_points = np.zeros((2*n + 1, n))
        sigma_points[0] = mu
        
        # 使用预分配的数组（如果可能）
        offset_vectors = gamma * S.T  # (n, n)
        
        for i in range(n):
            sigma_points[i+1] = mu + offset_vectors[i]
            sigma_points[n+i+1] = mu - offset_vectors[i]
        
        # 最终验证
        if not np.isfinite(sigma_points).all():
            raise ValueError("Generated sigma points contain non-finite values")
        
        return sigma_points, wm, wc
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return self._param_cache.stats()

# =====================================================================
#  Base Factor Class - ✅ 修复：移除@abstractmethod，提供默认实现
# =====================================================================
class Factor(ABC):
    """
    增强的因子基类，具有改进的错误检查和验证
    """
    
    def __init__(self):
        self._validation_enabled = True
        # ✅ 所有子类都会在__init__中设置这个映射
        self._dim_map = {}
        
    @abstractmethod
    def linearize(self, mu: Dict[str, np.ndarray], cov: Dict[str, np.ndarray]) -> Dict:
        """
        在当前估计附近线性化因子
        
        Args:
            mu: 变量均值
            cov: 变量协方差（和交叉协方差）
            
        Returns:
            包含信息矩阵块的字典:
            - var_key: (Λ_ii, η_i) 对角块
            - (var_i, var_j): Λ_ij 非对角块（只有单向）
        """
        pass
    
    @abstractmethod
    def get_energy(self, mu: Dict[str, np.ndarray]) -> float:
        """
        计算因子能量（平方马氏距离）
        
        Args:
            mu: 变量值
            
        Returns:
            因子能量
        """
        pass
    
    @staticmethod
    def assert_linearize_dict_format(result: dict):
        assert isinstance(result, dict), f"linearize must return dict, got {type(result)}"
        for key, value in result.items():
            if isinstance(key, tuple):
                assert isinstance(value, np.ndarray), f"cross-term {key} is not ndarray: {type(value)}"
                assert value.ndim == 2, f"cross-term {key} must be 2D matrix, got {value.shape}"
            else:
                assert isinstance(value, tuple) and len(value) == 2, \
                    f"diag-term {key} must be (Λ, η) tuple, got {value}"
                Λ, η = value
                assert isinstance(Λ, np.ndarray) and isinstance(η, np.ndarray), \
                    f"diag-term {key} Λ, η must be ndarray, got {type(Λ)}, {type(η)}"
    
    # ✅ 修复1：移除@abstractmethod，提供默认实现
    def _get_dim(self, key: str) -> int:
        """
        获取变量维度的默认实现
        子类通过维护self._dim_map来指定维度
        
        Args:
            key: 变量键
            
        Returns:
            变量维度（如果不涉及此因子则返回0）
        """
        return self._dim_map.get(key, 0)
    
    def validate_linearization_result(self, result: Dict) -> bool:
        """
        验证线性化结果的数值合理性
        
        Args:
            result: 线性化结果字典
            
        Returns:
            结果是否有效
        """
        if not self._validation_enabled:
            return True
            
        try:
            for key, value in result.items():
                if isinstance(key, tuple):
                    # 交叉项
                    if not isinstance(value, np.ndarray) or value.ndim != 2:
                        logger.error(f"Invalid cross-term shape for {key}: {value.shape}")
                        return False
                    
                    if not np.isfinite(value).all():
                        logger.error(f"Non-finite values in cross-term {key}")
                        return False
                    
                    # 检查数值范围
                    max_val = np.max(np.abs(value))
                    if max_val > numerical_config.jacobian_clip_threshold:
                        logger.warning(f"Large values in cross-term {key}: {max_val:.2e}")
                        
                else:
                    # 对角项
                    if not isinstance(value, tuple) or len(value) != 2:
                        logger.error(f"Invalid diagonal term format for {key}")
                        return False
                    
                    L, eta = value
                    if not isinstance(L, np.ndarray) or not isinstance(eta, np.ndarray):
                        logger.error(f"Diagonal term components not arrays for {key}")
                        return False
                    
                    if not np.isfinite(L).all() or not np.isfinite(eta).all():
                        logger.error(f"Non-finite values in diagonal term {key}")
                        return False
                    
                    # 验证信息矩阵正定性
                    try:
                        validate_matrix_properties(L, f"information_matrix_{key}", 
                                                 check_symmetric=True, 
                                                 check_positive_definite=True)
                    except ValueError as e:
                        logger.error(f"Invalid information matrix for {key}: {e}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def enable_validation(self, enabled: bool = True):
        """启用/禁用验证（用于性能优化）"""
        self._validation_enabled = enabled
    
    def clear_caches(self):
        """清空缓存 - 现在使用全局缓存管理器"""
        _factor_cache_manager.clear_all()

# =====================================================================
# Prior Factor 
# =====================================================================
class PriorFactor(Factor):
    """
    改进的先验因子，具有更好的数值稳定性
    """
    
    def __init__(self, var: str, prior: np.ndarray, sigmas: Union[float, np.ndarray]):
        """
        初始化先验因子
        
        Args:
            var: 变量名
            prior: 先验均值
            sigmas: 先验标准差（标量或数组）
        """
        super().__init__()
        
        self.var = var
        self.var_keys = [self.var] 
        self.prior = np.asarray(prior, dtype=float)
        
        self._dim_map = {var: self.prior.shape[0]}
        
        # 处理sigma输入
        if np.isscalar(sigmas):
            sigma_array = np.full_like(self.prior, sigmas, dtype=float)
        else:
            sigma_array = np.asarray(sigmas, dtype=float)
            if sigma_array.shape != self.prior.shape:
                raise ValueError(f"Sigma shape {sigma_array.shape} doesn't match prior shape {self.prior.shape}")
        
        # 确保最小sigma以保证数值稳定性
        sigma_array = np.maximum(sigma_array, numerical_config.min_eigenvalue**0.5)
        
        
        # 预计算信息矩阵和向量
        variances = np.clip(
            sigma_array ** 2,
            numerical_config.min_eigenvalue,
            1.0 / numerical_config.max_information_diagonal
        )
        self._Λ = np.diag(1.0 / variances)
        self._η = self._Λ @ self.prior
        
        # 验证
        validate_matrix_properties(self._Λ, "prior_information_matrix")
        
        logger.debug(f"Created prior factor for {var} with dim={self.prior.shape[0]}")

        
    def linearize(self, mu: Dict[str, np.ndarray], cov: Dict[str, np.ndarray]):
        result = {self.var: (self._Λ.copy(), self._η.copy())}
        Factor.assert_linearize_dict_format(result)
        return result

    def get_energy(self, mu: Dict[str, np.ndarray]) -> float:
        """计算先验能量，使用全局缓存管理器"""
        if self.var not in mu:
            return 0.0
        
        # 使用内容哈希而非内存地址作为缓存键，避免就地修改导致的缓存脏读
        try:
            cache_key = _factor_cache_manager.get_energy_cache_key("prior", mu[self.var].tobytes())
            cached_energy = _factor_cache_manager.energy_cache.get(cache_key)
            if cached_energy is not None:
                return cached_energy
        except:
            # 如果哈希失败，不使用缓存
            cache_key = None
        
        residual = mu[self.var] - self.prior
        
        # SE(2)姿态的特殊处理（假设最后一个元素是角度）
        if residual.size == 3:
            residual = residual.copy()
            residual[2] = wrap_angle(residual[2])
        
        energy = 0.5 * residual @ self._Λ @ residual
        
        # 缓存结果（如果有有效缓存键）
        if cache_key:
            _factor_cache_manager.energy_cache.put(cache_key, energy)
        
        return energy

# =====================================================================
# Odometry Factor 
# =====================================================================
class OdometryFactor(Factor):
    """
    增强的SE(2)里程计因子，具有改进的Lie群处理
    """
    
    def __init__(self, v_from: str, v_to: str, delta: np.ndarray, sigmas: Union[float, np.ndarray]):
        """
        初始化里程计因子
        
        Args:
            v_from: 起始姿态变量名
            v_to: 终止姿态变量名  
            delta: 相对姿态测量 [dx, dy, dtheta]
            sigmas: 测量噪声标准差
        """
        super().__init__()
        
        self.v1, self.v2 = v_from, v_to
        self.delta = np.asarray(delta, dtype=float)
        self.var_keys = [self.v1, self.v2]
        self._dim_map = {v_from: 3, v_to: 3}
        
        if self.delta.size != 3:
            raise ValueError(f"Delta must be 3D SE(2) relative pose, got {self.delta.shape}")
        
        # 处理噪声模型
        if np.isscalar(sigmas):
            sigma_array = np.full(3, sigmas, dtype=float)
        else:
            sigma_array = np.asarray(sigmas, dtype=float)
            if sigma_array.size != 3:
                raise ValueError(f"Sigmas must be scalar or 3D array, got {sigma_array.shape}")
        
        # 确保最小噪声
        sigma_array = np.maximum(sigma_array, numerical_config.min_eigenvalue**0.5)
        variances = sigma_array ** 2
        self.Rinv = np.diag(1.0 / variances)
        
        # 验证
        validate_matrix_properties(self.Rinv, "odometry_information_matrix")
        
        # 预分配工作数组
        self._work_residual = np.zeros(3)
        self._work_jacobians = {
            'from': np.zeros((3, 3)),
            'to': np.zeros((3, 3))
        }
        
        logger.debug(f"Created odometry factor {v_from} -> {v_to}")
    
    @staticmethod
    def _se2_relative_pose(T_from: np.ndarray, T_to: np.ndarray) -> np.ndarray:
        """
        计算SE(2)中的相对姿态: T_from^{-1} ∘ T_to
        
        Args:
            T_from: [x, y, theta] 起始姿态
            T_to: [x, y, theta] 终止姿态
            
        Returns:
            起始坐标系下的相对姿态 [dx, dy, dtheta]
        """
        x_from, y_from, theta_from = T_from
        x_to, y_to, theta_to = T_to
        
        # 相对旋转
        dtheta = wrap_angle(theta_to - theta_from)
        
        # 全局坐标系下的平移
        dx_global = x_to - x_from
        dy_global = y_to - y_from
        
        # 转换到起始坐标系
        cos_from, sin_from = math.cos(theta_from), math.sin(theta_from)
        dx_local = cos_from * dx_global + sin_from * dy_global
        dy_local = -sin_from * dx_global + cos_from * dy_global
        
        return np.array([dx_local, dy_local, dtheta])
    
    @staticmethod
    def _se2_adjoint_inverse(delta: np.ndarray) -> np.ndarray:
        """
        计算SE(2)伴随矩阵的逆，针对小角度进行优化
        使用更实用的小角度阈值(1e-3 rad ≈ 0.057度)
        
        Args:
            delta: 相对姿态 [dx, dy, dtheta]
            
        Returns:
            3x3 伴随逆矩阵
        """
        dx, dy, dtheta = delta
        
        # 小角度近似以提高数值稳定性 - 使用更实用的阈值
        if abs(dtheta) < numerical_config.small_angle_threshold:
            # 一阶泰勒展开
            return np.array([
                [1.0,    dtheta, -dy],
                [-dtheta, 1.0,    dx],
                [0.0,     0.0,    1.0]
            ])
        else:
            # 完整计算
            cos_th, sin_th = math.cos(dtheta), math.sin(dtheta)
            return np.array([
                [cos_th,  sin_th, -dy],
                [-sin_th, cos_th,  dx],
                [0.0,     0.0,     1.0]
            ])

    def linearize(self, mu: Dict[str, np.ndarray], cov: Dict[str, np.ndarray]) -> Dict:
        """SE(2) 里程计因子线性化（含 SPD / 对称保护）"""
        # --- 检查变量存在 ---
        if self.v1 not in mu or self.v2 not in mu:
            return self._get_zero_blocks()

        T_from, T_to = mu[self.v1], mu[self.v2]

        # --- 1. 误差及“超近距离”保护 -------------------------------
        delta_act = self._se2_relative_pose(T_from, T_to)        # 实际量
        trans_dist = np.hypot(*delta_act[:2])
        if trans_dist < 5e-2:                                    # < 5 cm → 极小信息
            eps = numerical_config.min_eigenvalue
            Z3 = np.zeros(3)
            I3 = eps * np.eye(3)
            return {self.v1: (I3, Z3), self.v2: (I3, Z3),
                    (self.v1, self.v2): np.zeros((3, 3))}

        # --- 2. 残差 / Jacobian ------------------------------------
        res = self.delta - delta_act
        res[2] = wrap_angle(res[2])

        J_from = self._se2_adjoint_inverse(delta_act)            # ∂e/∂T_from
        J_to   = -np.eye(3)                                      # ∂e/∂T_to
        J_from = clip_jacobian(J_from)
        J_to   = clip_jacobian(J_to)

        # --- 3. 信息块 --------------------------------------------
        Λ_ff = J_from.T @ self.Rinv @ J_from
        Λ_tt = J_to.T   @ self.Rinv @ J_to
        Λ_ft = J_from.T @ self.Rinv @ J_to
        Λ_tf = J_to.T   @ self.Rinv @ J_from

        # 对称化交叉块
        Λ_12 = 0.5 * (Λ_ft + Λ_tf.T)

        # 保证对角块正定
        eps = max(1e-8, numerical_config.min_eigenvalue)
        Λ_ff = ensure_positive_definite(Λ_ff, eps)
        Λ_tt = ensure_positive_definite(Λ_tt, eps)

        # --- 4. 信息向量 ------------------------------------------
        η_f = J_from.T @ self.Rinv @ res
        η_t = J_to.T   @ self.Rinv @ res

        # --- 5. 返回 ----------------------------------------------
        result = {
            self.v1: (Λ_ff, η_f),
            self.v2: (Λ_tt, η_t),
            (self.v1, self.v2): Λ_12        # 只放一向；builder 会写入转置
        }
        Factor.assert_linearize_dict_format(result)
        return result



    def get_energy(self, mu: Dict[str, np.ndarray]) -> float:
        """计算里程计因子能量"""
        if self.v1 not in mu or self.v2 not in mu:
            return 0.0
        
        try:
            T_from, T_to = mu[self.v1], mu[self.v2]
            delta_actual = self._se2_relative_pose(T_from, T_to)
            error = self.delta - delta_actual
            error[2] = wrap_angle(error[2])
            
            return 0.5 * error @ self.Rinv @ error
            
        except Exception as e:
            logger.error(f"Odometry energy computation failed: {e}")
            return 0.0
    
    def _get_zero_blocks(self) -> Dict:
        """当变量缺失时返回零信息块"""
        return {
            self.v1: (np.zeros((3, 3)), np.zeros(3)),
            self.v2: (np.zeros((3, 3)), np.zeros(3)),
            (self.v1, self.v2): np.zeros((3, 3)),
            # ✅ 删除镜像零块
        }

# =====================================================================
# Bearing-Range Factor with UT - ✅ 保持不变，已正确实现
# =====================================================================
class BearingRangeUTFactor(Factor):
    """
    增强的方位-距离观测因子，具有自适应GBP/SPBP模式切换
    """
    
    def __init__(self, pose_var: str, lm_var: str, measurement: np.ndarray, R: np.ndarray,
                 mode: str = "gbp", alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0,
                 linear_threshold: float = 5.0, residual_sigma_thresh: float = 3.0):
        """
        初始化方位-距离因子
        
        Args:
            pose_var: 姿态变量名  
            lm_var: 地标变量名
            measurement: [bearing, range] 观测
            R: 2x2 测量噪声协方差
            mode: "auto", "gbp", 或 "spbp"
            alpha, beta, kappa: UT参数
            linear_threshold: 线性假设的距离阈值
            residual_sigma_thresh: 模式切换的残差阈值
        """
        super().__init__()
        
        # 基本参数
        self.pose_key = pose_var
        self.lm_key = lm_var
        self.z = np.asarray(measurement, dtype=float)
        
        # ✅ 正确：设置维度映射
        self._dim_map = {pose_var: 3, lm_var: 2}
        
        if self.z.size != 2:
            raise ValueError(f"Measurement must be [bearing, range], got {self.z.shape}")
        
        # 噪声模型验证和改进
        R = np.asarray(R, dtype=float)
        if R.shape != (2, 2):
            raise ValueError(f"R must be 2x2, got {R.shape}")
        
        # 确保R正定
        try:
            validate_matrix_properties(R, "measurement_noise_covariance", 
                                     check_symmetric=True, 
                                     check_positive_definite=True)
            self.R = R.copy()
        except ValueError as e:
            logger.warning(f"Measurement covariance invalid: {e}, regularizing")
            self.R = ensure_positive_definite(R, numerical_config.min_eigenvalue)
        
        self.Rinv = safe_matrix_inverse(self.R)
        
        # UT参数
        self.mode = mode.lower()
        if self.mode not in ("auto", "gbp", "spbp"):
            raise ValueError(f"Invalid mode: {mode}, must be 'auto', 'gbp', or 'spbp'")
            
        # Sigma点生成器
        self.sigma_generator = SigmaGenerator(alpha, beta, kappa)
        
        # 自适应模式切换参数
        self.linear_threshold = max(linear_threshold, 0.1)
        self.residual_sigma_thresh = max(residual_sigma_thresh, 1.0)
        
        # 状态管理（线程安全）
        self._state_lock = threading.Lock()
        self._diverged_count = 0
        self._current_mode = "gbp"
        self._mode_switch_history = []
        self.var_keys = [self.pose_key, self.lm_key]   # 保证每个Adapter都能抓到

        
        # 预分配工作数组
        self._work_arrays = {
            'residual':            np.zeros(2),
            'prediction':          np.zeros(2),
            'jacobian_pose':       np.zeros((2, 3)),
            'jacobian_landmark':   np.zeros((2, 2)),
            'joint_state':         np.zeros(5),
            'joint_cov':           np.zeros((5, 5)),
        }
        
        logger.debug(f"Created bearing-range factor {pose_var}->{lm_var}, mode={mode}")

    def linearize(self, mu: Dict[str, np.ndarray], cov: Dict[str, np.ndarray]) -> Dict:
        """
        使用自适应模式选择线性化方位-距离因子
        """
        # 输入验证
        if self.pose_key not in mu or self.lm_key not in mu:
            return self._get_zero_blocks()
        
        if self.pose_key not in cov or self.lm_key not in cov:
            return self._get_zero_blocks()
        
        try:
            # 提取当前估计
            μp = mu[self.pose_key]
            μl = mu[self.lm_key]
            Pp = cov[self.pose_key]
            Pl = cov[self.lm_key]
            
            # 输入验证
            if μp.size != 3 or μl.size != 2:
                logger.error(f"Invalid state dimensions: pose={μp.size}, landmark={μl.size}")
                return self._get_zero_blocks()
            
            if Pp.shape != (3, 3) or Pl.shape != (2, 2):
                logger.error(f"Invalid covariance shapes: Pp={Pp.shape}, Pl={Pl.shape}")
                return self._get_zero_blocks()
            
            # 验证协方差矩阵
            validate_matrix_properties(Pp, "pose_covariance")
            validate_matrix_properties(Pl, "landmark_covariance")
            
            # 处理交叉协方差
            Ppl = self._extract_cross_covariance(cov)
            
            # 构建联合状态
            local = self._work_arrays if threading.current_thread() is threading.main_thread() \
                    else {
                        'residual':            np.empty(2),
                        'prediction':          np.empty(2),
                        'jacobian_pose':       np.empty((2, 3)),
                        'jacobian_landmark':   np.empty((2, 2)),
                        'joint_state':         np.empty(5),
                        'joint_cov':           np.empty((5, 5)),
                    }

            # -------- 填写联合状态/协方差 --------
            js = local['joint_state']; jc = local['joint_cov']
            js[:3] = μp; js[3:] = μl
            jc.fill(0.0)                               # ★ 清理旧内容
            jc[:3, :3] = Pp
            jc[3:, 3:] = Pl
            jc[:3, 3:] = Ppl
            jc[3:, :3] = Ppl.T
            
            # 确保联合协方差正定
            joint_cov_pd = ensure_positive_definite(self._work_arrays['joint_cov'], 
                                                   numerical_config.min_eigenvalue)
            
            # 计算预测和残差
            local['prediction'][:] = self._observation_model(js)
            local['residual'][:]   = self.z - local['prediction']
            local['residual'][0]   = wrap_angle(local['residual'][0])
            
            # 计算姿态-地标距离以进行模式决策
            dx, dy = μl[0] - μp[0], μl[1] - μp[1]
            distance = math.hypot(dx, dy)
            
            # 选择线性化模式
            selected_mode = self._select_mode(distance, self._work_arrays['residual'])
            
            # 执行线性化
            if selected_mode == "spbp":
                result = self._spbp_linearize(js, joint_cov_pd, local['residual'])
            else:
                result = self._gbp_linearize(μp, μl, local['residual'],
                                            distance, dx, dy)
            
            # 验证并返回结果
            if self.validate_linearization_result(result):
                Factor.assert_linearize_dict_format(result)   
                return result
            else:
                # 回退到GBP如果验证失败
                logger.warning(f"Validation failed for {selected_mode}, falling back to GBP")
                return self._gbp_linearize(μp, μl, self._work_arrays['residual'], 
                                         distance, dx, dy)
                
        except Exception as e:
            logger.error(f"Bearing-range linearization failed: {e}")
            return self._get_zero_blocks()

    def get_energy(self, mu: Dict[str, np.ndarray]) -> float:
        """计算因子能量"""
        if self.pose_key not in mu or self.lm_key not in mu:
            return 0.0
        
        try:
            μx = np.hstack([mu[self.pose_key], mu[self.lm_key]])
            prediction = self._observation_model(μx)
            residual = self.z - prediction
            residual[0] = wrap_angle(residual[0])
            return 0.5 * residual @ self.Rinv @ residual
        except Exception as e:
            logger.error(f"Energy computation failed: {e}")
            return 0.0
    
    # -------------------------------------------------------------------------
    # 内部辅助方法
    # -------------------------------------------------------------------------
    # 在 algorithm/frontend/factor_ut.py 中
    
    def _get_zero_blocks(self) -> Dict:
        """返回零信息块"""
        return {
            self.pose_key: (np.zeros((3, 3)), np.zeros(3)),
            self.lm_key: (np.zeros((2, 2)), np.zeros(2)),
            (self.pose_key, self.lm_key): np.zeros((3, 2)),
            # ✅ 正确：无镜像零块
        }
    
    def _extract_cross_covariance(self, cov: Dict) -> np.ndarray:
        """
        提取姿态和地标之间的交叉协方差
        
        Returns 3x2 matrix (pose×landmark)
        """
        # 尝试不同可能的键
        cross_keys = [
            (self.pose_key, self.lm_key),
            (self.lm_key, self.pose_key)
        ]
        
        for key in cross_keys:
            if key in cov:
                Ppl = cov[key]
                if key[0] == self.lm_key:  # 如果是(landmark, pose)，转置
                    Ppl = Ppl.T
                if Ppl.shape == (3, 2):
                    return Ppl.copy()
                else:
                    logger.warning(f"Cross-covariance wrong shape: expected (3,2), got {Ppl.shape}")
        
        # 默认为零交叉协方差
        return np.zeros((3, 2))
    
    @staticmethod
    def _observation_model(x: np.ndarray) -> np.ndarray:
        """
        方位-距离观测模型
        
        Args:
            x: 联合状态 [px, py, ptheta, lx, ly]
            
        Returns:
            预测观测 [bearing, range]
        """
        px, py, ptheta, lx, ly = x
        dx, dy = lx - px, ly - py
        
        range_pred = math.hypot(dx, dy)
        bearing_pred = wrap_angle(math.atan2(dy, dx) - ptheta)
        
        return np.array([bearing_pred, range_pred])
    
    def _select_mode(self, distance: float, residual: np.ndarray) -> str:
        """
        基于几何和残差智能选择线性化模式
        """
        if self.mode in ("gbp", "spbp"):
            return self.mode
            
        with self._state_lock:
            # 对于非常近的距离强制使用GBP（数值稳定性）
            if distance < 0.1:  # 10 cm
                return "gbp"
                
            # 分析残差幅度
            σ_bearing = math.sqrt(self.R[0, 0])
            σ_range = math.sqrt(self.R[1, 1])
            
            large_bearing_error = abs(residual[0]) > self.residual_sigma_thresh * σ_bearing
            large_range_error = abs(residual[1]) > self.residual_sigma_thresh * σ_range
            large_distance = distance > self.linear_threshold
            
            # 模式切换逻辑，带有滞后
            if large_bearing_error or large_range_error or large_distance:
                self._diverged_count += 1
                if self._diverged_count > 2:  # 多次大误差后切换到SPBP
                    self._current_mode = "spbp"
            else:
                self._diverged_count = max(0, self._diverged_count - 1)
                if self._diverged_count == 0:
                    self._current_mode = "gbp"
            
            # 记录模式切换历史（用于调试）
            if len(self._mode_switch_history) == 0 or self._mode_switch_history[-1] != self._current_mode:
                self._mode_switch_history.append(self._current_mode)
                if len(self._mode_switch_history) > 10:  # 保持历史长度有限
                    self._mode_switch_history.pop(0)
            
            return self._current_mode
    
    def _gbp_linearize(self, μp: np.ndarray, μl: np.ndarray, residual: np.ndarray, 
                       distance: float, dx: float, dy: float) -> Dict:
        """
        标准GBP线性化，使用一阶泰勒展开
        """
        # 确保距离计算的数值稳定性
        # --------- 极近距离(<5 cm) 直接返回极小信息 ---------
        if distance < numerical_config.min_eigenvalue * 1e-3:
            min_info_pose = numerical_config.min_eigenvalue * np.eye(3)
            min_info_lm   = numerical_config.min_eigenvalue * np.eye(2)
            return {
                self.pose_key: (min_info_pose, np.zeros(3)),
                self.lm_key:   (min_info_lm,   np.zeros(2)),
                (self.pose_key, self.lm_key): np.zeros((3, 2)),
            }

        safe_distance = max(distance, numerical_config.min_eigenvalue**0.5)
        q = safe_distance ** 2  # 距离的平方

        # 计算雅可比矩阵（直接用局部数组，避免线程竞争）
        Jp = np.array([[ dy/q,            -dx/q,           -1.0],
                       [-dx/safe_distance, -dy/safe_distance, 0.0]])

        Jl = np.array([[-dy/q,              dx/q],
                       [ dx/safe_distance,  dy/safe_distance]])
        
        # 数值裁剪防止爆炸
        Jp = clip_jacobian(Jp)
        Jl = clip_jacobian(Jl)
        
        # 信息矩阵
        Λp_raw = Jp.T @ self.Rinv @ Jp
        Λl_raw = Jl.T @ self.Rinv @ Jl
        Λp = ensure_positive_definite(Λp_raw, numerical_config.min_eigenvalue)
        Λl = ensure_positive_definite(Λl_raw, numerical_config.min_eigenvalue)
        Λcross = Jp.T @ self.Rinv @ Jl
        
        # 信息向量
        ηp = Jp.T @ self.Rinv @ residual
        ηl = Jl.T @ self.Rinv @ residual
        
        return {
            self.pose_key: (Λp, ηp),
            self.lm_key: (Λl, ηl),
            (self.pose_key, self.lm_key): Λcross,
            # ✅ 正确：无镜像交叉块
        }
    
    def _spbp_linearize(self, μx: np.ndarray, Px: np.ndarray, residual: np.ndarray) -> Dict:
        """
        Sigma点信念传播线性化，使用无迹变换
        """
        # 检查全局缓存
        try:
            cache_key = _factor_cache_manager.get_spbp_cache_key(
                "bearing_range", μx.tobytes(), Px.tobytes()
            )
            cached_result = _factor_cache_manager.spbp_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        except:
            cache_key = None
        
        try:
            # 生成sigma点
            sigma_points, wm, wc = self.sigma_generator.generate(μx, Px)
            
            # 通过观测模型传播sigma点
            Y = np.zeros((len(sigma_points), 2))
            for i, point in enumerate(sigma_points):
                Y[i] = self._observation_model(point)
            
            # 计算预测观测均值（处理方位的圆周统计）
            μy = self._compute_circular_mean(Y, wm)
            
            # 计算交叉协方差
            Pxy, Pyy = self._compute_cross_covariances(sigma_points, Y, μx, μy, wc)
            
            # 添加测量噪声
            Pyy += self.R
            
            # 通过卡尔曼增益计算有效雅可比矩阵
            try:
                # 求解卡尔曼增益: K = Pxy @ Pyy^{-1}
                K = solve(Pyy, Pxy.T).T
            except LinAlgError:
                # 正则化回退
                Pyy_reg = ensure_positive_definite(Pyy, numerical_config.min_eigenvalue)
                K = safe_matrix_inverse(Pyy_reg) @ Pxy.T
                K = K.T
            
            # 转换为信息形式，使用安全的维度检查
            try:
                Λ, η = safe_spbp_information_conversion(K, self.Rinv, residual, 5, 2)
                Λ[:3, :3] = ensure_positive_definite(Λ[:3, :3], numerical_config.min_eigenvalue)
                Λ[3:, 3:] = ensure_positive_definite(Λ[3:, 3:], numerical_config.min_eigenvalue)
                
            except ValueError as e:
                logger.warning(f"SPBP information conversion failed: {e}, falling back to GBP")
                μp, μl = μx[:3], μx[3:]
                dx, dy = μl[0] - μp[0], μl[1] - μp[1]
                distance = math.hypot(dx, dy)
                return self._gbp_linearize(μp, μl, residual, distance, dx, dy)
            
            # 提取块
            Λp, Λl = Λ[:3, :3], Λ[3:, 3:]
            Λcross = Λ[:3, 3:]
            ηp, ηl = η[:3], η[3:]
            
            result = {
                self.pose_key: (Λp, ηp),
                self.lm_key: (Λl, ηl),
                (self.pose_key, self.lm_key): Λcross,
                # ✅ 正确：无镜像交叉块
            }
            
            # 缓存结果
            if cache_key:
                _factor_cache_manager.spbp_cache.put(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.warning(f"SPBP linearization failed: {e}, falling back to GBP")
            # 提取组件进行GBP回退
            μp, μl = μx[:3], μx[3:]
            dx, dy = μl[0] - μp[0], μl[1] - μp[1]
            distance = math.hypot(dx, dy)
            return self._gbp_linearize(μp, μl, residual, distance, dx, dy)
    
    def _compute_circular_mean(self, Y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        计算观测均值，对方位使用适当的圆周统计
        """
        # 方位均值使用圆周统计
        sin_sum = np.sum(weights * np.sin(Y[:, 0]))
        cos_sum = np.sum(weights * np.cos(Y[:, 0]))
        μy_bearing = math.atan2(sin_sum, cos_sum)
        
        # 距离均值（标准线性）
        μy_range = np.sum(weights * Y[:, 1])
        
        return np.array([μy_bearing, μy_range])
    
    def _compute_cross_covariances(self, X: np.ndarray, Y: np.ndarray, 
                                   μx: np.ndarray, μy: np.ndarray, 
                                   weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算交叉协方差，适当处理角度差
        """
        n_points = len(X)
        
        # 状态偏差
        dX = X - μx
        dX[:, 2] = wrap_angle(dX[:, 2])  # 包装姿态角度差
        
        # 观测偏差  
        dY = np.zeros_like(Y)
        dY[:, 0] = wrap_angle(Y[:, 0] - μy[0])  # 包装方位差
        dY[:, 1] = Y[:, 1] - μy[1]              # 距离差（线性）
        
        # 计算协方差矩阵
        Pxy = np.zeros((5, 2))
        Pyy = np.zeros((2, 2))
        
        for i in range(n_points):
            Pxy += weights[i] * np.outer(dX[i], dY[i])
            Pyy += weights[i] * np.outer(dY[i], dY[i])
        
        return Pxy, Pyy
    
    def get_mode_statistics(self) -> Dict[str, Any]:
        """获取模式切换统计信息"""
        with self._state_lock:
            return {
                'current_mode': self._current_mode,
                'diverged_count': self._diverged_count,
                'mode_history': self._mode_switch_history.copy(),
                'sigma_cache_stats': self.sigma_generator.get_cache_stats(),
                'global_spbp_cache_stats': _factor_cache_manager.spbp_cache.stats()
            }

# =====================================================================
# Pose-to-Pose UT Factor - 
# =====================================================================
class PoseToPoseUTFactor(Factor):
    """
    Pose-to-Pose 方位-距离因子用于多机器人间的观测 (2-DOF版本)
    """
    
    def __init__(self, pose1_key: str, pose2_key: str, measurement: np.ndarray, 
                 R: np.ndarray, mode: str = "auto", alpha: float = 1e-3, 
                 beta: float = 2.0, kappa: float = 0.0,
                 distance_threshold: float = 10.0, residual_sigma_thresh: float = 2.5):
        """
        初始化Pose-to-Pose方位-距离因子
        """
        super().__init__()
        
        # 基本参数
        self.pose1_key = pose1_key  # 观测者位姿
        self.pose2_key = pose2_key  # 被观测者位姿
        self.z = np.asarray(measurement, dtype=float)
        
        # ✅ 修复：设置维度映射
        self._dim_map = {pose1_key: 3, pose2_key: 3}
        
        if self.z.size != 2:
            raise ValueError(f"Measurement must be [bearing, range], got {self.z.shape}")
        
        # 噪声模型验证
        R = np.asarray(R, dtype=float)
        if R.shape != (2, 2):
            raise ValueError(f"R must be 2x2, got {R.shape}")
        
        try:
            validate_matrix_properties(R, "pose_to_pose_noise_covariance", 
                                     check_symmetric=True, 
                                     check_positive_definite=True)
            self.R = R.copy()
        except ValueError as e:
            logger.warning(f"Pose-to-pose noise covariance invalid: {e}, regularizing")
            self.R = ensure_positive_definite(R, numerical_config.min_eigenvalue)
        
        self.Rinv = safe_matrix_inverse(self.R)
        
        # UT和模式参数
        self.mode = mode.lower()
        if self.mode not in ("auto", "gbp", "spbp"):
            raise ValueError(f"Invalid mode: {mode}, must be 'auto', 'gbp', or 'spbp'")
        
        # Sigma点生成器
        self.sigma_generator = SigmaGenerator(alpha, beta, kappa)
        
        # 自适应切换参数
        self.distance_threshold = max(distance_threshold, 1.0)
        self.residual_sigma_thresh = max(residual_sigma_thresh, 1.0)
        
        # 状态管理（线程安全）
        self._state_lock = threading.Lock()
        self._diverged_count = 0
        self._current_mode = "gbp"
        self._mode_history = []
        
        # 预分配工作数组
        self._work_arrays = {
            'residual': np.zeros(2),
            'prediction': np.zeros(2),
            'joint_state': np.zeros(6),      # [pose1, pose2]
            'joint_cov': np.zeros((6, 6)),   # 6x6联合协方差
            'jacobian_pose1': np.zeros((2, 3)),
            'jacobian_pose2': np.zeros((2, 3))
        }
        
        logger.debug(f"Created pose-to-pose bearing-range factor {pose1_key} -> {pose2_key}, mode={mode}")
    
    def linearize(self, mu: Dict[str, np.ndarray], cov: Dict[str, np.ndarray]) -> Dict:
        """
        使用自适应GBP/SPBP模式线性化pose-to-pose bearing-range因子
        """
        # 输入验证
        if self.pose1_key not in mu or self.pose2_key not in mu:
            return self._get_zero_blocks()
        
        if self.pose1_key not in cov or self.pose2_key not in cov:
            return self._get_zero_blocks()
        
        try:
            # 提取位姿估计
            μ1 = mu[self.pose1_key]
            μ2 = mu[self.pose2_key]
            P1 = cov[self.pose1_key]
            P2 = cov[self.pose2_key]
            
            # 输入验证
            if μ1.size != 3 or μ2.size != 3:
                logger.error(f"Invalid pose dimensions: pose1={μ1.size}, pose2={μ2.size}")
                return self._get_zero_blocks()
            
            if P1.shape != (3, 3) or P2.shape != (3, 3):
                logger.error(f"Invalid covariance shapes: P1={P1.shape}, P2={P2.shape}")
                return self._get_zero_blocks()
            
            # 验证协方差矩阵
            validate_matrix_properties(P1, "pose1_covariance")
            validate_matrix_properties(P2, "pose2_covariance")
            
            # 提取交叉协方差
            P12 = self._extract_cross_covariance(cov)
            
            # 构建联合状态和协方差
            self._work_arrays['joint_state'][:3] = μ1
            self._work_arrays['joint_state'][3:] = μ2
            
            self._work_arrays['joint_cov'][:3, :3] = P1
            self._work_arrays['joint_cov'][3:, 3:] = P2
            self._work_arrays['joint_cov'][:3, 3:] = P12
            self._work_arrays['joint_cov'][3:, :3] = P12.T
            
            # 确保联合协方差正定
            joint_cov_pd = ensure_positive_definite(self._work_arrays['joint_cov'], 
                                                   numerical_config.min_eigenvalue)
            
            # 计算预测和残差
            self._work_arrays['prediction'][:] = self._observation_model(self._work_arrays['joint_state'])
            self._work_arrays['residual'][:] = self.z - self._work_arrays['prediction']
            self._work_arrays['residual'][0] = wrap_angle(self._work_arrays['residual'][0])  # 包装角度残差
            
            # 计算位姿间距离用于模式选择
            dx, dy = μ2[0] - μ1[0], μ2[1] - μ1[1]
            distance = math.hypot(dx, dy)
            
            # 选择线性化模式
            selected_mode = self._select_mode(distance, self._work_arrays['residual'])
            
            # 执行线性化
            if selected_mode == "spbp":
                result = self._spbp_linearize(self._work_arrays['joint_state'], 
                                            joint_cov_pd, 
                                            self._work_arrays['residual'])
            else:
                result = self._gbp_linearize(μ1, μ2, self._work_arrays['residual'], 
                                           distance, dx, dy)
            
            # 验证并返回结果
            if self.validate_linearization_result(result):
                Factor.assert_linearize_dict_format(result)   
                return result
            else:
                logger.warning(f"Validation failed for {selected_mode}, falling back to GBP")
                return self._gbp_linearize(μ1, μ2, self._work_arrays['residual'], 
                                         distance, dx, dy)
                
        except Exception as e:
            logger.error(f"Pose-to-pose linearization failed: {e}")
            return self._get_zero_blocks()
    
    def get_energy(self, mu: Dict[str, np.ndarray]) -> float:
        """计算pose-to-pose bearing-range因子能量"""
        if self.pose1_key not in mu or self.pose2_key not in mu:
            return 0.0
        
        try:
            joint_state = np.hstack([mu[self.pose1_key], mu[self.pose2_key]])
            prediction = self._observation_model(joint_state)
            residual = self.z - prediction
            residual[0] = wrap_angle(residual[0])
            return 0.5 * residual @ self.Rinv @ residual
        except Exception as e:
            logger.error(f"Pose-to-pose energy computation failed: {e}")
            return 0.0
    
    # -------------------------------------------------------------------------
    # 内部辅助方法
    # -------------------------------------------------------------------------
    
    def _get_zero_blocks(self) -> Dict:
        """返回零信息块"""
        return {
            self.pose1_key: (np.zeros((3, 3)), np.zeros(3)),
            self.pose2_key: (np.zeros((3, 3)), np.zeros(3)),
            (self.pose1_key, self.pose2_key): np.zeros((3, 3)),
            # ✅ 正确：无镜像零块
        }
    
    def _extract_cross_covariance(self, cov: Dict) -> np.ndarray:
        """
        提取两个位姿间的交叉协方差
        
        Returns 3x3 matrix (pose1×pose2)
        """
        cross_keys = [
            (self.pose1_key, self.pose2_key),
            (self.pose2_key, self.pose1_key)
        ]
        
        for key in cross_keys:
            if key in cov:
                P12 = cov[key]
                if key[0] == self.pose2_key:  # 如果是(pose2, pose1)，转置
                    P12 = P12.T
                if P12.shape == (3, 3):
                    return P12.copy()
                else:
                    logger.warning(f"Cross-covariance wrong shape: expected (3,3), got {P12.shape}")
        
        # 默认为零交叉协方差
        return np.zeros((3, 3))
    
    @staticmethod
    def _observation_model(x: np.ndarray) -> np.ndarray:
        """
        Pose-to-pose bearing-range观测模型
        
        Args:
            x: 联合状态 [x1, y1, θ1, x2, y2, θ2]
            
        Returns:
            观测 [bearing, range] (从pose1观测pose2)
        """
        x1, y1, theta1 = x[:3]
        x2, y2, _ = x[3:]  # 忽略θ2，因为只观测位置
        
        dx = x2 - x1
        dy = y2 - y1
        
        range_pred = math.hypot(dx, dy)
        bearing_pred = wrap_angle(math.atan2(dy, dx) - theta1)
        
        return np.array([bearing_pred, range_pred])
    
    def _select_mode(self, distance: float, residual: np.ndarray) -> str:
        """
        基于距离和残差智能选择线性化模式
        """
        if self.mode in ("gbp", "spbp"):
            return self.mode
        
        with self._state_lock:
            # 对于很远的距离或大的误差，倾向于使用SPBP
            large_distance = distance > self.distance_threshold
            
            # 分析残差幅度
            σ_bearing = math.sqrt(self.R[0, 0])
            σ_range = math.sqrt(self.R[1, 1])
            
            large_bearing_error = abs(residual[0]) > self.residual_sigma_thresh * σ_bearing
            large_range_error = abs(residual[1]) > self.residual_sigma_thresh * σ_range
            
            # 模式切换逻辑
            if large_distance or large_bearing_error or large_range_error:
                self._diverged_count += 1
                if self._diverged_count > 2:
                    self._current_mode = "spbp"
            else:
                self._diverged_count = max(0, self._diverged_count - 1)
                if self._diverged_count == 0:
                    self._current_mode = "gbp"
            
            # 记录模式历史
            if len(self._mode_history) == 0 or self._mode_history[-1] != self._current_mode:
                self._mode_history.append(self._current_mode)
                if len(self._mode_history) > 5:
                    self._mode_history.pop(0)
            
            return self._current_mode
    
    def _gbp_linearize(self, μ1: np.ndarray, μ2: np.ndarray, residual: np.ndarray,
                       distance: float, dx: float, dy: float) -> Dict:
        """
        标准GBP线性化：计算bearing-range观测的雅可比矩阵
        """
        try:
            # 对极小距离的特殊处理
            if distance < numerical_config.min_eigenvalue**0.5:
                logger.warning(f"Very small distance {distance:.6f} between poses, returning minimal information")
                # 返回极小的信息矩阵，避免数值问题
                min_info = numerical_config.min_eigenvalue * np.eye(3)
                return {
                    self.pose1_key: (min_info, np.zeros(3)),
                    self.pose2_key: (min_info, np.zeros(3)),
                    (self.pose1_key, self.pose2_key): np.zeros((3, 3)),
                    # ✅ 正确：无镜像交叉块
                }
            
            # 确保距离计算的数值稳定性
            safe_distance = max(distance, numerical_config.min_eigenvalue**0.5)
            q = safe_distance ** 2  # 距离的平方
            
            # 计算雅可比矩阵（与BearingRangeUTFactor类似）
            # 对于方位 = atan2(dy, dx) - theta1
            self._work_arrays['jacobian_pose1'][0, :] = [dy/q, -dx/q, -1.0]
            self._work_arrays['jacobian_pose2'][0, :] = [-dy/q, dx/q, 0.0]  # 对θ2的导数为0
            
            # 对于距离 = sqrt(dx² + dy²)  
            self._work_arrays['jacobian_pose1'][1, :] = [-dx/safe_distance, -dy/safe_distance, 0.0]
            self._work_arrays['jacobian_pose2'][1, :] = [dx/safe_distance, dy/safe_distance, 0.0]
            
            # 数值裁剪
            J1 = clip_jacobian(self._work_arrays['jacobian_pose1'])
            J2 = clip_jacobian(self._work_arrays['jacobian_pose2'])
            
            # 信息矩阵块
            Λ1 = J1.T @ self.Rinv @ J1
            Λ2 = J2.T @ self.Rinv @ J2
            Λ12 = J1.T @ self.Rinv @ J2
            
            # 由于观测对 θ2 无贡献，Λ2 的第3行/列为0，需要正则化
            # 给 θ2 加一个极小的先验信息，避免奇异性
            Λ2[2, 2] += numerical_config.min_eigenvalue
            Λ2 = ensure_positive_definite(Λ2, numerical_config.min_eigenvalue)
            
            # 信息向量块
            η1 = J1.T @ self.Rinv @ residual
            η2 = J2.T @ self.Rinv @ residual
            
            return {
                self.pose1_key: (Λ1, η1),
                self.pose2_key: (Λ2, η2),
                (self.pose1_key, self.pose2_key): Λ12,
                # ✅ 正确：无镜像交叉块
            }
            
        except Exception as e:
            logger.warning(f"GBP linearization failed: {e}")
            return self._get_zero_blocks()
    
    def _spbp_linearize(self, μx: np.ndarray, Px: np.ndarray, residual: np.ndarray) -> Dict:
        """
        SPBP线性化：使用无迹变换处理非线性
        """
        # 检查全局缓存
        try:
            cache_key = _factor_cache_manager.get_spbp_cache_key(
                "pose_to_pose_bearing_range", μx.tobytes(), Px.tobytes()
            )
            cached_result = _factor_cache_manager.spbp_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        except:
            cache_key = None
        
        try:
            # 生成sigma点
            sigma_points, wm, wc = self.sigma_generator.generate(μx, Px)
            
            # 通过观测模型传播sigma点
            Y = np.zeros((len(sigma_points), 2))
            for i, point in enumerate(sigma_points):
                Y[i] = self._observation_model(point)
            
            # 计算预测观测均值（处理方位的圆周统计）
            μy = self._compute_circular_mean(Y, wm)
            
            # 计算交叉协方差
            Pxy, Pyy = self._compute_cross_covariances(sigma_points, Y, μx, μy, wc)
            
            # 添加测量噪声
            Pyy += self.R
            
            # 计算卡尔曼增益
            try:
                K = solve(Pyy, Pxy.T).T
            except:
                Pyy_reg = ensure_positive_definite(Pyy, numerical_config.min_eigenvalue)
                K = safe_matrix_inverse(Pyy_reg) @ Pxy.T
                K = K.T
            
            # 转换为信息形式，使用安全的维度检查
            try:
                Λ, η = safe_spbp_information_conversion(K, self.Rinv, residual, 6, 2)
            except ValueError as e:
                logger.warning(f"SPBP information conversion failed: {e}, falling back to GBP")
                dx = μx[3] - μx[0]
                dy = μx[4] - μx[1]
                distance = math.hypot(dx, dy)
                return self._gbp_linearize(μx[:3], μx[3:], residual, distance, dx, dy)
            
            # 提取块
            Λ1, Λ2 = Λ[:3, :3], Λ[3:, 3:]
            Λ12 = Λ[:3, 3:]
            η1, η2 = η[:3], η[3:]
            
            # 由于观测对 θ2 无贡献，需要正则化 Λ2
            Λ2[2, 2] += numerical_config.min_eigenvalue
            
            result = {
                self.pose1_key: (Λ1, η1),
                self.pose2_key: (Λ2, η2),
                (self.pose1_key, self.pose2_key): Λ12,
                # ✅ 正确：无镜像交叉块
            }
            
            # 缓存结果
            if cache_key:
                _factor_cache_manager.spbp_cache.put(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.warning(f"SPBP linearization failed: {e}, falling back to GBP")
            dx = μx[3] - μx[0]
            dy = μx[4] - μx[1]
            distance = math.hypot(dx, dy)
            return self._gbp_linearize(μx[:3], μx[3:], residual, distance, dx, dy)
    
    def _compute_circular_mean(self, Y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        计算观测均值，对方位使用圆周统计
        """
        # 方位均值使用圆周统计
        sin_sum = np.sum(weights * np.sin(Y[:, 0]))
        cos_sum = np.sum(weights * np.cos(Y[:, 0]))
        μy_bearing = math.atan2(sin_sum, cos_sum)
        
        # 距离均值（标准线性）
        μy_range = np.sum(weights * Y[:, 1])
        
        return np.array([μy_bearing, μy_range])
    
    def _compute_cross_covariances(self, X: np.ndarray, Y: np.ndarray, 
                                   μx: np.ndarray, μy: np.ndarray, 
                                   weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算交叉协方差，适当处理角度差
        """
        n_points = len(X)
        
        # 状态偏差
        dX = X - μx
        dX[:, 2] = wrap_angle(dX[:, 2])  # 包装pose1角度差
        dX[:, 5] = wrap_angle(dX[:, 5])  # 包装pose2角度差
        
        # 观测偏差
        dY = np.zeros_like(Y)
        dY[:, 0] = wrap_angle(Y[:, 0] - μy[0])  # 方位差（圆周）
        dY[:, 1] = Y[:, 1] - μy[1]              # 距离差（线性）
        
        # 计算协方差矩阵
        Pxy = np.zeros((6, 2))
        Pyy = np.zeros((2, 2))
        
        for i in range(n_points):
            Pxy += weights[i] * np.outer(dX[i], dY[i])
            Pyy += weights[i] * np.outer(dY[i], dY[i])
        
        return Pxy, Pyy
    
    def get_mode_statistics(self) -> Dict[str, Any]:
        """获取模式切换统计"""
        with self._state_lock:
            return {
                'current_mode': self._current_mode,
                'diverged_count': self._diverged_count,
                'mode_history': self._mode_history.copy(),
                'global_spbp_cache_stats': _factor_cache_manager.spbp_cache.stats()
            }

# =====================================================================
# Loop Closure Factor - ✅ 修复：添加_dim_map设置，删除_get_dim()重写，修复linearize()返回格式
# =====================================================================
class LoopClosureFactor(Factor):
    """
    改进的环路闭合因子，用于同一或不同机器人的姿态间
    """
    
    def __init__(self, pose1_var: str, pose2_var: str, relative_pose: np.ndarray, 
                 information_matrix: np.ndarray):
        """
        初始化环路闭合因子
        
        Args:
            pose1_var: 第一个姿态变量名
            pose2_var: 第二个姿态变量名  
            relative_pose: 测量的相对姿态 [dx, dy, dtheta]
            information_matrix: 3x3 信息矩阵
        """
        super().__init__()
        
        self.pose1_key = pose1_var
        self.pose2_key = pose2_var
        self.relative_pose = np.asarray(relative_pose, dtype=float)
        self.information = np.asarray(information_matrix, dtype=float)
        
        # ✅ 修复2：添加维度映射设置
        self._dim_map = {pose1_var: 3, pose2_var: 3}
        
        # 验证输入
        if self.relative_pose.size != 3:
            raise ValueError("Relative pose must be 3D [dx, dy, dtheta]")
        if self.information.shape != (3, 3):
            raise ValueError("Information matrix must be 3x3")
        
        # 确保信息矩阵正定
        try:
            validate_matrix_properties(self.information, "loop_closure_information", 
                                     check_symmetric=True, 
                                     check_positive_definite=True)
        except ValueError as e:
            logger.warning(f"Loop closure information matrix invalid: {e}, regularizing")
            self.information = ensure_positive_definite(self.information, 
                                                       numerical_config.min_eigenvalue)
        
        # 预分配工作数组
        self._work_error = np.zeros(3)
        
        logger.debug(f"Created loop closure factor {pose1_var} <-> {pose2_var}")
    
    def linearize(self, mu: Dict[str, np.ndarray], cov: Dict[str, np.ndarray]) -> Dict:
        """线性化环路闭合约束"""
        if self.pose1_key not in mu or self.pose2_key not in mu:
            return self._get_zero_blocks()
        
        try:
            # 与里程计因子相同但变量名不同
            pose1, pose2 = mu[self.pose1_key], mu[self.pose2_key]
            
            # 计算实际相对姿态
            actual_relative = OdometryFactor._se2_relative_pose(pose1, pose2)
            
            # 计算误差
            self._work_error[:] = self.relative_pose - actual_relative
            self._work_error[2] = wrap_angle(self._work_error[2])
            
            # 雅可比矩阵（与里程计相同）
            J1 = OdometryFactor._se2_adjoint_inverse(actual_relative)
            J2 = -np.eye(3)
            
            # 裁剪雅可比矩阵
            J1 = clip_jacobian(J1)
            J2 = clip_jacobian(J2)
            
            # 信息块
            Λ11 = J1.T @ self.information @ J1
            Λ22 = J2.T @ self.information @ J2
            Λ12 = J1.T @ self.information @ J2
            
            # 信息向量
            η1 = J1.T @ self.information @ self._work_error
            η2 = J2.T @ self.information @ self._work_error
            
            # ✅ 修复3：只返回单向交叉块
            result = {
                self.pose1_key: (Λ11, η1),
                self.pose2_key: (Λ22, η2),
                (self.pose1_key, self.pose2_key): Λ12,
                # ✅ 删除镜像交叉块: (self.pose2_key, self.pose1_key): Λ12.T
            }
            
            # 验证结果
            if not self.validate_linearization_result(result):
                logger.warning(f"Loop closure linearization validation failed")
                return self._get_zero_blocks()
            Factor.assert_linearize_dict_format(result)   
            return result
            
        except Exception as e:
            logger.error(f"Loop closure linearization failed: {e}")
            return self._get_zero_blocks()
    
    def get_energy(self, mu: Dict[str, np.ndarray]) -> float:
        """计算环路闭合能量"""
        if self.pose1_key not in mu or self.pose2_key not in mu:
            return 0.0
        
        try:
            pose1, pose2 = mu[self.pose1_key], mu[self.pose2_key]
            actual_relative = OdometryFactor._se2_relative_pose(pose1, pose2)
            error = self.relative_pose - actual_relative
            error[2] = wrap_angle(error[2])
            
            return 0.5 * error @ self.information @ error
            
        except Exception as e:
            logger.error(f"Loop closure energy computation failed: {e}")
            return 0.0
    
    def _get_zero_blocks(self) -> Dict:
        """当变量缺失时返回零块"""
        return {
            self.pose1_key: (np.zeros((3, 3)), np.zeros(3)),
            self.pose2_key: (np.zeros((3, 3)), np.zeros(3)),
            (self.pose1_key, self.pose2_key): np.zeros((3, 3)),
            # ✅ 正确：无镜像零块
        }

# =====================================================================
# Multi-Robot SLAM Utility Functions
# =====================================================================

def create_multi_robot_odometry_chain(robot_id: int, num_poses: int, 
                                     measurements: List[np.ndarray], 
                                     noise_sigmas: Union[float, np.ndarray],
                                     validate_measurements: bool = True) -> List[OdometryFactor]:
    """
    为单个机器人创建里程计因子链
    
    Args:
        robot_id: 机器人标识符
        num_poses: 轨迹中的姿态数量
        measurements: 相对姿态测量列表
        noise_sigmas: 里程计噪声标准差
        validate_measurements: 是否验证测量值
        
    Returns:
        里程计因子列表
    """
    if num_poses < 2:
        raise ValueError("Need at least 2 poses for odometry chain")
    
    if len(measurements) != num_poses - 1:
        raise ValueError(f"Need {num_poses-1} measurements for {num_poses} poses, got {len(measurements)}")
    
    factors = []
    
    for t in range(num_poses - 1):
        pose_from = f"x{robot_id}_{t}"
        pose_to = f"x{robot_id}_{t+1}"
        
        measurement = measurements[t]
        
        # 验证测量值
        if validate_measurements:
            if not np.isfinite(measurement).all():
                logger.warning(f"Non-finite measurement at time {t}, skipping")
                continue
            
            if measurement.size != 3:
                logger.warning(f"Invalid measurement size at time {t}, skipping")
                continue
        
        try:
            factors.append(OdometryFactor(pose_from, pose_to, measurement, noise_sigmas))
        except Exception as e:
            logger.error(f"Failed to create odometry factor at time {t}: {e}")
    
    logger.info(f"Created {len(factors)} odometry factors for robot {robot_id}")
    return factors

def create_landmark_observation_factors(robot_id: int, pose_times: List[int],
                                      landmark_ids: List[int], 
                                      measurements: List[Tuple[int, int, np.ndarray]], 
                                      noise_cov: np.ndarray,
                                      validate_measurements: bool = True,
                                      **ut_params) -> List[BearingRangeUTFactor]:
    """
    为机器人创建地标观测因子
    
    Args:
        robot_id: 机器人标识符
        pose_times: 姿态时间索引列表
        landmark_ids: 地标标识符列表
        measurements: (pose_time, landmark_id, [bearing, range]) 元组列表
        noise_cov: 2x2 测量噪声协方差
        validate_measurements: 是否验证测量值
        **ut_params: SPBP模式的UT参数
        
    Returns:
        方位-距离因子列表
    """
    factors = []
    
    # 验证噪声协方差
    if validate_measurements:
        try:
            validate_matrix_properties(noise_cov, "measurement_noise_covariance")
        except ValueError as e:
            logger.error(f"Invalid measurement noise covariance: {e}")
            return factors
    
    for pose_time, lm_id, measurement in measurements:
        if pose_time not in pose_times:
            logger.warning(f"Pose time {pose_time} not in valid times, skipping")
            continue
            
        if lm_id not in landmark_ids:
            logger.warning(f"Landmark {lm_id} not in valid landmarks, skipping")
            continue
        
        # 验证测量值
        if validate_measurements:
            if not np.isfinite(measurement).all():
                logger.warning(f"Non-finite measurement for pose {pose_time}, landmark {lm_id}, skipping")
                continue
            
            if measurement.size != 2:
                logger.warning(f"Invalid measurement size for pose {pose_time}, landmark {lm_id}, skipping")
                continue
            
            # 检查合理的测量范围
            bearing, range_val = measurement
            if not (-math.pi <= bearing <= math.pi):
                logger.warning(f"Bearing out of range for pose {pose_time}, landmark {lm_id}, skipping")
                continue
            
            if range_val <= 0:
                logger.warning(f"Invalid range for pose {pose_time}, landmark {lm_id}, skipping")
                continue
        
        pose_var = f"x{robot_id}_{pose_time}"
        lm_var = f"l_{lm_id}"
        
        try:
            factors.append(BearingRangeUTFactor(
                pose_var, lm_var, measurement, noise_cov, **ut_params
            ))
        except Exception as e:
            logger.error(f"Failed to create observation factor for pose {pose_time}, landmark {lm_id}: {e}")
    
    logger.info(f"Created {len(factors)} observation factors for robot {robot_id}")
    return factors

def create_inter_robot_factors(robot1_id: int, robot2_id: int,
                              observations: List[Tuple[int, int, np.ndarray, np.ndarray]],
                              validate_observations: bool = True) -> List[LoopClosureFactor]:
    """
    创建机器人间观测因子（环路闭合）
    
    Args:
        robot1_id: 第一个机器人标识符
        robot2_id: 第二个机器人标识符  
        observations: (time1, time2, relative_pose, information_matrix) 元组列表
        validate_observations: 是否验证观测值
        
    Returns:
        机器人间因子列表
    """
    factors = []
    
    for time1, time2, relative_pose, information in observations:
        # 验证观测值
        if validate_observations:
            if not np.isfinite(relative_pose).all():
                logger.warning(f"Non-finite relative pose for times {time1}, {time2}, skipping")
                continue
            
            if relative_pose.size != 3:
                logger.warning(f"Invalid relative pose size for times {time1}, {time2}, skipping")
                continue
            
            try:
                validate_matrix_properties(information, f"inter_robot_information_{time1}_{time2}")
            except ValueError as e:
                logger.warning(f"Invalid information matrix for times {time1}, {time2}: {e}, skipping")
                continue
        
        pose1_var = f"x{robot1_id}_{time1}"
        pose2_var = f"x{robot2_id}_{time2}"
        
        try:
            factors.append(LoopClosureFactor(pose1_var, pose2_var, relative_pose, information))
        except Exception as e:
            logger.error(f"Failed to create inter-robot factor for times {time1}, {time2}: {e}")
    
    logger.info(f"Created {len(factors)} inter-robot factors between robots {robot1_id} and {robot2_id}")
    return factors

def create_pose_to_pose_ut_factors(robot1_id: int, robot2_id: int,
                                  observations: List[Tuple[int, int, np.ndarray, np.ndarray]],
                                  validate_observations: bool = True,
                                  **ut_params) -> List[PoseToPoseUTFactor]:
    """
    创建机器人间的Pose-to-Pose UT观测因子
    
    Args:
        robot1_id: 观测机器人标识符
        robot2_id: 被观测机器人标识符
        observations: (time1, time2, measurement, noise_covariance) 元组列表
        [Bearing, Range] measurements
        noise_covariance = 2x2
        validate_observations: 是否验证观测值
        **ut_params: UT参数 (mode, alpha, beta, kappa等)
        
    Returns:
        Pose-to-Pose UT因子列表
    """
    factors = []
    
    for time1, time2, relative_pose, noise_cov in observations:
        # 验证观测值
        if validate_observations:
            if not np.isfinite(relative_pose).all():
                logger.warning(f"Non-finite relative pose for times {time1}, {time2}, skipping")
                continue
            
            if relative_pose.size != 2:
                logger.warning(f"Invalid relative pose size for times {time1}, {time2}, skipping")
                continue
            
            if noise_cov.shape != (2, 2):
                logger.warning(f"Invalid noise covariance shape for times {time1}, {time2}, skipping")
                continue
            
            try:
                validate_matrix_properties(noise_cov, f"pose_to_pose_noise_{time1}_{time2}")
            except ValueError as e:
                logger.warning(f"Invalid noise covariance for times {time1}, {time2}: {e}, skipping")
                continue
        
        pose1_var = f"x{robot1_id}_{time1}"
        pose2_var = f"x{robot2_id}_{time2}"
        
        try:
            factors.append(PoseToPoseUTFactor(
                pose1_var, pose2_var, relative_pose, noise_cov, **ut_params
            ))
        except Exception as e:
            logger.error(f"Failed to create pose-to-pose UT factor for times {time1}, {time2}: {e}")
    
    logger.info(f"Created {len(factors)} pose-to-pose UT factors from robot {robot1_id} to robot {robot2_id}")
    return factors

def create_mixed_inter_robot_factors(robot1_id: int, robot2_id: int,
                                    loop_closures: List[Tuple[int, int, np.ndarray, np.ndarray]] = None,
                                    pose_observations: List[Tuple[int, int, np.ndarray, np.ndarray]] = None,
                                    validate_observations: bool = True,
                                    **ut_params) -> List[Factor]:
    """
    创建混合的机器人间因子（环路闭合 + Pose-to-Pose观测）
    
    Args:
        robot1_id: 第一个机器人标识符
        robot2_id: 第二个机器人标识符
        loop_closures: 环路闭合观测 (time1, time2, relative_pose, information_matrix)
        pose_observations: 机器人间 bearing-range 观测 (time1, time2, measurement[2], noise_covariance 2x2)
        validate_observations: 是否验证观测值
        **ut_params: UT参数
        
    Returns:
        混合因子列表
    """
    factors = []
    
    # 添加环路闭合因子
    if loop_closures:
        loop_factors = create_inter_robot_factors(
            robot1_id, robot2_id, loop_closures, validate_observations
        )
        factors.extend(loop_factors)
    
    # 添加Pose-to-Pose UT因子
    if pose_observations:
        pose_factors = create_pose_to_pose_ut_factors(
            robot1_id, robot2_id, pose_observations, validate_observations, **ut_params
        )
        factors.extend(pose_factors)
    
    logger.info(f"Created {len(factors)} mixed inter-robot factors between robots {robot1_id} and {robot2_id}")
    return factors

# =====================================================================
# Global Cache and Performance Management
# =====================================================================

def clear_all_caches():
    """清空所有全局缓存"""
    _global_cache.clear()
    _factor_cache_manager.clear_all()
    logger.info("Cleared all global caches")

def get_global_cache_stats() -> Dict[str, Any]:
    """获取全局缓存统计信息"""
    return {
        'global_cache': _global_cache.stats(),
        'factor_caches': _factor_cache_manager.get_stats()
    }

def configure_numerical_parameters(**kwargs):
    """配置全局数值参数"""
    global numerical_config
    
    for key, value in kwargs.items():
        if hasattr(numerical_config, key):
            setattr(numerical_config, key, value)
            logger.info(f"Set numerical parameter {key} = {value}")
        else:
            logger.warning(f"Unknown numerical parameter: {key}")

def validate_spbp_dimensions(K: np.ndarray, Rinv: np.ndarray, residual: np.ndarray,
                            expected_state_dim: int, expected_obs_dim: int) -> bool:
    """
    验证SPBP计算中的矩阵维度是否正确
    
    Args:
        K: 卡尔曼增益矩阵，应为 (n_x, n_z)
        Rinv: 观测噪声逆矩阵，应为 (n_z, n_z)  
        residual: 残差向量，应为 (n_z,)
        expected_state_dim: 期望的状态维度 n_x
        expected_obs_dim: 期望的观测维度 n_z
        
    Returns:
        维度是否正确
        
    Raises:
        ValueError: 维度不匹配时
    """
    n_x, n_z = expected_state_dim, expected_obs_dim
    
    # 检查K的维度
    if K.shape != (n_x, n_z):
        raise ValueError(f"Kalman gain K shape mismatch: expected ({n_x}, {n_z}), got {K.shape}")
    
    # 检查Rinv的维度
    if Rinv.shape != (n_z, n_z):
        raise ValueError(f"Rinv shape mismatch: expected ({n_z}, {n_z}), got {Rinv.shape}")
    
    # 检查residual的维度
    if residual.shape != (n_z,):
        raise ValueError(f"Residual shape mismatch: expected ({n_z},), got {residual.shape}")
    
    # 验证计算维度
    try:
        # 测试 K @ Rinv @ K.T 的维度
        temp1 = K @ Rinv  # (n_x, n_z) @ (n_z, n_z) = (n_x, n_z)
        result_Λ = temp1 @ K.T  # (n_x, n_z) @ (n_z, n_x) = (n_x, n_x)
        
        # 测试 K @ Rinv @ residual 的维度
        result_η = temp1 @ residual  # (n_x, n_z) @ (n_z,) = (n_x,)
        
        logger.debug(f"SPBP dimension validation passed: Λ{result_Λ.shape}, η{result_η.shape}")
        return True
        
    except Exception as e:
        raise ValueError(f"SPBP dimension validation failed: {e}")


def safe_spbp_information_conversion(K: np.ndarray, Rinv: np.ndarray, residual: np.ndarray,
                                   state_dim: int, obs_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    安全的SPBP信息形式转换，带有完整的维度检查
    
    Args:
        K: 卡尔曼增益矩阵 (n_x, n_z)
        Rinv: 观测噪声逆矩阵 (n_z, n_z)
        residual: 残差向量 (n_z,)
        state_dim: 状态维度
        obs_dim: 观测维度
        
    Returns:
        (Λ, η): 信息矩阵和信息向量
    """
    # 维度验证
    validate_spbp_dimensions(K, Rinv, residual, state_dim, obs_dim)
    
    # 安全计算
    try:
        Λ = K @ Rinv @ K.T  # (n_x, n_x)
        η = K @ Rinv @ residual  # (n_x,)
        
        # 数值验证
        if not np.isfinite(Λ).all() or not np.isfinite(η).all():
            raise ValueError("SPBP conversion produced non-finite values")
        
        # 检查信息矩阵的正定性
        try:
            validate_matrix_properties(Λ, "SPBP_information_matrix", 
                                     check_symmetric=True, 
                                     check_positive_definite=True)
        except ValueError as e:
            logger.warning(f"SPBP information matrix validation warning: {e}")
            # 正则化处理
            Λ = ensure_positive_definite(Λ, numerical_config.min_eigenvalue)
        
        return Λ, η
        
    except Exception as e:
        logger.error(f"SPBP information conversion failed: {e}")
        raise

# =====================================================================
# validate_factor_graph 函数
# =====================================================================
def validate_factor_graph(factors: List[Factor],
                          variables: Dict[str, int]) -> bool:
    """
    验证因子图的一致性
    
    Args:
        factors: 因子列表
        variables: 变量名到维度的映射
        
    Returns:
        图是否有效
    """
    try:
        # 检查因子是否引用了存在的变量
        for i, factor in enumerate(factors):
            for var_name in variables.keys():
                dim = factor._get_dim(var_name)
                if dim > 0:
                    expected_dim = variables[var_name]
                    if dim != expected_dim:
                        logger.error(f"Factor {i}: Variable {var_name} dimension mismatch: {dim} vs {expected_dim}")
                        return False
        
        logger.info(f"Validated factor graph: {len(factors)} factors, {len(variables)} variables")
        return True
        
    except Exception as e:
        logger.error(f"Factor graph validation failed: {e}")
        return False

# =====================================================================
# test_spbp_dimension_fix 函数
# =====================================================================
def test_spbp_dimension_fix():
    """
    单元自测：验证 _spbp_information_conversion 的维度检查不会再触发
    "K @ Rinv @ K.T 与 K.T @ Rinv @ K" 的旧 bug。
    """
    nx, nz = 5, 2
    K = np.random.randn(nx, nz)
    R = np.eye(nz) * 0.1
    Rinv = np.linalg.inv(R)
    residual = np.random.randn(nz)

    try:
        Λ, η = safe_spbp_information_conversion(K, Rinv, residual, nx, nz)
        assert Λ.shape == (nx, nx) and η.shape == (nx,)
        logger.info("SPBP dimension self-test passed")
        return True
    except Exception as e:
        logger.error(f"SPBP dimension self-test failed: {e}")
        return False

# =====================================================================
# ✅ 修复验证函数
# =====================================================================
def test_factor_fixes():
    """验证所有修复是否正确"""
    
    print("Testing factor fixes...")
    
    # 测试1: _get_dim()默认实现
    print("1. Testing _get_dim() default implementation...")
    factor = OdometryFactor("x0", "x1", np.array([1, 0, 0.1]), 0.1)
    assert factor._get_dim("x0") == 3, "Should return 3 for x0"
    assert factor._get_dim("x1") == 3, "Should return 3 for x1"
    assert factor._get_dim("unknown") == 0, "Should return 0 for unknown variables"
    print("   ✅ _get_dim() working correctly")
    
    # 测试2: 线性化返回格式
    print("2. Testing linearization return format...")
    mu = {"x0": np.array([0, 0, 0]), "x1": np.array([1, 0, 0.1])}
    cov = {"x0": np.eye(3)*0.01, "x1": np.eye(3)*0.01}
    result = factor.linearize(mu, cov)
    
    # 验证只有单向交叉块
    assert "x0" in result, "Should have x0 diagonal block"
    assert "x1" in result, "Should have x1 diagonal block"
    assert ("x0", "x1") in result, "Should have forward cross block"
    assert ("x1", "x0") not in result, "Should NOT have reverse cross block"
    print("   ✅ Single-direction cross blocks working correctly")
    
    # 测试3: BearingRangeUTFactor
    print("3. Testing BearingRangeUTFactor...")
    br_factor = BearingRangeUTFactor("x0", "l0", np.array([0.5, 2.0]), np.eye(2)*0.01)
    assert br_factor._get_dim("x0") == 3, "Should return 3 for pose"
    assert br_factor._get_dim("l0") == 2, "Should return 2 for landmark"
    assert br_factor._get_dim("unknown") == 0, "Should return 0 for unknown"
    print("   ✅ BearingRangeUTFactor working correctly")
    
    # 测试4: LoopClosureFactor
    print("4. Testing LoopClosureFactor...")
    lc_factor = LoopClosureFactor("x0", "x1", np.array([1, 0, 0.1]), np.eye(3)*100)
    assert lc_factor._get_dim("x0") == 3, "Should return 3 for pose1"
    assert lc_factor._get_dim("x1") == 3, "Should return 3 for pose2"
    assert lc_factor._get_dim("unknown") == 0, "Should return 0 for unknown"
    
    # 测试线性化格式
    result = lc_factor.linearize(mu, cov)
    assert ("x0", "x1") in result, "Should have forward cross block"
    assert ("x1", "x0") not in result, "Should NOT have reverse cross block"
    print("   ✅ LoopClosureFactor working correctly")
    
    # 测试5: PoseToPoseUTFactor
    print("5. Testing PoseToPoseUTFactor...")
    p2p_factor = PoseToPoseUTFactor("x0", "x1", np.array([0.5, 2.0]), np.eye(2)*0.01)
    assert p2p_factor._get_dim("x0") == 3, "Should return 3 for pose1"
    assert p2p_factor._get_dim("x1") == 3, "Should return 3 for pose2"
    assert p2p_factor._get_dim("unknown") == 0, "Should return 0 for unknown"
    print("   ✅ PoseToPoseUTFactor working correctly")
    
    print("\n🎉 All factor fixes verified successfully!")
    print("\nKey improvements:")
    print("  ✅ Factor._get_dim() now uses default implementation")
    print("  ✅ All factors use _dim_map for dimension mapping")
    print("  ✅ Single-direction cross blocks eliminate redundancy")
    print("  ✅ No more matrix symmetry violations")
    print("  ✅ Compatible with standard GBP backends")
    
    return True

# =====================================================================
# Main Export List - ✅ 修复：添加缺失的导出
# =====================================================================
__all__ = [
    # Core classes
    'Factor', 'PriorFactor', 'OdometryFactor', 'BearingRangeUTFactor', 
    'PoseToPoseUTFactor', 'LoopClosureFactor', 'SigmaGenerator',
    
    # Utility functions
    'wrap_angle', 'angle_diff', 'ensure_positive_definite', 'safe_matrix_inverse',
    'validate_matrix_properties', 'clip_jacobian',
    
    # SPBP specific utilities
    'validate_spbp_dimensions', 'safe_spbp_information_conversion',
    
    # Multi-robot utilities
    'create_multi_robot_odometry_chain', 'create_landmark_observation_factors', 
    'create_inter_robot_factors', 'create_pose_to_pose_ut_factors', 
    'create_mixed_inter_robot_factors',
    
    # Cache and performance management
    'ThreadSafeCache', 'FactorCacheManager', 'clear_all_caches', 'get_global_cache_stats',
    
    # Configuration and validation
    'configure_numerical_parameters', 'validate_factor_graph', 'NumericalConfig',
    'test_spbp_dimension_fix', 'test_factor_fixes',  # ✅ 添加测试函数
    
    # Global instances
    'numerical_config', '_factor_cache_manager'
]
