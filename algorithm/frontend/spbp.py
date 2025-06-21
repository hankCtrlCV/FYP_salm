"""
Enhanced Sigma-Point Belief Propagation (SPBP) Implementation
- Numerically stable sigma point generation
- Robust unscented transform with proper cross-covariance computation
- Full compatibility with factor_ut.py and gbp.py
- Memory-efficient caching with proper cleanup

Author: Enhanced for Multi-Robot SLAM
Date: 2025-06-18
"""

from __future__ import annotations
import numpy as np
import math
import threading
from typing import Callable, Tuple, Dict, Any, Optional, List
import logging

logger = logging.getLogger("SPBP")

# Constants
_EPSILON = 1e-12
_MAX_CACHE_SIZE = 1000
_DEFAULT_REGULARIZATION = 1e-10

def ensure_positive_definite(matrix: np.ndarray, min_eigenvalue: float = _DEFAULT_REGULARIZATION) -> np.ndarray:
    """
    Ensure matrix is positive definite through eigenvalue regularization
    
    Args:
        matrix: Input matrix
        min_eigenvalue: Minimum eigenvalue threshold
        
    Returns:
        Positive definite matrix
    """
    # Symmetrize
    matrix = 0.5 * (matrix + matrix.T)
    
    # Eigenvalue decomposition
    eigenvals, eigenvecs = np.linalg.eigh(matrix)
    
    # Regularize eigenvalues
    eigenvals = np.maximum(eigenvals, min_eigenvalue)
    
    return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

class SigmaPointGenerator:
    """
    Enhanced sigma point generator for Unscented Transform
    
    Features:
    - Numerically stable sigma point generation
    - Adaptive regularization for ill-conditioned covariances
    - Thread-safe operation
    - Memory-efficient caching
    """
    
    def __init__(self, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0,
                 enable_caching: bool = True, max_cache_size: int = _MAX_CACHE_SIZE):
        """
        Initialize sigma point generator
        
        Args:
            alpha: Spread parameter (1e-4 ≤ α ≤ 1)
            beta: Higher order moments parameter (β = 2 optimal for Gaussian)
            kappa: Secondary scaling parameter
            enable_caching: Enable result caching for performance
            max_cache_size: Maximum number of cached results
        """
        # Validate parameters
        if not (1e-4 <= alpha <= 1.0):
            raise ValueError(f"Alpha must be in [1e-4, 1.0], got {alpha}")
        if kappa < 0:
            raise ValueError(f"Kappa must be non-negative, got {kappa}")
        
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.enable_caching = enable_caching
        self.max_cache_size = max_cache_size
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Caches
        self._sigma_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._lambda_cache: Dict[int, float] = {}
        self._sqrt_cache: Dict[str, np.ndarray] = {}
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0

    def generate(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate sigma points for unscented transform
        
        Args:
            mean: Mean vector (n,)
            covariance: Covariance matrix (n, n)
            
        Returns:
            sigma_points: Sigma points (2n+1, n)
            weights_mean: Mean weights (2n+1,)
            weights_cov: Covariance weights (2n+1,)
        """
        mean = np.asarray(mean, dtype=np.float64)
        covariance = np.asarray(covariance, dtype=np.float64)
        
        # Input validation
        if mean.ndim != 1:
            raise ValueError(f"Mean must be 1D, got shape {mean.shape}")
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError(f"Covariance must be square matrix, got shape {covariance.shape}")
        if mean.size != covariance.shape[0]:
            raise ValueError(f"Dimension mismatch: mean {mean.size}, cov {covariance.shape[0]}")
        
        n = mean.size
        
        with self._lock:
            # Check cache
            if self.enable_caching:
                cache_key = self._compute_cache_key(mean, covariance)
                if cache_key in self._sigma_cache:
                    self.cache_hits += 1
                    return self._sigma_cache[cache_key]
                else:
                    self.cache_misses += 1
            
            # Compute scaling parameters
            lambda_val = self._compute_lambda(n)
            gamma = math.sqrt(n + lambda_val)
            
            # Ensure positive definite covariance
            cov_pd = ensure_positive_definite(covariance, _DEFAULT_REGULARIZATION)
            
            # Compute matrix square root
            sqrt_matrix = self._compute_matrix_sqrt(cov_pd, n)
            
            # Generate sigma points
            sigma_points = np.zeros((2*n + 1, n))
            sigma_points[0] = mean
            
            for i in range(n):
                offset = gamma * sqrt_matrix[:, i]
                sigma_points[i + 1] = mean + offset
                sigma_points[n + i + 1] = mean - offset
            
            # Compute weights
            weights_mean, weights_cov = self._compute_weights(n, lambda_val)
            
            # Cache result
            result = (sigma_points, weights_mean, weights_cov)
            if self.enable_caching:
                self._update_cache(cache_key, result)
            
            return result

    def _compute_lambda(self, n: int) -> float:
        """Compute lambda scaling parameter with caching"""
        if n in self._lambda_cache:
            return self._lambda_cache[n]
        
        lambda_val = self.alpha**2 * (n + self.kappa) - n
        
        # Ensure numerical stability
        if lambda_val + n <= 0:
            logger.warning(f"Adjusting lambda for stability: n={n}, original lambda={lambda_val}")
            lambda_val = 3 - n
        
        self._lambda_cache[n] = lambda_val
        return lambda_val

    def _compute_matrix_sqrt(self, matrix: np.ndarray, n: int) -> np.ndarray:
        """Compute matrix square root with multiple fallback methods"""
        # Create cache key for matrix square root
        if self.enable_caching:
            sqrt_key = f"sqrt_{hash(matrix.tobytes())}_{n}"
            if sqrt_key in self._sqrt_cache:
                return self._sqrt_cache[sqrt_key]
        
        # Method 1: Try Cholesky decomposition
        try:
            sqrt_matrix = np.linalg.cholesky(matrix)
            if self.enable_caching:
                self._sqrt_cache[sqrt_key] = sqrt_matrix
            return sqrt_matrix
        except np.linalg.LinAlgError:
            logger.debug("Cholesky decomposition failed, trying eigenvalue method")
        
        # Method 2: Eigenvalue decomposition
        try:
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals = np.maximum(eigenvals, _DEFAULT_REGULARIZATION)
            sqrt_matrix = eigenvecs @ np.diag(np.sqrt(eigenvals))
            if self.enable_caching:
                self._sqrt_cache[sqrt_key] = sqrt_matrix
            return sqrt_matrix
        except Exception as e:
            logger.error(f"Matrix square root computation failed: {e}")
            # Fallback to identity scaled by matrix trace
            return np.eye(n) * math.sqrt(np.trace(matrix) / n)

    def _compute_weights(self, n: int, lambda_val: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and covariance weights"""
        # Mean weights
        weights_mean = np.full(2*n + 1, 0.5 / (n + lambda_val))
        weights_mean[0] = lambda_val / (n + lambda_val)
        
        # Covariance weights
        weights_cov = weights_mean.copy()
        weights_cov[0] = weights_mean[0] + (1 - self.alpha**2 + self.beta)
        
        # Validate weights
        if not np.isclose(np.sum(weights_mean), 1.0, rtol=1e-10):
            logger.warning(f"Mean weights don't sum to 1: {np.sum(weights_mean)}")
        
        return weights_mean, weights_cov

    def _compute_cache_key(self, mean: np.ndarray, covariance: np.ndarray) -> str:
        """Compute stable cache key"""
        # Round to reduce floating point precision issues
        mean_rounded = np.round(mean, decimals=12)
        cov_rounded = np.round(covariance, decimals=12)
        
        # Create hash from parameters and data
        params_str = f"{self.alpha:.10f}_{self.beta:.10f}_{self.kappa:.10f}"
        mean_str = np.array_str(mean_rounded, precision=12)
        cov_str = np.array_str(cov_rounded, precision=12)
        
        return f"{params_str}_{hash(mean_str)}_{hash(cov_str)}"

    def _update_cache(self, key: str, result: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """Update cache with size management"""
        if len(self._sigma_cache) >= self.max_cache_size:
            # Remove oldest entries (FIFO)
            oldest_keys = list(self._sigma_cache.keys())[:len(self._sigma_cache)//2]
            for old_key in oldest_keys:
                del self._sigma_cache[old_key]
        
        # Also manage sqrt_cache size
        if len(self._sqrt_cache) >= self.max_cache_size:
            oldest_sqrt_keys = list(self._sqrt_cache.keys())[:len(self._sqrt_cache)//2]
            for old_key in oldest_sqrt_keys:
                del self._sqrt_cache[old_key]
        
        self._sigma_cache[key] = result

    def clear_cache(self):
        """Clear all caches"""
        with self._lock:
            self._sigma_cache.clear()
            self._lambda_cache.clear()
            self._sqrt_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._sigma_cache),
            "max_cache_size": self.max_cache_size
        }

class UnscentedTransform:
    """
    Enhanced Unscented Transform implementation
    """
    
    def __init__(self, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0,
                 enable_caching: bool = True):
        """
        Initialize Unscented Transform
        
        Args:
            alpha, beta, kappa: UT parameters
            enable_caching: Enable caching for performance
        """
        self.generator = SigmaPointGenerator(alpha, beta, kappa, enable_caching)

    def transform(self, 
                 nonlinear_function: Callable[[np.ndarray], np.ndarray],
                 mean_input: np.ndarray,
                 covariance_input: np.ndarray,
                 noise_covariance: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply unscented transform to propagate uncertainty through nonlinear function
        
        Args:
            nonlinear_function: Function f: R^n -> R^m
            mean_input: Input mean (n,)
            covariance_input: Input covariance (n, n)
            noise_covariance: Additive noise covariance (m, m), optional
            
        Returns:
            mean_output: Output mean (m,)
            covariance_output: Output covariance (m, m)
            cross_covariance: Cross-covariance between input and output (n, m)
        """
        # Generate sigma points
        sigma_points, weights_mean, weights_cov = self.generator.generate(mean_input, covariance_input)
        
        # Propagate sigma points through nonlinear function
        try:
            # Evaluate function at each sigma point
            output_points = []
            for point in sigma_points:
                output = nonlinear_function(point)
                output_points.append(np.asarray(output, dtype=np.float64))
            
            output_points = np.stack(output_points)  # (2n+1, m)
            
        except Exception as e:
            logger.error(f"Function evaluation failed: {e}")
            raise RuntimeError(f"Nonlinear function evaluation failed: {e}") from e
        
        # Compute output statistics
        mean_output = self._compute_weighted_mean(output_points, weights_mean)
        covariance_output = self._compute_output_covariance(
            output_points, mean_output, weights_cov, noise_covariance
        )
        cross_covariance = self._compute_cross_covariance(
            sigma_points, mean_input, output_points, mean_output, weights_cov
        )
        
        return mean_output, covariance_output, cross_covariance

    def _compute_weighted_mean(self, points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Compute weighted mean of points"""
        return np.average(points, axis=0, weights=weights)

    def _compute_output_covariance(self, output_points: np.ndarray, mean_output: np.ndarray,
                                  weights_cov: np.ndarray, noise_covariance: Optional[np.ndarray]) -> np.ndarray:
        """Compute output covariance matrix"""
        # Compute deviations
        deviations = output_points - mean_output  # (2n+1, m)
        
        # Weighted covariance
        covariance = np.zeros((deviations.shape[1], deviations.shape[1]))
        for i, weight in enumerate(weights_cov):
            covariance += weight * np.outer(deviations[i], deviations[i])
        
        # Add noise covariance if provided
        if noise_covariance is not None:
            covariance += noise_covariance
        
        # Ensure positive definite
        covariance = ensure_positive_definite(covariance, _EPSILON)
        
        return covariance

    def _compute_cross_covariance(self, input_points: np.ndarray, mean_input: np.ndarray,
                                 output_points: np.ndarray, mean_output: np.ndarray,
                                 weights_cov: np.ndarray) -> np.ndarray:
        """Compute cross-covariance between input and output"""
        # Compute deviations
        input_deviations = input_points - mean_input    # (2n+1, n)
        output_deviations = output_points - mean_output  # (2n+1, m)
        
        # Weighted cross-covariance
        cross_cov = np.zeros((input_deviations.shape[1], output_deviations.shape[1]))
        for i, weight in enumerate(weights_cov):
            cross_cov += weight * np.outer(input_deviations[i], output_deviations[i])
        
        return cross_cov

    def transform_with_angles(self,
                            nonlinear_function: Callable[[np.ndarray], np.ndarray],
                            mean_input: np.ndarray,
                            covariance_input: np.ndarray,
                            angle_indices_input: Optional[List[int]] = None,
                            angle_indices_output: Optional[List[int]] = None,
                            noise_covariance: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Unscented transform with proper handling of angular variables
        
        Args:
            nonlinear_function: Function f: R^n -> R^m
            mean_input: Input mean (n,)
            covariance_input: Input covariance (n, n)
            angle_indices_input: Indices of angular variables in input
            angle_indices_output: Indices of angular variables in output
            noise_covariance: Additive noise covariance (m, m)
            
        Returns:
            mean_output: Output mean (m,)
            covariance_output: Output covariance (m, m)
            cross_covariance: Cross-covariance (n, m)
        """
        # Generate sigma points (angles are automatically handled in differences)
        sigma_points, weights_mean, weights_cov = self.generator.generate(mean_input, covariance_input)
        
        # Wrap input angles if specified
        if angle_indices_input:
            for point in sigma_points:
                for idx in angle_indices_input:
                    if 0 <= idx < len(point):
                        point[idx] = math.atan2(math.sin(point[idx]), math.cos(point[idx]))
        
        # Propagate through function
        output_points = []
        for point in sigma_points:
            output = nonlinear_function(point)
            output_points.append(np.asarray(output, dtype=np.float64))
        
        output_points = np.stack(output_points)
        
        # Compute output mean with circular statistics for angles
        mean_output = self._compute_circular_mean(output_points, weights_mean, angle_indices_output)
        
        # Compute covariances with proper angle handling
        covariance_output = self._compute_circular_covariance(
            output_points, mean_output, weights_cov, angle_indices_output, noise_covariance
        )
        
        cross_covariance = self._compute_circular_cross_covariance(
            sigma_points, mean_input, output_points, mean_output, weights_cov,
            angle_indices_input, angle_indices_output
        )
        
        return mean_output, covariance_output, cross_covariance

    def _compute_circular_mean(self, points: np.ndarray, weights: np.ndarray,
                              angle_indices: Optional[List[int]]) -> np.ndarray:
        """Compute mean with circular statistics for angular components"""
        mean = np.average(points, axis=0, weights=weights)
        
        if angle_indices:
            for idx in angle_indices:
                if 0 <= idx < len(mean):
                    # Use circular mean for angular components
                    sin_sum = np.sum(weights * np.sin(points[:, idx]))
                    cos_sum = np.sum(weights * np.cos(points[:, idx]))
                    mean[idx] = math.atan2(sin_sum, cos_sum)
        
        return mean

    def _compute_circular_covariance(self, points: np.ndarray, mean: np.ndarray,
                                   weights: np.ndarray, angle_indices: Optional[List[int]],
                                   noise_covariance: Optional[np.ndarray]) -> np.ndarray:
        """Compute covariance with proper angle difference handling"""
        deviations = points - mean
        
        # Wrap angle differences
        if angle_indices:
            for idx in angle_indices:
                if 0 <= idx < deviations.shape[1]:
                    deviations[:, idx] = np.arctan2(
                        np.sin(deviations[:, idx]), 
                        np.cos(deviations[:, idx])
                    )
        
        # Weighted covariance
        covariance = np.zeros((deviations.shape[1], deviations.shape[1]))
        for i, weight in enumerate(weights):
            covariance += weight * np.outer(deviations[i], deviations[i])
        
        if noise_covariance is not None:
            covariance += noise_covariance
        
        return ensure_positive_definite(covariance, _EPSILON)

    def _compute_circular_cross_covariance(self, input_points: np.ndarray, mean_input: np.ndarray,
                                         output_points: np.ndarray, mean_output: np.ndarray,
                                         weights: np.ndarray,
                                         angle_indices_input: Optional[List[int]],
                                         angle_indices_output: Optional[List[int]]) -> np.ndarray:
        """Compute cross-covariance with angle handling"""
        input_deviations = input_points - mean_input
        output_deviations = output_points - mean_output
        
        # Wrap angle differences
        if angle_indices_input:
            for idx in angle_indices_input:
                if 0 <= idx < input_deviations.shape[1]:
                    input_deviations[:, idx] = np.arctan2(
                        np.sin(input_deviations[:, idx]),
                        np.cos(input_deviations[:, idx])
                    )
        
        if angle_indices_output:
            for idx in angle_indices_output:
                if 0 <= idx < output_deviations.shape[1]:
                    output_deviations[:, idx] = np.arctan2(
                        np.sin(output_deviations[:, idx]),
                        np.cos(output_deviations[:, idx])
                    )
        
        # Weighted cross-covariance
        cross_cov = np.zeros((input_deviations.shape[1], output_deviations.shape[1]))
        for i, weight in enumerate(weights):
            cross_cov += weight * np.outer(input_deviations[i], output_deviations[i])
        
        return cross_cov

# =====================================================================
# Convenience Functions
# =====================================================================

def unscented_transform(nonlinear_function: Callable[[np.ndarray], np.ndarray],
                       mean_input: np.ndarray,
                       covariance_input: np.ndarray,
                       noise_covariance: Optional[np.ndarray] = None,
                       alpha: float = 1e-3,
                       beta: float = 2.0,
                       kappa: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function for unscented transform
    
    Args:
        nonlinear_function: Nonlinear function f: R^n -> R^m
        mean_input: Input mean (n,)
        covariance_input: Input covariance (n, n)
        noise_covariance: Additive noise covariance (m, m)
        alpha, beta, kappa: UT parameters
        
    Returns:
        mean_output: Output mean (m,)
        covariance_output: Output covariance (m, m)
        cross_covariance: Cross-covariance (n, m)
    """
    ut = UnscentedTransform(alpha, beta, kappa)
    return ut.transform(nonlinear_function, mean_input, covariance_input, noise_covariance)

def create_bearing_range_ut(pose_mean: np.ndarray, pose_cov: np.ndarray,
                           landmark_mean: np.ndarray, landmark_cov: np.ndarray,
                           cross_cov: Optional[np.ndarray] = None,
                           noise_cov: Optional[np.ndarray] = None,
                           **ut_params) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Specialized UT for bearing-range observations
    
    Args:
        pose_mean: Robot pose mean [x, y, theta]
        pose_cov: Robot pose covariance (3, 3)
        landmark_mean: Landmark position mean [x, y]
        landmark_cov: Landmark covariance (2, 2)
        cross_cov: Cross-covariance between pose and landmark (3, 2)
        noise_cov: Observation noise covariance (2, 2)
        **ut_params: UT parameters
        
    Returns:
        predicted_obs: Predicted observation [bearing, range]
        obs_cov: Observation covariance (2, 2)
        cross_cov_out: Cross-covariance (5, 2)
    """
    # Construct joint state
    joint_mean = np.hstack([pose_mean, landmark_mean])
    
    if cross_cov is None:
        cross_cov = np.zeros((3, 2))
    
    joint_cov = np.block([
        [pose_cov, cross_cov],
        [cross_cov.T, landmark_cov]
    ])
    
    # Observation function
    def obs_function(state):
        px, py, ptheta, lx, ly = state
        dx, dy = lx - px, ly - py
        range_val = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx) - ptheta
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))  # Wrap
        return np.array([bearing, range_val])
    
    # Apply UT
    ut = UnscentedTransform(**ut_params)
    return ut.transform_with_angles(
        obs_function, joint_mean, joint_cov,
        angle_indices_input=[2],  # pose theta
        angle_indices_output=[0], # bearing
        noise_covariance=noise_cov
    )

# Global instance for backward compatibility
_default_generator = SigmaPointGenerator()

def generate_sigma_points(mean: np.ndarray, covariance: np.ndarray,
                         alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
    """Backward compatibility function"""
    generator = SigmaPointGenerator(alpha, beta, kappa)
    return generator.generate(mean, covariance)

# Backward compatibility alias for factor_ut.py
class SigmaGenerator(SigmaPointGenerator):
    """
    Backward compatibility wrapper for factor_ut.py
    Maintains the same interface as the original SigmaGenerator
    """
    def __init__(self, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        super().__init__(alpha, beta, kappa)

# Export main classes and functions
__all__ = [
    'SigmaPointGenerator', 'SigmaGenerator', 'UnscentedTransform',
    'unscented_transform', 'create_bearing_range_ut', 'generate_sigma_points',
    'ensure_positive_definite'
]