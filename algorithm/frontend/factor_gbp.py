"""
GBP-only Factor implementations for Multi-Robot SLAM
Removed UT (SPBP) and related utilities, focusing solely on classic Gaussian Belief Propagation
"""
from __future__ import annotations
import numpy as np
import math
import threading
import logging
from typing import Dict, Tuple, Any, List
from abc import ABC, abstractmethod

# Configure logging
logger = logging.getLogger(__name__)

# =====================================================================
# Numerical Configuration
# =====================================================================
class NumericalConfig:
    """
    Thresholds and regularization parameters for numerical stability
    """
    max_condition_number: float = 1.0e6
    min_eigenvalue: float = 1.0e-6
    jacobian_clip_threshold: float = 1.0e6

numerical_config = NumericalConfig()

# =====================================================================
# Utility Functions
# =====================================================================

def wrap_angle(theta: float) -> float:
    """Wrap angle into [-π, π]"""
    return math.atan2(math.sin(theta), math.cos(theta))


def validate_matrix_properties(matrix: np.ndarray, name: str = "matrix") -> None:
    """Ensure matrix is square and positive definite"""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square, got {matrix.shape}")
    # check eigenvalues
    eig = np.linalg.eigvalsh(matrix)
    if np.any(eig <= 0):
        raise ValueError(f"{name} not positive definite (min eigenvalue {eig.min():.2e})")


def ensure_positive_definite(matrix: np.ndarray) -> np.ndarray:
    """Add regularization to enforce positive definiteness"""
    matrix = 0.5 * (matrix + matrix.T)
    try:
        np.linalg.cholesky(matrix)
        return matrix
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals_reg = np.clip(eigvals, numerical_config.min_eigenvalue, None)
        reg_matrix = eigvecs @ np.diag(eigvals_reg) @ eigvecs.T
        return 0.5 * (reg_matrix + reg_matrix.T)


def clip_jacobian(J: np.ndarray) -> np.ndarray:
    """Clip Jacobian entries to avoid numeric explosion"""
    thresh = numerical_config.jacobian_clip_threshold
    if np.max(np.abs(J)) <= thresh:
        return J
    return np.clip(J, -thresh, thresh)

# =====================================================================
# Base Factor Class
# =====================================================================
class Factor(ABC):
    """Base class for GBP factors"""
    def __init__(self):
        self._dim_map: Dict[str, int] = {}

    @abstractmethod
    def linearize(self, mu: Dict[str, np.ndarray]) -> Dict[Any, Any]:
        """Return info blocks: {var: (Λ, η), (var_i,var_j): Λ_ij}"""
        pass

    @abstractmethod
    def get_energy(self, mu: Dict[str, np.ndarray]) -> float:
        """Return scalar energy"""
        pass

    def _get_dim(self, key: str) -> int:
        """Variable dimension or 0 if unused"""
        return self._dim_map.get(key, 0)

# =====================================================================
# Prior Factor
# =====================================================================
class PriorFactor(Factor):
    def __init__(self, var: str, prior: np.ndarray, sigma: float):
        super().__init__()
        self.var = var
        self.prior = prior.copy()
        self._dim_map = {var: prior.size}
        var = prior
        # build info
        info = 1.0/(sigma**2)
        self._Λ = np.eye(prior.size)*info
        self._η = self._Λ @ self.prior
        validate_matrix_properties(self._Λ, "prior_information")

    def linearize(self, mu: Dict[str, np.ndarray]) -> Dict[Any, Any]:
        return {self.var: (self._Λ.copy(), self._η.copy())}

    def get_energy(self, mu: Dict[str, np.ndarray]) -> float:
        x = mu.get(self.var)
        if x is None:
            return 0.0
        res = x - self.prior
        if res.size==3:
            res[2]=wrap_angle(res[2])
        return 0.5 * res @ self._Λ @ res

# =====================================================================
# Odometry Factor
# =====================================================================
class OdometryFactor(Factor):
    def __init__(self, v_from: str, v_to: str, delta: np.ndarray, sigma: float):
        super().__init__()
        self.v1, self.v2 = v_from, v_to
        self.delta = delta.copy()
        self._dim_map = {v_from: 3, v_to: 3}
        info = 1.0/(sigma**2)
        self.Rinv = np.eye(3)*info

    @staticmethod
    def _se2_relative_pose(a: np.ndarray, b: np.ndarray)->np.ndarray:
        dx = b[0]-a[0]
        dy = b[1]-a[1]
        dtheta = wrap_angle(b[2]-a[2])
        c,s = math.cos(a[2]), math.sin(a[2])
        x =  c*dx + s*dy
        y = -s*dx + c*dy
        return np.array([x,y,dtheta])

    @staticmethod
    def _adj_inv(delta: np.ndarray)->np.ndarray:
        dx,dy,th = delta
        if abs(th)<numerical_config.min_eigenvalue:
            return np.array([[1,th,-dy],[-th,1,dx],[0,0,1]])
        c,s = math.cos(th), math.sin(th)
        return np.array([[c, s,-dy],[-s,c,dx],[0,0,1]])

    def linearize(self, mu: Dict[str, np.ndarray]) -> Dict[Any, Any]:
        a = mu.get(self.v1); b = mu.get(self.v2)
        if a is None or b is None:
            return {}
        delta_act = self._se2_relative_pose(a,b)
        res = self.delta - delta_act
        res[2]=wrap_angle(res[2])
        J1 = clip_jacobian(self._adj_inv(delta_act))
        J2 = clip_jacobian(-np.eye(3))
        Λ11 = J1.T @ self.Rinv @ J1
        Λ22 = J2.T @ self.Rinv @ J2
        Λ12 = J1.T @ self.Rinv @ J2
        η1 = J1.T @ self.Rinv @ res
        η2 = J2.T @ self.Rinv @ res
        return {self.v1:(Λ11,η1), self.v2:(Λ22,η2), (self.v1,self.v2):Λ12}

    def get_energy(self, mu: Dict[str, np.ndarray]) -> float:
        a = mu.get(self.v1); b = mu.get(self.v2)
        if a is None or b is None:
            return 0.0
        res = self.delta - self._se2_relative_pose(a,b)
        res[2]=wrap_angle(res[2])
        return 0.5 * res @ self.Rinv @ res

# =====================================================================
# Bearing-Range Factor
# =====================================================================
class BearingRangeFactor(Factor):
    def __init__(self, pose: str, lm: str, z: np.ndarray, R: np.ndarray):
        super().__init__()
        self.pose_key,self.lm_key=pose,lm
        self.z=z.copy()
        validate_matrix_properties(R, "measurement_cov")
        self.Rinv=np.linalg.inv(R)
        self._dim_map={pose:3, lm:2}

    def linearize(self, mu: Dict[str, np.ndarray]) -> Dict[Any, Any]:
        p=mu.get(self.pose_key); l=mu.get(self.lm_key)
        if p is None or l is None:
            return {}
        dx,dy = l[0]-p[0], l[1]-p[1]
        dist = math.hypot(dx,dy)
        bearing = wrap_angle(math.atan2(dy,dx)-p[2])
        res = np.array([wrap_angle(self.z[0]-bearing), self.z[1]-dist])
        safe= max(dist,numerical_config.min_eigenvalue**0.5)
        q=safe**2
        Jp = np.array([[ dy/q, -dx/q, -1],[ -dx/safe, -dy/safe,0]])
        Jl = np.array([[-dy/q, dx/q],[ dx/safe, dy/safe]])
        Jp, Jl = clip_jacobian(Jp), clip_jacobian(Jl)
        Λp = ensure_positive_definite(Jp.T@self.Rinv@Jp)
        Λl = ensure_positive_definite(Jl.T@self.Rinv@Jl)
        Λpl = Jp.T@self.Rinv@Jl
        ηp = Jp.T@self.Rinv@res
        ηl = Jl.T@self.Rinv@res
        return {self.pose_key:(Λp,ηp), self.lm_key:(Λl,ηl), (self.pose_key,self.lm_key):Λpl}

    def get_energy(self, mu: Dict[str, np.ndarray]) -> float:
        p=mu.get(self.pose_key); l=mu.get(self.lm_key)
        if p is None or l is None:
            return 0.0
        dx,dy = l[0]-p[0], l[1]-p[1]
        pred = np.array([wrap_angle(math.atan2(dy,dx)-p[2]), math.hypot(dx,dy)])
        res = self.z - pred
        res[0]=wrap_angle(res[0])
        return 0.5 * res @ self.Rinv @ res

# =====================================================================
# Loop Closure Factor
# =====================================================================
class LoopClosureFactor(Factor):
    def __init__(self, p1: str, p2: str, rel: np.ndarray, info: np.ndarray):
        super().__init__()
        self.p1,self.p2=p1,p2
        self.rel=rel.copy()
        validate_matrix_properties(info, "loop_info")
        self.info=info.copy()
        self._dim_map={p1:3,p2:3}

    def linearize(self, mu: Dict[str, np.ndarray]) -> Dict[Any, Any]:
        a=mu.get(self.p1); b=mu.get(self.p2)
        if a is None or b is None:
            return {}
        delta=OdometryFactor._se2_relative_pose(a,b)
        res=self.rel-delta
        res[2]=wrap_angle(res[2])
        J1=OdometryFactor._adj_inv(delta)
        J2=-np.eye(3)
        J1,J2=clip_jacobian(J1),clip_jacobian(J2)
        Λ11=J1.T@self.info@J1
        Λ22=J2.T@self.info@J2
        Λ12=J1.T@self.info@J2
        η1=J1.T@self.info@res
        η2=J2.T@self.info@res
        return {self.p1:(Λ11,η1), self.p2:(Λ22,η2), (self.p1,self.p2):Λ12}

    def get_energy(self, mu: Dict[str, np.ndarray]) -> float:
        a=mu.get(self.p1); b=mu.get(self.p2)
        if a is None or b is None:
            return 0.0
        delta=OdometryFactor._se2_relative_pose(a,b)
        res=self.rel-delta
        res[2]=wrap_angle(res[2])
        return 0.5 * res @ self.info @ res

# =====================================================================
# Graph Validation
# =====================================================================
def validate_factor_graph(factors: List[Factor], variables: Dict[str,int]) -> bool:
    for f in factors:
        for v,dim in variables.items():
            d=f._get_dim(v)
            if d not in (0,dim):
                logger.error(f"Var {v} dim mismatch: {d} vs {dim}")
                return False
    return True
