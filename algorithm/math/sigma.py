"""
Unscented transform helpers.
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class UTParam:
    alpha: float = 1e-1
    beta:  float = 2.0
    kappa: float = 0.0

def sigma_points(mu: np.ndarray, cov: np.ndarray,
                 p: UTParam = UTParam()) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # TODO: 生成 2n+1 sigma & 权重
    ...
