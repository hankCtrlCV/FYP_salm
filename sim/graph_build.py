"""
Utility: build factor graph for a given robot index.
Returns list[Factor] + initial guess dict{key:μ}.
"""
from typing import List, Dict, Tuple
import numpy as np
from algorithm.frontend.factor_ut import BearingRangeUTFactor

def build_graph(robot_id: int,
                paths_init: np.ndarray,
                lm_init: np.ndarray,
                meas: list,
                R_noise: np.ndarray):
    factors: List = []
    initial: Dict[str, np.ndarray] = {}

    # insert variables
    for t, pose0 in enumerate(paths_init[robot_id]):
        key = f"x{robot_id}_{t}"
        initial[key] = pose0
    for j, lm0 in enumerate(lm_init):
        initial[f"l{j}"] = lm0

    # construct factors
    for t, obs in enumerate(meas[robot_id]):
        pose_key = f"x{robot_id}_{t}"
        for j, z in obs:
            lm_key = f"l{j}"
            factors.append(BearingRangeUTFactor(pose_key, lm_key, z, R_noise))

    return factors, initial
