"""GBP-Factor-Graph Builder (3-DoF pose: x, y, θ)"""

from __future__ import annotations
import math, logging, numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

from algorithm.frontend.factor_ut import (
    PriorFactor, OdometryFactor, BearingRangeUTFactor
)

logger = logging.getLogger("GBPBuilder")

def wrap_angle(theta: float) -> float:
    """Wrap to [-π, π]"""
    while theta > math.pi:
        theta -= 2 * math.pi
    while theta < -math.pi:
        theta += 2 * math.pi
    return theta

class GBPGraphBuilder:
    # Simplified configuration
    _DEF = dict(
        prior_sigma_xy=0.10, prior_sigma_theta=0.05,
        odom_sigma_xy=0.10, odom_sigma_theta=0.02,
        obs_sigma_range=0.30, obs_sigma_bearing=math.radians(2.0),
        ut_mode="auto", ut_alpha=0.30, ut_kappa=2.0,
        max_obs_range=25.0
    )

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = {**self._DEF, **(cfg or {})}
        self._check_cfg()
        
        self.factors = []
        self.variables = {}
        self._R_cache = None
        self._landmarks = None
        
        # Precompute odometry sigma for performance
        self._odom_sigma = np.array([
            self.cfg["odom_sigma_xy"],
            self.cfg["odom_sigma_xy"],
            self.cfg["odom_sigma_theta"]
        ])

    def build(self, robot_paths, landmark_pos, meas) -> Tuple[List, Dict]:
        self._check_input(robot_paths, landmark_pos, meas)
        self.factors.clear()
        self.variables.clear()
        self._R_cache = None
        self._landmarks = landmark_pos.copy()

        # Add landmarks
        for lid, pos in enumerate(self._landmarks):
            self.variables[f"l_{lid}"] = pos.astype(float)

        # Add robots and their factors
        for rid in range(robot_paths.shape[0]):
            # Extract measurements for this robot
            robot_meas = self._extract_robot_measurements(
                rid, 
                robot_paths.shape[1], 
                meas
            )
            self._add_robot(rid, robot_paths[rid], robot_meas)

        logger.info("✔ build done — factors:%d  vars:%d",
                    len(self.factors), len(self.variables))
        return self.factors, self.variables

    def _add_robot(self, rid: int, poses, obs_rlm):
        T = poses.shape[0]
        
        # Add pose variables
        for t in range(T):
            self.variables[f"x{rid}_{t}"] = poses[t].astype(float)

        # Add prior at t=0
        prior_sigma = np.array([
            self.cfg["prior_sigma_xy"],
            self.cfg["prior_sigma_xy"],
            self.cfg["prior_sigma_theta"]
        ])
        self.factors.append(PriorFactor(f"x{rid}_0", poses[0], prior_sigma))

        # Add odometry factors
        for t in range(T-1):
            dx, dy = poses[t+1, :2] - poses[t, :2]
            dtheta = wrap_angle(poses[t+1, 2] - poses[t, 2])
            delta = np.array([dx, dy, dtheta])
            self.factors.append(OdometryFactor(
                f"x{rid}_{t}", f"x{rid}_{t+1}", delta, self._odom_sigma
            ))

        # Add observation factors
        R = self._obs_R()
        for t, obs_list in enumerate(obs_rlm):
            if t >= T:
                continue
                
            pose_key = f"x{rid}_{t}"
            rob_pose = poses[t]

            for lm_id, z in obs_list:
                if self._is_visible(rob_pose, lm_id):
                    self.factors.append(BearingRangeUTFactor(
                        pose_key, f"l_{lm_id}", z.astype(float), R,
                        mode=self.cfg["ut_mode"],
                        alpha=self.cfg["ut_alpha"],
                        kappa=self.cfg["ut_kappa"]
                    ))

    def _is_visible(self, pose_xyθ, lid) -> bool:
        return lid < len(self._landmarks) and np.linalg.norm(
            pose_xyθ[:2] - self._landmarks[lid]
        ) <= self.cfg["max_obs_range"]

    def _obs_R(self) -> np.ndarray:
        if self._R_cache is None:
            self._R_cache = np.diag([
                self.cfg["obs_sigma_range"] ** 2,
                self.cfg["obs_sigma_bearing"] ** 2
            ])
        return self._R_cache.copy()

    def _check_cfg(self):
        if self.cfg["ut_mode"] not in ("gbp", "spbp", "auto"):
            raise ValueError("ut_mode must be gbp/spbp/auto")
        if not (0 < self.cfg["ut_alpha"] <= 1):
            raise ValueError("ut_alpha must be in (0,1]")
        if self.cfg["ut_kappa"] < 0:
            raise ValueError("ut_kappa >= 0 required")

    def _check_input(self, paths, lms, meas):
        if paths.ndim != 3 or paths.shape[2] != 3:
            raise ValueError("robot_paths must be (R,T,3)")
        if lms.ndim != 2 or lms.shape[1] != 2:
            raise ValueError("landmark_positions must be (L,2)")
        if not isinstance(meas, list):
            raise ValueError("meas must be a list")
    
    def _extract_robot_measurements(self, rid, time_steps, meas_list):
        """Extract measurements for a specific robot using defaultdict"""
        time_map = defaultdict(list)
        
        for m in meas_list:
            # Handle both measurement formats
            if (m.get("type") == "br_lm" and m.get("robot") == rid and 
                "time" in m and 0 <= m["time"] < time_steps):
                
                t = m["time"]
                if "bearing_range" in m:
                    z = np.asarray(m["bearing_range"], dtype=float)
                elif "range" in m and "bearing" in m:  # Backward compatibility
                    z = np.array([m["range"], m["bearing"]], dtype=float)
                else:
                    continue
                    
                lm_id = m.get("id", -1)
                time_map[t].append((lm_id, z))
        
        # Convert to list of lists
        return [time_map[t] for t in range(time_steps)]