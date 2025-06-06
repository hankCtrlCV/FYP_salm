"""GBP-Factor-Graph Builder (3-DoF pose: x, y, θ)"""

from __future__ import annotations
import math, logging, numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter

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
            
        # ============ 添加所有机器人到地标的观测因子 ============
        logger.debug("Adding robot-landmark observation factors...")
        obs_count = 0
        for m in meas:
            # 支持两种测量类型标识
            if m.get("type") in ["robot_lm", "br_lm"]:
                rid = m.get("robot")
                t = m.get("time")
                lid = m.get("id", -1)
                
                # 检查必要字段是否存在
                if rid is None or t is None or lid < 0:
                    continue
                    
                pose_key = f"x{rid}_{t}"
                lm_key = f"l_{lid}"
                
                # 确保变量存在
                if pose_key not in self.variables:
                    logger.debug(f"Skipping obs: Pose variable {pose_key} not found")
                    continue
                if lm_key not in self.variables:
                    logger.debug(f"Skipping obs: Landmark variable {lm_key} not found")
                    continue
                
                # 统一量测数据格式
                z = None
                if "bearing_range" in m:
                    z = np.asarray(m["bearing_range"], dtype=float)
                elif "bearing" in m and "range" in m:
                    z = np.array([m["bearing"], m["range"]], dtype=float)
                
                if z is None or len(z) != 2:
                    logger.warning(f"Invalid observation format for pose {pose_key} and landmark {lm_key}")
                    continue
                
                # 使用BearingRangeUTFactor创建因子
                self.factors.append(BearingRangeUTFactor(
                    pose_key,
                    lm_key,
                    z,
                    self._obs_R(),
                    mode=self.cfg["ut_mode"],
                    alpha=self.cfg["ut_alpha"],
                    kappa=self.cfg["ut_kappa"]
                ))
                obs_count += 1
                
        logger.info(f"Added {obs_count} bearing-range observation factors")
        
        # 打印因子统计
        self._log_factor_stats()
        
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
        
        logger.debug(f"Added {T} poses and {T-1} odometry factors for robot {rid}")

    def _log_factor_stats(self):
        """记录详细的因子统计信息"""
        factor_stats = Counter()
        landmark_observations = set()
        
        for f in self.factors:
            factor_name = type(f).__name__
            factor_stats[factor_name] += 1
            
            # 检查是否是观测因子，并记录关联的地标
            if factor_name == "BearingRangeUTFactor":
                # 获取变量的标准方式
                if hasattr(f, 'var1') and hasattr(f, 'var2'):
                    # 假设第二个变量是地标
                    landmark_var = f.var2
                elif hasattr(f, 'variables') and len(f.variables) >= 2:
                    # 从variables列表中获取地标变量
                    landmark_var = f.variables[1]
                else:
                    # 对于无法识别的因子，跳过
                    continue
                    
                # 检查是否是地标变量
                if landmark_var.startswith("l_"):
                    landmark_observations.add(landmark_var)
        
        logger.info("Factor graph statistics:")
        for ft, count in factor_stats.items():
            logger.info(f"  - {ft}: {count}")
        
        # 统计地标观测覆盖率
        all_landmarks = {f"l_{i}" for i in range(len(self._landmarks))}
        observed_count = len(landmark_observations)
        total_count = len(all_landmarks)
        coverage = observed_count / total_count * 100 if total_count else 0
        
        logger.info(f"  - Landmark observation coverage: {observed_count}/{total_count} ({coverage:.1f}%)")

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
        """Extract measurements for a specific robot using defaultdict - 改进版本"""
        time_map = defaultdict(list)
        
        for m in meas_list:
            # 支持两种测量类型标识
            if m.get("type") in ["robot_lm", "br_lm"]:
                m_rid = m.get("robot")
                t = m.get("time")
                
                # 检查机器人ID和时间
                if m_rid != rid or t is None or t < 0 or t >= time_steps:
                    continue
                    
                # 统一量测数据格式
                z = None
                if "bearing_range" in m:
                    z = np.asarray(m["bearing_range"], dtype=float)
                elif "bearing" in m and "range" in m:
                    z = np.array([m["bearing"], m["range"]], dtype=float)
                
                if z is None or len(z) != 2:
                    continue
                    
                lm_id = m.get("id", -1)
                if lm_id < 0:
                    continue
                    
                time_map[t].append((lm_id, z))
        
        # Convert to list of lists
        obs_list = []
        for t in range(time_steps):
            obs_list.append(time_map[t])
        
        logger.debug(f"Extracted {sum(len(o) for o in obs_list)} observations for robot {rid}")
        return obs_list