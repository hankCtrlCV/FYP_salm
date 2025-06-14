"""Optimized GBP-Factor-Graph Builder (3-DoF pose: x, y, θ)"""

from __future__ import annotations
import math, logging, numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict, Counter

from algorithm.frontend.factor_ut import (
    PriorFactor, OdometryFactor, BearingRangeUTFactor
)

logger = logging.getLogger("GBPBuilder")

def wrap_angle(theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wrap angle(s) to [-π, π] efficiently"""
    if isinstance(theta, np.ndarray):
        return np.arctan2(np.sin(theta), np.cos(theta))
    return math.atan2(math.sin(theta), math.cos(theta))

class GBPGraphBuilder:
    _DEFAULT_CONFIG = dict(
        prior_sigma_xy=0.10, prior_sigma_theta=0.05,
        odom_sigma_xy=0.10, odom_sigma_theta=0.02,
        obs_sigma_range=0.30, obs_sigma_bearing=math.radians(2.0),
        ut_mode="auto", ut_alpha=0.30, ut_kappa=2.0,
        max_obs_range=25.0, max_obs_angle=math.pi,  # Add FOV support
        enable_visibility_check=True
    )

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = {**self._DEFAULT_CONFIG, **(cfg or {})}
        self._validate_config()
        
        self.factors = []
        self.variables = {}
        self._R_cache = None
        self._landmarks = None
        self.robot_paths = None
        
        # Precompute odometry sigma
        self._odom_sigma = np.array([
            self.cfg["odom_sigma_xy"],
            self.cfg["odom_sigma_xy"],
            self.cfg["odom_sigma_theta"]
        ], dtype=np.float64)  # Ensure consistent precision

    def build(self, robot_paths: np.ndarray, landmark_pos: np.ndarray, 
             measurements: List[Dict]) -> Tuple[List, Dict]:
        """Build the factor graph with enhanced validation"""
        self._validate_input(robot_paths, landmark_pos, measurements)
        self._reset_state()
        self._landmarks = landmark_pos.copy()
        self.robot_paths = robot_paths

        # Add landmarks
        for lid, pos in enumerate(self._landmarks):
            self.variables[f"l_{lid}"] = pos.astype(np.float64)  # Consistent precision

        # Add robots
        for rid in range(robot_paths.shape[0]):
            self._add_robot(rid, robot_paths[rid])
            
        # Add observations
        self._add_observation_factors(measurements)
        
        # Log detailed stats
        self._log_factor_stats()
        
        logger.info("Build complete - factors: %d, variables: %d", 
                   len(self.factors), len(self.variables))
        return self.factors, self.variables

    def _add_robot(self, rid: int, poses: np.ndarray):
        """Add robot with optimized odometry calculation"""
        T = poses.shape[0]
        
        # Add pose variables
        for t in range(T):
            self.variables[f"x{rid}_{t}"] = poses[t].astype(np.float64)

        # Add prior factor
        prior_sigma = np.array([
            self.cfg["prior_sigma_xy"],
            self.cfg["prior_sigma_xy"],
            self.cfg["prior_sigma_theta"]
        ], dtype=np.float64)
        self.factors.append(PriorFactor(f"x{rid}_0", poses[0], prior_sigma))

        # Add odometry factors with direct calculations (avoid matrix mult)
        for t in range(T-1):
            pose_t = poses[t]
            pose_t1 = poses[t+1]
            
            # Direct trigonometric calculations (faster than matrix)
            ct = math.cos(pose_t[2])
            st = math.sin(pose_t[2])
            
            # World frame displacement
            dx_world = pose_t1[0] - pose_t[0]
            dy_world = pose_t1[1] - pose_t[1]
            
            # Convert to body frame
            dx_body = ct * dx_world + st * dy_world
            dy_body = -st * dx_world + ct * dy_world
            dtheta = wrap_angle(pose_t1[2] - pose_t[2])
            
            delta = np.array([dx_body, dy_body, dtheta], dtype=np.float64)
            self.factors.append(OdometryFactor(
                f"x{rid}_{t}", f"x{rid}_{t+1}", delta, self._odom_sigma
            ))
        
        logger.debug("Added %d poses and %d odometry for robot %d", T, T-1, rid)

    def _add_observation_factors(self, measurements: List[Dict]):
        """Add observations with optimized visibility check"""
        obs_count = 0
        skipped_count = 0
        
        for m in measurements:
            if m.get("type") not in ["robot_lm", "br_lm"]:
                continue
                
            rid = m.get("robot")
            t = m.get("time")
            lid = m.get("id", -1)
            
            # Validation checks
            if rid is None or t is None or lid < 0 or lid >= len(self._landmarks):
                skipped_count += 1
                continue
                
            if rid < 0 or rid >= self.robot_paths.shape[0]:
                skipped_count += 1
                continue
                
            # Per-robot time validation
            robot_path = self.robot_paths[rid]
            if t < 0 or t >= robot_path.shape[0]:
                skipped_count += 1
                continue
                
            # Create keys
            pose_key = f"x{rid}_{t}"
            lm_key = f"l_{lid}"
            
            # Skip if variables missing
            if pose_key not in self.variables or lm_key not in self.variables:
                skipped_count += 1
                continue
            
            # Extract and normalize measurement
            z = self._extract_measurement(m)
            if z is None:
                skipped_count += 1
                continue
                
            # Visibility check (optimized)
            if self.cfg["enable_visibility_check"]:
                pose = self.variables[pose_key]
                lm_pos = self._landmarks[lid]
                if not self._is_visible(pose, lm_pos):
                    skipped_count += 1
                    continue
                
            # Create factor with explicit landmark key
            factor = BearingRangeUTFactor(
                pose_key,
                lm_key,
                z,
                self._obs_R(),
                mode=self.cfg["ut_mode"],
                alpha=self.cfg["ut_alpha"],
                kappa=self.cfg["ut_kappa"]
            )
            factor.landmark_key = lm_key
            self.factors.append(factor)
            obs_count += 1
            
        logger.info("Added %d bearing-range observations [bearing, range] (skipped %d)", 
           obs_count, skipped_count)

    def _extract_measurement(self, measurement: Dict) -> Optional[np.ndarray]:
        """Extract measurement ensuring [bearing, range] order with proper units"""
        z = None
        
        if "bearing_range" in measurement:
            br = measurement["bearing_range"]
            if not isinstance(br, (list, tuple, np.ndarray)) or len(br) != 2:
                return None
            bearing = float(br[0])
            range_val = float(br[1])
        elif "bearing" in measurement and "range" in measurement:
            bearing = measurement["bearing"]
            range_val = measurement["range"]
        else:
            return None
            
        # Convert bearing to radians if needed (detect degrees)
        if isinstance(bearing, (int, float)) and abs(bearing) > 2 * math.pi:
            bearing = math.radians(bearing)
            
        # Normalize bearing
        bearing = wrap_angle(float(bearing))
        
        return np.array([bearing, range_val], dtype=np.float64)  # [bearing, range]

    def _is_visible(self, pose: np.ndarray, lm_pos: np.ndarray) -> bool:
        """Enhanced visibility check with FOV support"""
        """Note: This uses standard [bearing, range] format internally"""
        # Distance check
        dx = lm_pos[0] - pose[0]
        dy = lm_pos[1] - pose[1]
        distance = math.hypot(dx, dy)
        
        if distance > self.cfg["max_obs_range"]:
            return False
            
        # Field-of-view check if configured
        if self.cfg["max_obs_angle"] < math.pi:
            angle_to_lm = wrap_angle(math.atan2(dy, dx) - pose[2])
            if abs(angle_to_lm) > self.cfg["max_obs_angle"] / 2:
                return False
                
        return True

    def _log_factor_stats(self):
        """Efficient factor statistics logging"""
        factor_counts = Counter()
        landmark_obs = defaultdict(int)
        
        for f in self.factors:
            name = type(f).__name__
            factor_counts[name] += 1
            
            if isinstance(f, BearingRangeUTFactor):
                if hasattr(f, 'landmark_key') and f.landmark_key.startswith("l_"):
                    landmark_obs[f.landmark_key] += 1
        
        total_factors = len(self.factors)
        logger.info("Factor graph statistics:")
        for name, count in factor_counts.items():
            logger.info("  - %s: %d (%.1f%%)", name, count, count/total_factors*100)
        
        # Landmark coverage stats
        total_lms = len(self._landmarks)
        observed_lms = len(landmark_obs)
        avg_obs = sum(landmark_obs.values()) / observed_lms if observed_lms else 0
        
        logger.info("Landmarks: observed %d/%d (%.1f%%), avg obs %.2f",
                   observed_lms, total_lms, observed_lms/total_lms*100, avg_obs)

    def _obs_R(self) -> np.ndarray:
        """Noise covariance with precision control"""
        if self._R_cache is None:
            self._R_cache = np.diag([
                self.cfg["obs_sigma_bearing"] ** 2,
                self.cfg["obs_sigma_range"] ** 2
            ]).astype(np.float64)  # Ensure consistent precision
        return self._R_cache.copy()

    def _validate_config(self):
        """Extended configuration validation"""
        if self.cfg["ut_mode"] not in ("gbp", "spbp", "auto"):
            raise ValueError("ut_mode must be 'gbp', 'spbp', or 'auto'")
        if not (0 < self.cfg["ut_alpha"] <= 1):
            raise ValueError("ut_alpha must be in (0,1]")
        if self.cfg["ut_kappa"] < 0:
            raise ValueError("ut_kappa must be >= 0")
        if self.cfg["max_obs_range"] <= 0:
            raise ValueError("max_obs_range must be > 0")
        if not (0 < self.cfg["max_obs_angle"] <= 2*math.pi):
            raise ValueError("max_obs_angle must be in (0, 2π]")

    def _validate_input(self, paths, lms, meas):
        """Enhanced input validation"""
        if paths.ndim != 3 or paths.shape[2] != 3:
            raise ValueError("robot_paths must be 3D array (R, T, 3)")
        
        # Allow different trajectory lengths
        if paths.shape[0] > 1:
            lengths = [p.shape[0] for p in paths]
            if max(lengths) - min(lengths) > 1:
                logger.warning("Trajectories have varying lengths (%s), using max", lengths)
                
        if lms.ndim != 2 or lms.shape[1] != 2:
            raise ValueError("landmarks must be 2D array (L, 2)")
        if not isinstance(meas, list):
            raise ValueError("measurements must be a list")
        if not meas:
            logger.warning("No measurements provided")
        if not all(isinstance(m, dict) for m in meas):
            raise ValueError("All measurements must be dictionaries")

    def _reset_state(self):
        """Complete state reset"""
        self.factors.clear()
        self.variables.clear()
        self._R_cache = None
        self._landmarks = None
        self.robot_paths = None

    @classmethod
    def from_yaml(cls, file_path: str):
        """Optional: Load configuration from YAML file"""
        try:
            import yaml
            with open(file_path, 'r') as f:
                cfg = yaml.safe_load(f)
            return cls(cfg)
        except ImportError:
            logger.warning("PyYAML not installed, using defaults")
            return cls()
        except Exception as e:
            logger.error("YAML load failed: %s", str(e))
            return cls()
            
    def to_yaml(self, file_path: str):
        """Optional: Save configuration to YAML file"""
        try:
            import yaml
            with open(file_path, 'w') as f:
                yaml.dump(self.cfg, f)
            return True
        except ImportError:
            logger.warning("PyYAML not installed, save skipped")
            return False
        except Exception as e:
            logger.error("YAML save failed: %s", str(e))
            return False