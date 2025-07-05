"""
Measurement Generation for Multi-Robot SLAM
- Consistent with factor_ut.py and graph_build.py interfaces
- Proper coordinate frame handling
- Support for both centralized and distributed scenarios

Author: Enhanced for Multi-Robot SLAM
Date: 2025-07-06
Version: 1.1 - Fixed critical issues identified in analysis
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Any, Optional, Union, Set
from utils.cfg_loader import load_common  
import dataclasses
import logging

logger = logging.getLogger("MeasurementGen")

# =====================================================================
# ✅ 修复6: 统一使用共同的工具函数，避免重复实现
# =====================================================================
try:
    from utils.cfg_loader import _flatten_yaml
except ImportError:
    def _flatten_yaml(d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback: 把多层 YAML 打平成 {leaf_key: value}，忽略中间层名字。
        e.g. {'noise': {'obs_sigma_bearing': 0.05}}  ➜  {'obs_sigma_bearing': 0.05}
        """
        flat = {}
        for k, v in d.items():
            if isinstance(v, dict):
                flat.update(_flatten_yaml(v))     # 递归但不带前缀
            else:
                flat[k] = v
        return flat

# Constants
_EPSILON = 1e-9
_TWO_PI = 2 * math.pi

def wrap_angle(theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Robust angle wrapping to [-π, π]"""
    if isinstance(theta, np.ndarray):
        return np.arctan2(np.sin(theta), np.cos(theta))
    else:
        return math.atan2(math.sin(theta), math.cos(theta))

def compute_bearing_range(observer_pose: np.ndarray, 
                         target_pos: np.ndarray) -> Tuple[float, float]:
    """
    Compute bearing and range from observer to target
    
    Args:
        observer_pose: [x, y, theta] observer pose
        target_pos: [x, y] target position
        
    Returns:
        (bearing, range): bearing in observer frame [-π, π], range in meters
    """
    # Extract observer position and orientation
    obs_x, obs_y, obs_theta = observer_pose
    target_x, target_y = target_pos
    
    # Compute displacement vector
    dx = target_x - obs_x
    dy = target_y - obs_y
    
    # Compute range
    range_val = math.hypot(dx, dy)
    
    # Compute bearing in global frame
    global_bearing = math.atan2(dy, dx)
    
    # Convert to observer's local frame
    local_bearing = wrap_angle(global_bearing - obs_theta)
    
    return local_bearing, range_val

@dataclasses.dataclass(kw_only=True)
class MeasurementBase:
    """Base class for all measurements"""
    time: int
    bearing: float  # radians, in observer's local frame
    range_val: float   # ✅ 修复3: 避免与内置range()冲突
    noise_std_bearing: float = math.radians(2.0)
    noise_std_range: float = 0.15
    
    @property
    def measurement_vector(self) -> np.ndarray:
        """Return measurement as [bearing, range] vector"""
        return np.array([self.bearing, self.range_val], dtype=np.float64)
    
    @property
    def noise_covariance(self) -> np.ndarray:
        """Return 2x2 noise covariance matrix"""
        return np.diag([self.noise_std_bearing**2, self.noise_std_range**2])

@dataclasses.dataclass(kw_only=True)
class RobotLandmarkMeasurement(MeasurementBase):
    """Robot observing landmark measurement"""
    robot_id: int
    landmark_id: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with graph_build.py"""
        return {
            "type": "robot_lm",
            "robot": self.robot_id,
            "time": self.time,
            "id": self.landmark_id,
            "bearing_range": self.measurement_vector.copy(),
            "bearing": self.bearing,
            "range": self.range_val,  # ✅ 修复3: 使用range_val
            "noise_covariance": self.noise_covariance
        }

@dataclasses.dataclass(kw_only=True)
class InterRobotMeasurement(MeasurementBase):
    """Robot observing another robot measurement"""
    observer_robot_id: int
    observed_robot_id: int
    observer_time: int
    observed_time: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for inter-robot constraints"""
        return {
            "type": "inter_robot",
            "observer_robot": self.observer_robot_id,
            "observer_time": self.observer_time,
            "observed_robot": self.observed_robot_id, 
            "observed_time": self.observed_time,
            "bearing_range": self.measurement_vector.copy(),
            "bearing": self.bearing,
            "range": self.range_val,  # ✅ 修复3: 使用range_val
            "noise_covariance": self.noise_covariance
        }

@dataclasses.dataclass(kw_only=True)
class LoopClosureMeasurement:
    """Loop closure measurement between poses"""
    robot1_id: int
    time1: int
    robot2_id: int
    time2: int
    relative_pose: np.ndarray  # [dx, dy, dtheta] 
    information_matrix: np.ndarray  # 3x3 information matrix
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for loop closure"""
        return {
            "type": "loop_closure",
            "robot1": self.robot1_id,
            "time1": self.time1,
            "robot2": self.robot2_id,
            "time2": self.time2,
            "relative_pose": self.relative_pose.copy(),
            "information_matrix": self.information_matrix.copy()
        }

class BearingRangeMeas:
    """✅ 修复3: 非dataclass实现的测量类，彻底避免参数名冲突"""
    def __init__(self, robot=None, time=None, id=None, bearing=None, 
                 range_val=None, *, legacy_range=None, **kwargs):  # ✅ 最安全做法：keyword-only legacy参数
        self.type = kwargs.get("type", "robot_lm")
        self.robot = robot
        self.time = time
        self.id = id
        self.bearing = bearing
        
        # ✅ 修复3: 彻底避免range参数名冲突
        if range_val is not None:
            self.range_val = range_val
        elif legacy_range is not None:
            self.range_val = legacy_range
            logger.warning("Parameter 'legacy_range' is deprecated, use 'range_val' instead")
        else:
            # 检查kwargs中是否有旧的range参数（向后兼容但发出警告）
            if 'range' in kwargs:
                self.range_val = kwargs.pop('range')
                logger.warning("Parameter 'range' shadows built-in function. Use 'range_val' instead.")
            else:
                self.range_val = None
            
        # --- 统一 bearing_range 表示 ---
        if 'bearing_range' in kwargs and kwargs['bearing_range'] is not None:
            # 若调用方直接传入 bearing_range（可能是 list / tuple / ndarray）
            self.bearing_range = np.asarray(kwargs.pop('bearing_range'), dtype=float)
        elif bearing is not None and self.range_val is not None:
            # 根据单独的 bearing 与 range 拼
            self.bearing_range = np.array([bearing, self.range_val], dtype=float)
        else:
            self.bearing_range = None
        # ---------------------------------

        # ✅ 修复2: 统一inter-robot接口
        if self.type == "inter_robot":
            self.observer_robot = kwargs.get("observer_robot")
            self.observer_time = kwargs.get("observer_time")
            self.observed_robot = kwargs.get("observed_robot")
            self.observed_time = kwargs.get("observed_time")

        # 其他可能的属性
        for key, value in kwargs.items():
            if key not in ['bearing_range', 'type', 'observer_robot', 'observer_time', 
                          'observed_robot', 'observed_time']:
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        仅返回 graph_build 解析所需字段，并把 ndarray → Python list，\
        以免 JSON 序列化或 deepcopy 出错。
        """
        if self.type == "robot_lm":
            base = {
                "type": "robot_lm",
                "robot": self.robot,
                "time": self.time,
                "id": self.id,
            }
        else:  # inter_robot
            base = {
                "type": "inter_robot",
                "observer_robot": self.observer_robot,
                "observer_time": self.observer_time,
                "observed_robot": self.observed_robot,
                "observed_time": self.observed_time,
            }

        # 公共测量字段
        if self.bearing is not None:
            base["bearing"] = float(self.bearing)
        if self.range_val is not None:
            base["range"] = float(self.range_val)  # graph_build期望"range"键
        if self.bearing_range is not None:
            base["bearing_range"] = (
                self.bearing_range.tolist()
                if hasattr(self.bearing_range, "tolist")
                else list(self.bearing_range)
            )
        if hasattr(self, "noise_covariance") and self.noise_covariance is not None:
            base["noise_covariance"] = self.noise_covariance.tolist()

        return base

def make_meas(**kwargs) -> "BearingRangeMeas":
    """✅ 兼容旧调用：make_meas(robot=..., time=..., ...)"""
    return BearingRangeMeas(**kwargs)


class MeasurementGenerator:
    """
    Comprehensive measurement generator for multi-robot SLAM
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, worker_id: int = 0):
        """
        Initialize measurement generator
        
        Args:
            config: Configuration dictionary
            worker_id: ✅ 修复8: Worker ID for multi-process safety
        """
        
        # ② 读取并打平 yaml
        yaml_cfg = _flatten_yaml(load_common())

        # ③ 根据 yaml 填充 Measurement 需要的字段；若 yaml 没提供则退回旧缺省
        default_cfg = {
            # ---- 噪声 ----
            "bearing_noise_std": yaml_cfg.get(
                "bearing_noise_std",
                yaml_cfg.get("obs_sigma_bearing", math.radians(2.0))),
            "range_noise_std": yaml_cfg.get(
                "range_noise_std",
                yaml_cfg.get("obs_sigma_range", 0.15)),

            # ---- 可见距离/角度 ----
            "max_landmark_range": yaml_cfg.get("max_landmark_range", 25.0),
            "min_landmark_range": yaml_cfg.get("min_landmark_range", 0.5),
            "max_robot_range":    yaml_cfg.get("max_robot_range",    20.0),
            "min_robot_range":    yaml_cfg.get("min_robot_range",    1.0),
            "landmark_fov":       yaml_cfg.get("landmark_fov",
                                               yaml_cfg.get("max_obs_angle", _TWO_PI)),
            "robot_fov":          yaml_cfg.get("robot_fov",
                                               yaml_cfg.get("max_obs_angle", _TWO_PI)),

            # ---- 探测概率 ----
            "landmark_detection_prob":
                yaml_cfg.get("landmark_detection_prob", 0.8),
            "robot_detection_prob":
                yaml_cfg.get("robot_detection_prob", 0.6),

            # ---- Loop-closure 阈值 ----
            "enable_loop_closure":
                yaml_cfg.get("enable_loop_closure", True),
            "loop_closure_distance_threshold":
                yaml_cfg.get("loop_closure_distance_threshold", 2.0),
            "loop_closure_time_threshold":
                yaml_cfg.get("loop_closure_time_threshold", 10),

            # ---- 核心功能配置 ----
            "add_noise": True,
            "range_dependent_noise": False,
            "enable_inter_robot_measurements": True,
            "random_seed": 42,
            
            # ✅ 修复5: 移除未实现的配置项，添加TODO注释
            # TODO: 未来可实现这些高级特性
            # "adaptive_noise": False,  # 自适应噪声模型
            # "synchronous_observations": True,  # 同步观测模式
        }

        # ④ 三层覆盖：`default_cfg` < `yaml_cfg` < 外部 `config`
        #    （yaml 已经被打平，但我们只取 Measurement 关心的键）
        merged_from_yaml = {k: yaml_cfg[k] for k in default_cfg.keys() if k in yaml_cfg}
        self.config = {**default_cfg, **merged_from_yaml, **(config or {})}
        
        # ✅ 修复8: 多进程安全的随机数生成
        base_seed = self.config["random_seed"]
        self.rng = np.random.default_rng(base_seed + worker_id * 10000)
        
        # Statistics
        self.stats = {
            "landmark_measurements": 0,
            "inter_robot_measurements": 0, 
            "loop_closures": 0,
            "total_measurements": 0
        }

    def generate_all_measurements(self, 
                                robot_trajectories: Union[np.ndarray, List[np.ndarray]],
                                landmark_positions: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate all types of measurements
        
        Args:
            robot_trajectories: Robot paths, either (R,T,3) array or list of (T_i,3) arrays
            landmark_positions: Landmark positions (L,2)
            
        Returns:
            (measurements, loop_closures): Regular measurements and loop closure constraints
        """
        logger.info("Generating measurements for multi-robot SLAM...")
        
        # Normalize input format
        robot_paths = self._normalize_robot_trajectories(robot_trajectories)
        self._validate_inputs(robot_paths, landmark_positions)
        
        # Reset statistics
        for key in self.stats:
            self.stats[key] = 0
        
        measurements = []
        loop_closures = []
        
        # Generate landmark observations
        landmark_meas = self._generate_landmark_measurements(robot_paths, landmark_positions)
        measurements.extend(landmark_meas)
        self.stats["landmark_measurements"] = len(landmark_meas)
        
        # Generate inter-robot measurements
        if self.config["enable_inter_robot_measurements"]:
            inter_robot_meas = self._generate_inter_robot_measurements(robot_paths)
            measurements.extend(inter_robot_meas)
            self.stats["inter_robot_measurements"] = len(inter_robot_meas)
        
        # Generate loop closures
        if self.config["enable_loop_closure"]:
            loop_closures = self._generate_loop_closures(robot_paths)
            self.stats["loop_closures"] = len(loop_closures)
        
        self.stats["total_measurements"] = len(measurements)
        
        self._log_statistics()
        
        return measurements, loop_closures

    def _normalize_robot_trajectories(self, trajectories) -> List[np.ndarray]:
        """Normalize robot trajectories to consistent format"""
        if isinstance(trajectories, np.ndarray):
            if trajectories.ndim == 3:
                # Format: (R, T, 3)
                return [trajectories[i] for i in range(trajectories.shape[0])]
            elif trajectories.ndim == 2:
                # Single robot: (T, 3)
                return [trajectories]
            else:
                raise ValueError(f"Invalid trajectory dimensions: {trajectories.shape}")
        elif isinstance(trajectories, list):
            return [np.asarray(traj, dtype=np.float64) for traj in trajectories]
        else:
            raise ValueError(f"Invalid trajectory type: {type(trajectories)}")

    def _generate_landmark_measurements(self, robot_paths: List[np.ndarray], 
                                      landmarks: np.ndarray) -> List[Dict]:
        """Generate robot-landmark observations"""
        measurements = []
        
        for robot_id, path in enumerate(robot_paths):
            for time_step, pose in enumerate(path):
                for landmark_id, landmark_pos in enumerate(landmarks):
                    
                    # Compute true bearing and range
                    bearing, range_val = compute_bearing_range(pose, landmark_pos)
                    
                    # ✅ 修复7: 简化可见性检查，移除未使用参数
                    if not self._is_landmark_visible(bearing, range_val):
                        continue
                    
                    # Probabilistic detection
                    if self.rng.random() > self.config["landmark_detection_prob"]:
                        continue
                    
                    # ✅ 修复4: 在加噪前计算噪声标准差
                    true_range = range_val  # 保存真实距离
                    bearing_std = self.config["bearing_noise_std"]
                    range_std = self._get_range_noise_std(true_range)  # 基于真实距离
                    
                    # Add noise if enabled
                    if self.config["add_noise"]:
                        bearing, range_val = self._add_measurement_noise(bearing, range_val, 
                                                                       bearing_std, range_std)
                    
                    # Create noise covariance matrix
                    noise_covariance = np.diag([bearing_std**2, range_std**2])

                    # 使用make_meas创建兼容对象
                    measurement = make_meas(
                        robot=robot_id,
                        time=time_step,
                        id=landmark_id,
                        bearing=bearing,
                        range_val=range_val,  # ✅ 修复3: 使用range_val
                        bearing_range=[bearing, range_val],
                        noise_covariance=noise_covariance
                    )

                    measurements.append(measurement.to_dict())
        
        return measurements

    def _generate_inter_robot_measurements(self, robot_paths: List[np.ndarray]) -> List[Dict]:
        """Generate inter-robot observations"""
        measurements = []
        num_robots = len(robot_paths)
        
        if num_robots < 2:
            return measurements
        
        # Find common time steps if trajectories have different lengths
        min_time_steps = min(len(path) for path in robot_paths)
        
        for time_step in range(min_time_steps):
            for observer_id in range(num_robots):
                observer_pose = robot_paths[observer_id][time_step]
                
                for observed_id in range(num_robots):
                    if observed_id == observer_id:
                        continue
                    
                    observed_pose = robot_paths[observed_id][time_step]
                    observed_pos = observed_pose[:2]  # ✅ 确认：只传位置给bearing_range计算
                    
                    # Compute bearing and range
                    bearing, range_val = compute_bearing_range(observer_pose, observed_pos)
                    
                    # ✅ 修复7: 简化可见性检查接口
                    if not self._is_robot_visible(bearing, range_val):
                        continue
                    
                    # Probabilistic detection
                    if self.rng.random() > self.config["robot_detection_prob"]:
                        continue
                    
                    # ✅ 修复4: 在加噪前计算噪声标准差
                    true_range = range_val
                    bearing_std = self.config["bearing_noise_std"]
                    range_std = self._get_range_noise_std(true_range)
                    
                    # Add noise
                    if self.config["add_noise"]:
                        bearing, range_val = self._add_measurement_noise(bearing, range_val,
                                                                       bearing_std, range_std)
                    
                    # Create measurement
                    noise_covariance = np.diag([bearing_std**2, range_std**2])
                    
                    measurement = make_meas(
                        type="inter_robot",  
                        observer_robot=observer_id,
                        observed_robot=observed_id,
                        observer_time=time_step,
                        observed_time=time_step,
                        bearing=bearing,
                        range_val=range_val,  # ✅ 修复3: 使用range_val
                        bearing_range=np.array([bearing, range_val], dtype=np.float64),
                        noise_covariance=noise_covariance,
                    )

                    measurements.append(measurement.to_dict())
        
        return measurements

    def _generate_loop_closures(self, robot_paths: List[np.ndarray]) -> List[Dict]:
        """✅ 修复1: Generate loop closure constraints with deduplication"""
        loop_closures = []
        distance_threshold = self.config["loop_closure_distance_threshold"]
        time_threshold = self.config["loop_closure_time_threshold"]
        
        # ✅ 修复1: 使用集合记录已添加的约束，避免重复
        added_constraints: Set[Tuple[int, int, int, int]] = set()
        
        for robot1_id, path1 in enumerate(robot_paths):
            for robot2_id, path2 in enumerate(robot_paths):
                # ✅ 修复1: 避免对称约束 - 只处理 robot1_id <= robot2_id 的情况
                if robot2_id < robot1_id:
                    continue
                
                for t1, pose1 in enumerate(path1):
                    # 同一机器人需要时间间隔，不同机器人不需要
                    start_t2 = t1 + time_threshold if robot1_id == robot2_id else 0
                    
                    for t2, pose2 in enumerate(path2):
                        if robot1_id == robot2_id and t2 <= start_t2:  # ✅ 避免t1=t2的情况
                            continue
                        
                        # ✅ 修复1: 检查是否已添加过这个约束（考虑方向性）
                        constraint_key = (robot1_id, t1, robot2_id, t2)
                        if constraint_key in added_constraints:
                            continue
                        
                        # Check if poses are close enough
                        distance = np.linalg.norm(pose1[:2] - pose2[:2])
                        if distance > distance_threshold:
                            continue
                        
                        # Compute relative pose
                        relative_pose = self._compute_relative_pose(pose1, pose2)
                        
                        # Create information matrix (inverse of covariance)
                        # More uncertain than odometry measurements
                        sigma_xy = self.config["range_noise_std"] * 2
                        sigma_theta = self.config["bearing_noise_std"] * 2
                        information = np.diag([1/sigma_xy**2, 1/sigma_xy**2, 1/sigma_theta**2])
                        
                        measurement = LoopClosureMeasurement(
                            robot1_id=robot1_id,
                            time1=t1,
                            robot2_id=robot2_id,
                            time2=t2,
                            relative_pose=relative_pose,
                            information_matrix=information
                        )
                        
                        loop_closures.append(measurement.to_dict())
                        added_constraints.add(constraint_key)  # ✅ 记录已添加的约束
        
        logger.debug(f"Generated {len(loop_closures)} unique loop closure constraints")
        return loop_closures

    def _compute_relative_pose(self, pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
        """
        Compute relative pose from pose1 to pose2 in SE(2)
        
        Args:
            pose1: [x1, y1, theta1] 
            pose2: [x2, y2, theta2]
            
        Returns:
            [dx, dy, dtheta] in pose1's local frame
        """
        x1, y1, theta1 = pose1
        x2, y2, theta2 = pose2
        
        # Global translation
        dx_global = x2 - x1
        dy_global = y2 - y1
        
        # Rotate to pose1's local frame
        cos_theta1 = math.cos(theta1)
        sin_theta1 = math.sin(theta1)
        
        dx_local = cos_theta1 * dx_global + sin_theta1 * dy_global
        dy_local = -sin_theta1 * dx_global + cos_theta1 * dy_global
        
        # Relative orientation
        dtheta = wrap_angle(theta2 - theta1)
        
        return np.array([dx_local, dy_local, dtheta], dtype=np.float64)

    def _is_landmark_visible(self, bearing: float, range_val: float) -> bool:
        """✅ 修复7: Check if landmark is visible (simplified interface)"""
        # Range check
        if range_val < self.config["min_landmark_range"] or range_val > self.config["max_landmark_range"]:
            return False
        
        # Field of view check
        half_fov = self.config["landmark_fov"] / 2
        if half_fov < math.pi and abs(bearing) > half_fov:
            return False
        
        return True

    def _is_robot_visible(self, bearing: float, range_val: float) -> bool:
        """✅ 修复7: Check if robot is visible (simplified interface)"""
        # Range check
        if range_val < self.config["min_robot_range"] or range_val > self.config["max_robot_range"]:
            return False
        
        # Field of view check
        half_fov = self.config["robot_fov"] / 2
        if half_fov < math.pi and abs(bearing) > half_fov:
            return False
        
        return True

    def _add_measurement_noise(self, bearing: float, range_val: float,
                             bearing_std: float, range_std: float) -> Tuple[float, float]:
        """✅ 修复4: Add noise to bearing and range measurements with pre-computed std"""
        # Add Gaussian noise
        noisy_bearing = bearing + self.rng.normal(0, bearing_std)
        noisy_range = range_val + self.rng.normal(0, range_std)
        
        # Wrap bearing and ensure positive range
        noisy_bearing = wrap_angle(noisy_bearing)
        noisy_range = max(noisy_range, 0.01)  # Minimum 1cm
        
        return noisy_bearing, noisy_range

    def _get_range_noise_std(self, true_range_val: float) -> float:
        """✅ 修复4: Get range noise standard deviation based on true range"""
        base_std = self.config["range_noise_std"]
        
        if self.config["range_dependent_noise"]:
            # Increase noise with distance (1% per meter)
            return base_std * (1.0 + 0.01 * true_range_val)
        
        return base_std

    def _validate_inputs(self, robot_paths: List[np.ndarray], landmarks: np.ndarray):
        """Validate input data"""
        if not robot_paths:
            raise ValueError("No robot trajectories provided")
        
        for i, path in enumerate(robot_paths):
            if path.ndim != 2 or path.shape[1] != 3:
                raise ValueError(f"Robot {i} trajectory must be (T, 3), got {path.shape}")
        
        if landmarks.ndim != 2 or landmarks.shape[1] != 2:
            raise ValueError(f"Landmarks must be (L, 2), got {landmarks.shape}")

    def _log_statistics(self):
        """✅ 修复9: Log measurement generation statistics with debug level for large datasets"""
        if self.stats["total_measurements"] > 1000:
            # 大数据集使用debug级别，避免日志过多
            log_func = logger.debug
        else:
            log_func = logger.info
            
        log_func("=== Measurement Generation Summary ===")
        log_func("Landmark measurements: %d", self.stats["landmark_measurements"])
        log_func("Inter-robot measurements: %d", self.stats["inter_robot_measurements"])
        log_func("Loop closures: %d", self.stats["loop_closures"])
        log_func("Total measurements: %d", self.stats["total_measurements"])

# =====================================================================
# Convenience Functions
# =====================================================================

def generate_single_robot_measurements(trajectory: np.ndarray,
                                     landmarks: np.ndarray,
                                     config: Optional[Dict] = None) -> List[Dict]:
    """
    Generate measurements for single robot SLAM
    
    Args:
        trajectory: Robot trajectory (T, 3)
        landmarks: Landmark positions (L, 2)
        config: Optional configuration
        
    Returns:
        List of measurement dictionaries
    """
    generator = MeasurementGenerator(config)
    measurements, _ = generator.generate_all_measurements([trajectory], landmarks)
    return measurements

def generate_multi_robot_measurements(robot_trajectories: Union[np.ndarray, List[np.ndarray]],
                                    landmarks: np.ndarray,
                                    config: Optional[Dict] = None,
                                    worker_id: int = 0) -> Tuple[List[Dict], List[Dict]]:
    """
    ✅ 修复8: Generate measurements for multi-robot SLAM with worker ID support
    
    Args:
        robot_trajectories: Robot trajectories
        landmarks: Landmark positions  
        config: Optional configuration
        worker_id: Worker ID for multi-process safety
        
    Returns:
        (measurements, loop_closures): Regular measurements and loop closure constraints
    """
    generator = MeasurementGenerator(config, worker_id)
    return generator.generate_all_measurements(robot_trajectories, landmarks)

def generate_noisy_measurements(true_measurements: List[Dict],
                               noise_config: Dict,
                               worker_id: int = 0) -> List[Dict]:
    """
    ✅ 修复8: Add noise to existing measurements with worker ID support
    
    Args:
        true_measurements: Noiseless measurements
        noise_config: Noise configuration
        worker_id: Worker ID for multi-process safety
        
    Returns:
        Noisy measurements
    """
    base_seed = noise_config.get("random_seed", 42)
    rng = np.random.default_rng(base_seed + worker_id * 10000)
    bearing_std = noise_config.get("bearing_noise_std", math.radians(2.0))
    range_std = noise_config.get("range_noise_std", 0.15)
    
    noisy_measurements = []
    
    for measurement in true_measurements:
        if "bearing_range" in measurement:
            bearing, range_val = measurement["bearing_range"]
            
            # Add noise
            noisy_bearing = bearing + rng.normal(0, bearing_std)
            noisy_range = range_val + rng.normal(0, range_std)
            
            # Wrap and clamp
            noisy_bearing = wrap_angle(noisy_bearing)
            noisy_range = max(noisy_range, 0.01)
            
            # Update measurement
            noisy_measurement = measurement.copy()
            noisy_measurement["bearing_range"] = np.array([noisy_bearing, noisy_range])
            if "bearing" in noisy_measurement:
                noisy_measurement["bearing"] = noisy_bearing
            if "range" in noisy_measurement:
                noisy_measurement["range"] = noisy_range
            noisy_measurements.append(noisy_measurement)
        else:
            noisy_measurements.append(measurement.copy())
    
    return noisy_measurements


# Export main classes and functions
__all__ = [
    'MeasurementGenerator', 
    'RobotLandmarkMeasurement', 
    'InterRobotMeasurement', 
    'LoopClosureMeasurement',
    'generate_single_robot_measurements',
    'generate_multi_robot_measurements', 
    'generate_noisy_measurements',
    'compute_bearing_range',
    'wrap_angle',
    'BearingRangeMeas',
    'make_meas',
]