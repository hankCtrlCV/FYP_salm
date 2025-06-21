"""
Enhanced Measurement Generation for Multi-Robot SLAM
- Consistent with factor_ut.py and graph_build.py interfaces
- Proper coordinate frame handling
- Support for both centralized and distributed scenarios

Author: Enhanced for Multi-Robot SLAM
Date: 2025-06-18
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Any, Optional, Union
import dataclasses
import logging

logger = logging.getLogger("MeasurementGen")

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

@dataclasses.dataclass
class MeasurementBase:
    """Base class for all measurements"""
    time: int
    bearing: float  # radians, in observer's local frame
    range: float   # meters
    noise_std_bearing: float = math.radians(2.0)
    noise_std_range: float = 0.15
    
    @property
    def measurement_vector(self) -> np.ndarray:
        """Return measurement as [bearing, range] vector"""
        return np.array([self.bearing, self.range], dtype=np.float64)
    
    @property
    def noise_covariance(self) -> np.ndarray:
        """Return 2x2 noise covariance matrix"""
        return np.diag([self.noise_std_bearing**2, self.noise_std_range**2])

@dataclasses.dataclass
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
            "range": self.range,
            "noise_covariance": self.noise_covariance
        }

@dataclasses.dataclass 
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
            "range": self.range,
            "noise_covariance": self.noise_covariance
        }

@dataclasses.dataclass
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

class MeasurementGenerator:
    """
    Comprehensive measurement generator for multi-robot SLAM
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize measurement generator
        
        Args:
            config: Configuration dictionary
        """
        default_config = {
            # Noise parameters
            "bearing_noise_std": math.radians(2.0),
            "range_noise_std": 0.15,
            
            # Observation limits
            "max_landmark_range": 25.0,
            "min_landmark_range": 0.5,
            "max_robot_range": 20.0,
            "min_robot_range": 1.0,
            
            # Detection probabilities
            "landmark_detection_prob": 0.8,
            "robot_detection_prob": 0.6,
            
            # Field of view (full circle = 2π)
            "landmark_fov": _TWO_PI,
            "robot_fov": _TWO_PI,
            
            # Loop closure detection
            "enable_loop_closure": True,
            "loop_closure_distance_threshold": 2.0,
            "loop_closure_time_threshold": 10,
            
            # Noise options
            "add_noise": True,
            "adaptive_noise": False,
            "range_dependent_noise": False,
            
            # Multi-robot options
            "enable_inter_robot_measurements": True,
            "synchronous_observations": True,
            
            # Random seed
            "random_seed": 42
        }
        
        self.config = {**default_config, **(config or {})}
        self.rng = np.random.default_rng(self.config["random_seed"])
        
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
                    
                    # Check visibility constraints
                    if not self._is_landmark_visible(pose, landmark_pos, bearing, range_val):
                        continue
                    
                    # Probabilistic detection
                    if self.rng.random() > self.config["landmark_detection_prob"]:
                        continue
                    
                    # Add noise if enabled
                    if self.config["add_noise"]:
                        bearing, range_val = self._add_measurement_noise(bearing, range_val)
                    
                    # Create measurement
                    measurement = RobotLandmarkMeasurement(
                        time=time_step,
                        bearing=bearing,
                        range=range_val,
                        robot_id=robot_id,
                        landmark_id=landmark_id,
                        noise_std_bearing=self.config["bearing_noise_std"],
                        noise_std_range=self._get_range_noise_std(range_val)
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
                    observed_pos = observed_pose[:2]
                    
                    # Compute bearing and range
                    bearing, range_val = compute_bearing_range(observer_pose, observed_pos)
                    
                    # Check visibility constraints
                    if not self._is_robot_visible(observer_pose, observed_pos, bearing, range_val):
                        continue
                    
                    # Probabilistic detection
                    if self.rng.random() > self.config["robot_detection_prob"]:
                        continue
                    
                    # Add noise
                    if self.config["add_noise"]:
                        bearing, range_val = self._add_measurement_noise(bearing, range_val)
                    
                    # Create measurement
                    measurement = InterRobotMeasurement(
                        time=time_step,
                        bearing=bearing,
                        range=range_val,
                        observer_robot_id=observer_id,
                        observed_robot_id=observed_id,
                        observer_time=time_step,
                        observed_time=time_step,
                        noise_std_bearing=self.config["bearing_noise_std"],
                        noise_std_range=self._get_range_noise_std(range_val)
                    )
                    
                    measurements.append(measurement.to_dict())
        
        return measurements

    def _generate_loop_closures(self, robot_paths: List[np.ndarray]) -> List[Dict]:
        """Generate loop closure constraints"""
        loop_closures = []
        distance_threshold = self.config["loop_closure_distance_threshold"]
        time_threshold = self.config["loop_closure_time_threshold"]
        
        for robot1_id, path1 in enumerate(robot_paths):
            for robot2_id, path2 in enumerate(robot_paths):
                # Allow intra-robot and inter-robot loop closures
                
                for t1, pose1 in enumerate(path1):
                    start_t2 = t1 + time_threshold if robot1_id == robot2_id else 0
                    
                    for t2, pose2 in enumerate(path2):
                        if robot1_id == robot2_id and t2 < start_t2:
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

    def _is_landmark_visible(self, observer_pose: np.ndarray, landmark_pos: np.ndarray,
                           bearing: float, range_val: float) -> bool:
        """Check if landmark is visible from observer pose"""
        # Range check
        if range_val < self.config["min_landmark_range"] or range_val > self.config["max_landmark_range"]:
            return False
        
        # Field of view check
        half_fov = self.config["landmark_fov"] / 2
        if half_fov < math.pi and abs(bearing) > half_fov:
            return False
        
        return True

    def _is_robot_visible(self, observer_pose: np.ndarray, observed_pos: np.ndarray,
                         bearing: float, range_val: float) -> bool:
        """Check if robot is visible from observer pose"""
        # Range check
        if range_val < self.config["min_robot_range"] or range_val > self.config["max_robot_range"]:
            return False
        
        # Field of view check
        half_fov = self.config["robot_fov"] / 2
        if half_fov < math.pi and abs(bearing) > half_fov:
            return False
        
        return True

    def _add_measurement_noise(self, bearing: float, range_val: float) -> Tuple[float, float]:
        """Add noise to bearing and range measurements"""
        # Get noise standard deviations
        bearing_std = self.config["bearing_noise_std"]
        range_std = self._get_range_noise_std(range_val)
        
        # Add Gaussian noise
        noisy_bearing = bearing + self.rng.normal(0, bearing_std)
        noisy_range = range_val + self.rng.normal(0, range_std)
        
        # Wrap bearing and ensure positive range
        noisy_bearing = wrap_angle(noisy_bearing)
        noisy_range = max(noisy_range, 0.01)  # Minimum 1cm
        
        return noisy_bearing, noisy_range

    def _get_range_noise_std(self, range_val: float) -> float:
        """Get range noise standard deviation (possibly range-dependent)"""
        base_std = self.config["range_noise_std"]
        
        if self.config["range_dependent_noise"]:
            # Increase noise with distance (1% per meter)
            return base_std * (1.0 + 0.01 * range_val)
        
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
        """Log measurement generation statistics"""
        logger.info("=== Measurement Generation Summary ===")
        logger.info("Landmark measurements: %d", self.stats["landmark_measurements"])
        logger.info("Inter-robot measurements: %d", self.stats["inter_robot_measurements"])
        logger.info("Loop closures: %d", self.stats["loop_closures"])
        logger.info("Total measurements: %d", self.stats["total_measurements"])

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
                                    config: Optional[Dict] = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate measurements for multi-robot SLAM
    
    Args:
        robot_trajectories: Robot trajectories
        landmarks: Landmark positions  
        config: Optional configuration
        
    Returns:
        (measurements, loop_closures): Regular measurements and loop closure constraints
    """
    generator = MeasurementGenerator(config)
    return generator.generate_all_measurements(robot_trajectories, landmarks)

def generate_noisy_measurements(true_measurements: List[Dict],
                               noise_config: Dict) -> List[Dict]:
    """
    Add noise to existing measurements
    
    Args:
        true_measurements: Noiseless measurements
        noise_config: Noise configuration
        
    Returns:
        Noisy measurements
    """
    rng = np.random.default_rng(noise_config.get("random_seed", 42))
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
    'wrap_angle'
]