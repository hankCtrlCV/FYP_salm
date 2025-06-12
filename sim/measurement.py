"""
Robust Measurement Generation for Multi-Robot SLAM
- Strict numerical stability
- Comprehensive validation checks
- Explicit coordinate frame handling
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Any, Optional, Union
import dataclasses

# 数值常数
_EPSILON = 1e-9
_RADIAN_FULL_CIRCLE = 2 * math.pi

@dataclasses.dataclass
class Measurement:
    """严格类型的测量数据类"""
    time: int
    cov: np.ndarray  # 2x2 covariance matrix
    measured_range: float
    measured_bearing: float  # [-π, π] in rad
    
    @property
    def values(self) -> Tuple[float, float]:
        return (self.measured_range, self.measured_bearing)

@dataclasses.dataclass
class RobotLandmarkMeasurement(Measurement):
    """机器人观测地标测量"""
    robot_id: int
    landmark_id: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "robot_lm",
            "robot": self.robot_id,
            "time": self.time,
            "id": self.landmark_id,
            "bearing_range": np.array([self.measured_range, self.measured_bearing])
        }

@dataclasses.dataclass
class RobotRobotMeasurement(Measurement):
    """机器人观测另一个机器人测量"""
    observer_id: int
    observed_id: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "robot_robot",
            "from_robot": self.observer_id,
            "from_time": self.time,
            "to_robot": self.observed_id,
            "to_time": self.time,
            "bearing_range": np.array([self.measured_range, self.measured_bearing])
        }

def wrap_angle(theta: float) -> float:
    """数学精确的角度归一化"""
    return math.atan2(math.sin(theta), math.cos(theta))

def relative_bearing(
    observer_pos: np.ndarray, 
    observed_pos: np.ndarray,
    observer_yaw: Optional[float] = None
) -> Tuple[float, float]:
    """
    计算两个点之间的相对位置（距离和方位角）
    可选：减去观察者偏航角获得相对方位
    
    Args:
        observer_pos: (x, y) 观察者位置
        observed_pos: (x, y) 被观察点位置
        observer_yaw: 观察者偏航角（弧度）
    
    Returns:
        tuple: (range, bearing) 单位: meters, radians
    """
    # 检查输入维度
    if observer_pos.shape != (2,) or observed_pos.shape != (2,):
        raise ValueError("Positions must be 2D vectors")
    
    # 计算位移向量
    delta = observed_pos - observer_pos
    
    # 计算距离
    distance = math.hypot(delta[0], delta[1])
    
    # 计算全局方位角（从正x轴逆时针）
    global_bearing = math.atan2(delta[1], delta[0])
    
    # 转换为相对方位
    if observer_yaw is not None:
        relative_bearing_ = wrap_angle(global_bearing - observer_yaw)
        return distance, relative_bearing_
    
    return distance, global_bearing

def generate_measurements(
    robot_paths: np.ndarray,  # (num_robots, timesteps, 3) [x, y, theta]
    landmark_positions: np.ndarray,  # (num_landmarks, 2)
    *,
    # 观测模型参数
    range_noise_std: float = 0.30,       # 距离噪声标准差 (m)
    bearing_noise_std: float = math.radians(2.0),  # 方位角噪声标准差 (rad)
    # 可见性参数
    max_landmark_range: float = 25.0,    # 最大地标可见距离 (m)
    max_robot_range: float = 20.0,       # 最大机器人可见距离 (m)
    landmark_detection_prob: float = 1.0,  # 地标检测概率
    robot_detection_prob: float = 0.5,   # 机器人检测概率
    add_noise: bool = True,
    random_seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    为多机器人SLAM系统生成模拟测量数据
    
    Args:
        robot_paths: 形状 (R, T, 3) 的机器人轨迹，每个位姿为 [x, y, theta]
        landmark_positions: 形状 (L, 2) 的地标位置
        range_noise_std: 距离测量噪声标准差
        bearing_noise_std: 方位测量噪声标准差（弧度）
        max_landmark_range: 地标最大可见范围（米）
        max_robot_range: 机器人最大可见范围（米）
        landmark_detection_prob: 每个地标的检测概率
        robot_detection_prob: 机器人相互检测概率
        add_noise: 是否添加测量噪声
        random_seed: 随机数生成种子
    
    Returns:
        测量字典列表，每个字典包含：
        - 对于机器人-地标测量：
            type: "robot_lm"
            robot: 机器人ID (int)
            time: 时间步 (int)
            id: 地标ID (int)
            bearing_range: [距离, 相对方位]
        
        - 对于机器人-机器人测量：
            type: "robot_robot"
            from_robot: 观察者ID (int)
            from_time: 观察时间步 (int)
            to_robot: 被观察机器人ID (int)
            to_time: 被观察时间步 (int)
            bearing_range: [距离, 相对方位]
    """
    # 严格的输入验证
    if robot_paths.ndim != 3 or robot_paths.shape[2] != 3:
        raise ValueError("robot_paths must be 3D array with shape (R, T, 3)")
    
    if landmark_positions.ndim != 2 or landmark_positions.shape[1] != 2:
        raise ValueError("landmark_positions must be 2D array with shape (L, 2)")
    
    if not 0.0 <= landmark_detection_prob <= 1.0:
        raise ValueError("landmark_detection_prob must be in [0, 1]")
    
    if not 0.0 <= robot_detection_prob <= 1.0:
        raise ValueError("robot_detection_prob must be in [0, 1]")
    
    # 初始化随机数生成器
    rng = np.random.default_rng(random_seed)
    
    # 提取维度信息
    num_robots, timesteps, _ = robot_paths.shape
    num_landmarks = landmark_positions.shape[0]
    
    # 存储所有测量的列表
    measurements = []
    
    # ====== 机器人 → 地标观测 ======
    for robot_id in range(num_robots):
        for time_step in range(timesteps):
            # 提取当前机器人状态
            current_state = robot_paths[robot_id, time_step]
            position = current_state[:2]
            yaw = current_state[2]
            
            # 计算到所有地标的距离
            deltas = landmark_positions - position
            distances = np.hypot(deltas[:, 0], deltas[:, 1])
            
            # 识别在范围内的地标
            in_range_mask = distances < max_landmark_range
            in_range_indices = np.where(in_range_mask)[0]
            
            if in_range_indices.size == 0:
                continue
                
            # 概率检测：随机决定哪些地标被实际检测到
            detection_mask = rng.random(len(in_range_indices)) < landmark_detection_prob
            detected_indices = in_range_indices[detection_mask]
            
            if detected_indices.size == 0:
                continue
                
            # 处理每个检测到的地标
            for lm_index in detected_indices:
                # 提取地标位置
                lm_pos = landmark_positions[lm_index]
                
                # 计算真实距离和方位
                true_distance, true_bearing = relative_bearing(
                    position, lm_pos, observer_yaw=yaw
                )
                
                # 添加测量噪声
                if add_noise:
                    noisy_distance = true_distance + rng.normal(0, range_noise_std)
                    noisy_bearing = true_bearing + rng.normal(0, bearing_noise_std)
                    noisy_bearing = wrap_angle(noisy_bearing)  # 归一化角度
                else:
                    noisy_distance, noisy_bearing = true_distance, true_bearing
                
                # 创建测量对象
                measurement = RobotLandmarkMeasurement(
                    time=time_step,
                    cov=np.diag([range_noise_std**2, bearing_noise_std**2]),
                    measured_range=noisy_distance,
                    measured_bearing=noisy_bearing,
                    robot_id=robot_id,
                    landmark_id=lm_index
                )
                
                # 添加到结果列表
                measurements.append(measurement.to_dict())
    
    # ====== 机器人 ↔ 机器人观测 ======
    for time_step in range(timesteps):
        for observer_id in range(num_robots):
            # 提取观察者状态
            observer_state = robot_paths[observer_id, time_step]
            observer_pos = observer_state[:2]
            observer_yaw = observer_state[2]
            
            for observed_id in range(num_robots):
                # 跳过自观测
                if observed_id == observer_id:
                    continue
                
                # 提取被观察机器人状态
                observed_state = robot_paths[observed_id, time_step]
                observed_pos = observed_state[:2]
                
                # 计算距离
                distance = math.hypot(observed_pos[0] - observer_pos[0],
                                      observed_pos[1] - observer_pos[1])
                
                # 检查是否在范围内并概率检测
                if distance > max_robot_range or rng.random() > robot_detection_prob:
                    continue
                
                # 计算真实相对方位
                true_distance, true_bearing = relative_bearing(
                    observer_pos, observed_pos, observer_yaw
                )
                
                # 添加测量噪声
                if add_noise:
                    noisy_distance = true_distance + rng.normal(0, range_noise_std)
                    noisy_bearing = true_bearing + rng.normal(0, bearing_noise_std)
                    noisy_bearing = wrap_angle(noisy_bearing)  # 归一化角度
                else:
                    noisy_distance, noisy_bearing = true_distance, true_bearing
                
                # 创建测量对象
                measurement = RobotRobotMeasurement(
                    time=time_step,
                    cov=np.diag([range_noise_std**2, bearing_noise_std**2]),
                    measured_range=noisy_distance,
                    measured_bearing=noisy_bearing,
                    observer_id=observer_id,
                    observed_id=observed_id
                )
                
                # 添加到结果列表
                measurements.append(measurement.to_dict())
    
    return measurements