"""
Enhanced World Generation for Multi-Robot SLAM
- Robust trajectory generation with proper orientation handling
- Multiple distribution support for landmarks
- Consistent coordinate frame handling
- Full compatibility with measurement and graph building modules

Author: Enhanced for Multi-Robot SLAM
Date: 2025-07-06
Version: 1.1 - Fixed critical reproducibility and collision issues
"""

import numpy as np
import math
from typing import Union, List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger("WorldGen")

def wrap_angle(theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Robust angle wrapping to [-π, π]"""
    if isinstance(theta, np.ndarray):
        return np.arctan2(np.sin(theta), np.cos(theta))
    else:
        return math.atan2(math.sin(theta), math.cos(theta))

def landmarks(
    num: int = 5,
    limits: Union[float, Tuple[float, float]] = 20.0,
    distribution: str = "uniform",
    random_seed: int = 42,
    worker_id: int = 0,
    **kwargs
) -> np.ndarray:
    """
    ✅ 修复1: Generate landmark positions with various distributions and proper RNG
    
    Args:
        num: Number of landmarks
        limits: Spatial limits (scalar for symmetric, tuple for [x_lim, y_lim])
        distribution: Distribution type ("uniform", "gaussian", "cluster", "grid")
        random_seed: Random seed for reproducibility
        worker_id: Worker ID for multi-process safety
        **kwargs: Distribution-specific parameters
        
    Returns:
        Landmark positions (num, 2)
    """
    if num <= 0:
        raise ValueError("Number of landmarks must be positive")
    
    # ✅ 修复1: 可复现的RNG，支持多进程
    rng = np.random.default_rng(random_seed + worker_id * 10000)
    
    # Handle limits parameter
    if isinstance(limits, (int, float)):
        x_lim, y_lim = float(limits), float(limits)
    else:
        x_lim, y_lim = float(limits[0]), float(limits[1])
    
    if distribution == "uniform":
        return rng.uniform([-x_lim, -y_lim], [x_lim, y_lim], size=(num, 2))
    
    elif distribution == "gaussian":
        mu = kwargs.get("mu", [0.0, 0.0])
        sigma = kwargs.get("sigma", [x_lim/3, y_lim/3])
        
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)
        
        if mu.size == 1:
            mu = np.full(2, mu.item())
        if sigma.size == 1:
            sigma = np.full(2, sigma.item())
            
        return rng.normal(mu, sigma, size=(num, 2))
    
    elif distribution == "cluster":
        num_clusters = kwargs.get("num_clusters", 3)
        cluster_std = kwargs.get("cluster_std", min(x_lim, y_lim) / 10)
        
        # ✅ 修复4: 确保每个cluster至少有一个landmark
        num_clusters = min(num_clusters, num)  # 不能超过landmark数量
        
        # Generate cluster centers
        cluster_centers = rng.uniform(
            [-x_lim*0.8, -y_lim*0.8], 
            [x_lim*0.8, y_lim*0.8], 
            size=(num_clusters, 2)
        )
        
        # 确保每个cluster至少分配一个landmark
        landmarks_per_cluster = np.full(num_clusters, num // num_clusters)
        remaining = num % num_clusters
        landmarks_per_cluster[:remaining] += 1
        
        landmarks_pos = np.zeros((num, 2))
        idx = 0
        for i in range(num_clusters):
            count = landmarks_per_cluster[i]
            landmarks_pos[idx:idx+count] = (
                cluster_centers[i] + 
                rng.normal(0, cluster_std, size=(count, 2))
            )
            idx += count
        
        return landmarks_pos
    
    elif distribution == "grid":
        # ✅ 修复3: 限制grid_noise避免重叠
        grid_noise = kwargs.get("noise", 0.0)
        
        # Determine grid dimensions
        cols = int(np.ceil(np.sqrt(num)))
        rows = int(np.ceil(num / cols))
        
        # Generate grid points
        x_spacing = 2 * x_lim / (cols + 1)
        y_spacing = 2 * y_lim / (rows + 1)
        
        # ✅ 修复3: 限制噪声幅度，避免landmarks重叠
        max_noise = min(x_spacing, y_spacing) / 2
        grid_noise = min(grid_noise, max_noise)
        if grid_noise != kwargs.get("noise", 0.0):
            logger.warning(f"Grid noise reduced from {kwargs.get('noise', 0.0)} to {grid_noise} to prevent overlap")
        
        landmarks_pos = []
        for i in range(rows):
            for j in range(cols):
                if len(landmarks_pos) >= num:
                    break
                
                x = -x_lim + (j + 1) * x_spacing
                y = -y_lim + (i + 1) * y_spacing
                
                # Add noise if specified
                if grid_noise > 0:
                    x += rng.normal(0, grid_noise)
                    y += rng.normal(0, grid_noise)
                
                landmarks_pos.append([x, y])
        
        return np.array(landmarks_pos[:num])
    
    elif distribution == "perimeter":
        # Landmarks around the perimeter
        angles = np.linspace(0, 2*np.pi, num, endpoint=False)
        radius = kwargs.get("radius", min(x_lim, y_lim) * 0.8)
        radius_noise = kwargs.get("radius_noise", 0.0)
        
        landmarks_pos = np.zeros((num, 2))
        for i, angle in enumerate(angles):
            r = radius
            if radius_noise > 0:
                r += rng.normal(0, radius_noise)
            
            landmarks_pos[i, 0] = r * np.cos(angle)
            landmarks_pos[i, 1] = r * np.sin(angle)
        
        return landmarks_pos
    
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

def paths(
    R: int = 2,
    T: int = 100,
    motion_type: str = "figure8",
    center: Tuple[float, float] = (0.0, 0.0),
    radius: float = 10.0,
    angular_velocity: float = 0.1,
    linear_velocity: float = 0.15,
    noise_std: Union[float, List[float], np.ndarray] = 0.0,
    robot_separation: float = 2.0,
    time_offset: bool = False,
    random_seed: int = 42,
    worker_id: int = 0,
    enforce_separation: bool = True,
    **kwargs
) -> np.ndarray:
    """
    ✅ 修复1,2: Generate robot trajectories with proper RNG and collision checking
    
    Args:
        R: Number of robots
        T: Number of time steps
        motion_type: Motion pattern ("circular", "linear", "figure8", "random", "spiral")
        center: Center point for motion patterns
        radius: Characteristic radius for motion
        angular_velocity: Angular velocity for rotational motions (rad/step)
        linear_velocity: Linear velocity for translational motions (m/step)
        noise_std: Noise standard deviation(s) [σx, σy, σθ]
        robot_separation: Minimum separation between robots
        time_offset: Whether to apply time offsets between robots
        random_seed: Random seed for reproducibility
        worker_id: Worker ID for multi-process safety
        enforce_separation: Whether to enforce minimum robot separation
        **kwargs: Motion-specific parameters
        
    Returns:
        Robot trajectories (R, T, 3) or (T, 3) for single robot
    """
    # Input validation
    if R <= 0 or T <= 0:
        raise ValueError("R and T must be positive")
    
    if radius <= 0:
        raise ValueError("Radius must be positive")
    
    # ✅ 修复1: 可复现的RNG
    rng = np.random.default_rng(random_seed + worker_id * 10000)
    
    # Initialize trajectory array
    trajectories = np.zeros((R, T, 3))  # [x, y, theta]
    cx, cy = center
    
    # Generate trajectories based on motion type
    if motion_type == "circular":
        _generate_circular_paths(trajectories, R, T, cx, cy, radius, 
                                angular_velocity, robot_separation, time_offset, rng)
    
    elif motion_type == "linear":
        _generate_linear_paths(trajectories, R, T, radius, linear_velocity, 
                              robot_separation, enforce_separation, rng, **kwargs)
    
    elif motion_type == "figure8":
        _generate_figure8_paths(trajectories, R, T, cx, cy, radius,
                               angular_velocity, robot_separation, time_offset, rng)
    
    elif motion_type == "random":
        _generate_random_paths(trajectories, R, T, radius, linear_velocity, 
                              robot_separation, enforce_separation, rng, **kwargs)
    
    elif motion_type == "spiral":
        _generate_spiral_paths(trajectories, R, T, cx, cy, radius,
                              angular_velocity, robot_separation, rng, **kwargs)
    
    else:
        raise ValueError(f"Unsupported motion type: {motion_type}")
    
    # Add noise if specified
    if np.any(np.asarray(noise_std) > 0):
        trajectories = _add_trajectory_noise(trajectories, noise_std, R, T, rng)
    
    # Wrap all angles to [-π, π]
    trajectories[:, :, 2] = wrap_angle(trajectories[:, :, 2])
    
    # ✅ 修复2: 强制执行机器人间距检查
    if enforce_separation and R > 1:
        trajectories = _enforce_robot_separation(trajectories, robot_separation, rng)
    
    # Return format: (T, 3) for single robot, (R, T, 3) for multiple robots
    if R == 1:
        return trajectories[0]
    
    return trajectories

def _generate_circular_paths(trajectories: np.ndarray, R: int, T: int,
                           cx: float, cy: float, radius: float,
                           angular_velocity: float, robot_separation: float,
                           time_offset: bool, rng: np.random.Generator):
    """Generate circular motion trajectories"""
    for r in range(R):
        # Robot-specific parameters
        robot_radius = radius * (1 + 0.1 * (r - R//2))  # Slight radius variation
        phase_offset = r * 2 * np.pi / R  # Equal angular spacing
        
        # Time offset for staggered motion
        if time_offset:
            t_offset = r * T // (R * 2)
        else:
            t_offset = 0
        
        for t in range(T):
            effective_time = t + t_offset
            angle = phase_offset + angular_velocity * effective_time
            
            # Position
            trajectories[r, t, 0] = cx + robot_radius * np.cos(angle)
            trajectories[r, t, 1] = cy + robot_radius * np.sin(angle)
            
            # Orientation (tangent to circle)
            trajectories[r, t, 2] = angle + np.pi/2

def _generate_linear_paths(trajectories: np.ndarray, R: int, T: int,
                          radius: float, linear_velocity: float,
                          robot_separation: float, enforce_separation: bool,
                          rng: np.random.Generator, **kwargs):
    """✅ 修复5: Generate linear motion trajectories with boundary and collision handling"""
    max_turn_angle = kwargs.get("max_turn_angle", math.pi/6)  # 30度最大转向
    boundary_margin = kwargs.get("boundary_margin", radius * 0.1)
    
    for r in range(R):
        # ✅ 修复5: 确保起始位置在边界内且满足最小间距
        max_attempts = 100
        for attempt in range(max_attempts):
            start_angle = rng.uniform(0, 2*np.pi)
            start_radius = rng.uniform(0, radius - boundary_margin)
            
            start_x = start_radius * np.cos(start_angle)
            start_y = start_radius * np.sin(start_angle)
            
            # 检查与其他机器人的间距
            if r == 0 or not enforce_separation:
                break
            
            valid_position = True
            for prev_r in range(r):
                dist = np.linalg.norm([start_x - trajectories[prev_r, 0, 0], 
                                     start_y - trajectories[prev_r, 0, 1]])
                if dist < robot_separation:
                    valid_position = False
                    break
            
            if valid_position:
                break
        
        # Movement direction
        direction = rng.uniform(0, 2*np.pi)
        dx = linear_velocity * np.cos(direction)
        dy = linear_velocity * np.sin(direction)
        
        x, y, theta = start_x, start_y, direction
        
        for t in range(T):
            # Store current pose
            trajectories[r, t, 0] = x
            trajectories[r, t, 1] = y
            trajectories[r, t, 2] = theta
            
            # ✅ 修复5: 边界检查和转向
            next_x = x + dx
            next_y = y + dy
            
            # 检查边界碰撞
            if (abs(next_x) > radius - boundary_margin or 
                abs(next_y) > radius - boundary_margin):
                # 反弹或随机转向
                theta += rng.uniform(-max_turn_angle, max_turn_angle)
                dx = linear_velocity * np.cos(theta)
                dy = linear_velocity * np.sin(theta)
            else:
                # 偶尔随机转向
                if rng.random() < 0.05:  # 5%概率转向
                    theta += rng.uniform(-max_turn_angle, max_turn_angle)
                    dx = linear_velocity * np.cos(theta)
                    dy = linear_velocity * np.sin(theta)
            
            # Move forward
            x += dx
            y += dy

def _generate_figure8_paths(trajectories: np.ndarray, R: int, T: int,
                           cx: float, cy: float, radius: float,
                           angular_velocity: float, robot_separation: float,
                           time_offset: bool, rng: np.random.Generator):
    """✅ 修复8,10: Generate figure-8 motion with proper period and continuous orientation"""
    
    # ✅ 修复8: 确保完整的8字周期
    # 完整8字需要4π弧度，自动调整角速度确保至少一个完整周期
    if T * angular_velocity < 4 * np.pi:
        adjusted_velocity = 4 * np.pi / T
        logger.info(f"Adjusted angular_velocity from {angular_velocity} to {adjusted_velocity} for complete figure-8")
        angular_velocity = adjusted_velocity
    
    for r in range(R):
        # Robot-specific parameters
        robot_radius = radius * (1 + 0.1 * (r - R//2))
        phase_offset = r * 2 * np.pi / R
        
        if time_offset:
            t_offset = r * T // (R * 2)
        else:
            t_offset = 0
        
        # Pre-compute all positions first
        positions = np.zeros((T, 2))
        for t in range(T):
            effective_time = t + t_offset
            # Parameter for figure-8 (0 to 4π for complete cycle)
            u = effective_time * angular_velocity + phase_offset
            
            # Parametric figure-8 equations
            positions[t, 0] = cx + robot_radius * np.sin(u)
            positions[t, 1] = cy + robot_radius * np.sin(2 * u)
        
        # ✅ 修复10: 改进的方向计算，确保连续性
        orientations = np.zeros(T)
        for t in range(T):
            if t == 0:
                # Forward difference
                if T > 1:
                    dx = positions[1, 0] - positions[0, 0]
                    dy = positions[1, 1] - positions[0, 1]
                else:
                    dx, dy = 1.0, 0.0  # Default direction
            elif t == T - 1:
                # Backward difference
                dx = positions[T-1, 0] - positions[T-2, 0]
                dy = positions[T-1, 1] - positions[T-2, 1]
            else:
                # Central difference (more accurate)
                dx = positions[t+1, 0] - positions[t-1, 0]
                dy = positions[t+1, 1] - positions[t-1, 1]
            
            # ✅ 修复10: 确保角度连续性
            current_angle = math.atan2(dy, dx)
            if t > 0:
                # 确保与前一个角度的连续性
                angle_diff = wrap_angle(current_angle - orientations[t-1])
                if abs(angle_diff) > np.pi/2:  # 大角度跳跃，可能是wrap问题
                    current_angle = orientations[t-1] + angle_diff
            
            orientations[t] = current_angle
            
            # Store position and orientation
            trajectories[r, t, 0] = positions[t, 0]
            trajectories[r, t, 1] = positions[t, 1]
            trajectories[r, t, 2] = orientations[t]

def _generate_random_paths(trajectories: np.ndarray, R: int, T: int,
                          radius: float, linear_velocity: float,
                          robot_separation: float, enforce_separation: bool,
                          rng: np.random.Generator, **kwargs):
    """✅ 修复6: Generate random walk trajectories with improved turning logic"""
    turn_frequency = kwargs.get("turn_frequency", 10)  # Steps between direction changes
    max_turn_angle = kwargs.get("max_turn_angle", math.pi/4)  # Maximum turn per step
    
    for r in range(R):
        # Initial conditions with separation enforcement
        for attempt in range(100):
            x = rng.uniform(-radius/2, radius/2)
            y = rng.uniform(-radius/2, radius/2)
            
            if r == 0 or not enforce_separation:
                break
                
            # Check separation from other robots
            valid_position = True
            for prev_r in range(r):
                dist = np.linalg.norm([x - trajectories[prev_r, 0, 0], 
                                     y - trajectories[prev_r, 0, 1]])
                if dist < robot_separation:
                    valid_position = False
                    break
            if valid_position:
                break
        
        theta = rng.uniform(0, 2*np.pi)
        
        for t in range(T):
            # ✅ 修复6: 先转向再移动，避免高频抖动
            if t % turn_frequency == 0 and t > 0:
                turn_angle = rng.uniform(-max_turn_angle, max_turn_angle)
                theta += turn_angle
            
            # Store current pose
            trajectories[r, t, 0] = x
            trajectories[r, t, 1] = y
            trajectories[r, t, 2] = theta
            
            # Move forward
            x += linear_velocity * np.cos(theta)
            y += linear_velocity * np.sin(theta)
            
            # Boundary reflection
            if abs(x) > radius:
                theta = np.pi - theta  # Reflect horizontally
                x = np.clip(x, -radius, radius)
            if abs(y) > radius:
                theta = -theta  # Reflect vertically
                y = np.clip(y, -radius, radius)

def _generate_spiral_paths(trajectories: np.ndarray, R: int, T: int,
                          cx: float, cy: float, radius: float,
                          angular_velocity: float, robot_separation: float,
                          rng: np.random.Generator, **kwargs):
    """✅ 修复7: Generate spiral motion trajectories with functional spiral_rate"""
    spiral_rate = kwargs.get("spiral_rate", radius / (T * 2))  # Rate of radius change
    inward = kwargs.get("inward", True)  # True for inward spiral, False for outward
    
    for r in range(R):
        phase_offset = r * 2 * np.pi / R
        
        for t in range(T):
            # ✅ 修复7: 使用spiral_rate参数
            if inward:
                current_radius = radius - spiral_rate * t
                current_radius = max(current_radius, 0.1)  # 最小半径
            else:
                current_radius = spiral_rate * t
                current_radius = min(current_radius, radius)  # 最大半径
            
            angle = phase_offset + angular_velocity * t
            
            # Position
            trajectories[r, t, 0] = cx + current_radius * np.cos(angle)
            trajectories[r, t, 1] = cy + current_radius * np.sin(angle)
            
            # Orientation (tangent to spiral)
            trajectories[r, t, 2] = angle + np.pi/2

def _add_trajectory_noise(trajectories: np.ndarray, noise_std: Union[float, List, np.ndarray],
                         R: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """✅ 修复9: Add noise with proper independence between robots"""
    noise_std_arr = np.asarray(noise_std, dtype=float)
    
    # Parse noise standard deviations
    if noise_std_arr.ndim == 0:  # Scalar
        pos_std = np.full(2, noise_std_arr.item())
        angle_std = noise_std_arr.item()
    elif noise_std_arr.size == 1:  # Single element array
        pos_std = np.full(2, noise_std_arr.item())
        angle_std = noise_std_arr.item()
    elif noise_std_arr.size == 2:  # [σx, σy]
        pos_std = noise_std_arr
        angle_std = 0.0
    elif noise_std_arr.size >= 3:  # [σx, σy, σθ]
        pos_std = noise_std_arr[:2]
        angle_std = noise_std_arr[2]
    else:
        raise ValueError("noise_std must have 1, 2, or 3 elements")
    
    # ✅ 修复9: 一次性生成所有噪声，确保独立性
    if np.any(pos_std > 0):
        x_noise = rng.normal(0, pos_std[0], size=(R, T))
        y_noise = rng.normal(0, pos_std[1], size=(R, T))
        trajectories[:, :, 0] += x_noise
        trajectories[:, :, 1] += y_noise
    
    # Add orientation noise
    if angle_std > 0:
        angle_noise = rng.normal(0, angle_std, size=(R, T))
        trajectories[:, :, 2] += angle_noise
    
    return trajectories

def _enforce_robot_separation(trajectories: np.ndarray, min_separation: float,
                             rng: np.random.Generator) -> np.ndarray:
    """✅ 修复2: Enforce minimum separation between robots"""
    R, T, _ = trajectories.shape
    max_iterations = 10
    
    for iteration in range(max_iterations):
        violations = 0
        
        for t in range(T):
            for i in range(R):
                for j in range(i+1, R):
                    dist = np.linalg.norm(trajectories[i, t, :2] - trajectories[j, t, :2])
                    
                    if dist < min_separation:
                        violations += 1
                        
                        # 计算推开方向
                        direction = trajectories[i, t, :2] - trajectories[j, t, :2]
                        if np.linalg.norm(direction) < 1e-6:
                            # 如果重叠，随机方向
                            direction = rng.uniform(-1, 1, 2)
                        
                        direction = direction / np.linalg.norm(direction)
                        
                        # 推开到最小距离
                        push_distance = (min_separation - dist) / 2
                        trajectories[i, t, :2] += direction * push_distance
                        trajectories[j, t, :2] -= direction * push_distance
        
        if violations == 0:
            break
        
        if iteration == max_iterations - 1:
            logger.warning(f"Could not resolve all separation violations after {max_iterations} iterations")
    
    return trajectories

# =====================================================================
# Convenience Functions
# =====================================================================

def create_single_robot_world(T: int = 50, motion_type: str = "figure8",
                             num_landmarks: int = 5, world_size: float = 15.0,
                             noise_level: float = 0.05, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a simple single-robot world
    
    Args:
        T: Number of time steps
        motion_type: Robot motion pattern
        num_landmarks: Number of landmarks
        world_size: World size limit
        noise_level: Noise level for trajectory
        random_seed: Random seed for reproducibility
        
    Returns:
        (trajectory, landmarks): Robot path and landmark positions
    """
    # Generate single robot trajectory
    trajectory = paths(
        R=1, T=T,
        motion_type=motion_type,
        radius=world_size * 0.6,
        noise_std=[noise_level, noise_level, noise_level * 0.1],
        random_seed=random_seed
    )
    
    # Generate landmarks
    landmark_positions = landmarks(
        num=num_landmarks,
        limits=world_size,
        distribution="uniform",
        random_seed=random_seed
    )
    
    return trajectory, landmark_positions

def create_multi_robot_world(R: int = 2, T: int = 50, motion_type: str = "figure8",
                            num_landmarks: int = 8, world_size: float = 20.0,
                            noise_level: float = 0.05, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a multi-robot world
    
    Args:
        R: Number of robots
        T: Number of time steps  
        motion_type: Robot motion pattern
        num_landmarks: Number of landmarks
        world_size: World size limit
        noise_level: Noise level for trajectories
        random_seed: Random seed for reproducibility
        
    Returns:
        (trajectories, landmarks): Robot paths and landmark positions
    """
    # Generate robot trajectories
    trajectories = paths(
        R=R, T=T,
        motion_type=motion_type,
        radius=world_size * 0.5,
        noise_std=[noise_level, noise_level, noise_level * 0.1],
        time_offset=True,  # Stagger robot motions
        random_seed=random_seed
    )
    
    # Generate landmarks with good coverage
    landmark_positions = landmarks(
        num=num_landmarks,
        limits=world_size,
        distribution="gaussian",
        sigma=[world_size * 0.4, world_size * 0.4],
        random_seed=random_seed
    )
    
    return trajectories, landmark_positions

def validate_world_consistency(trajectories: np.ndarray, landmarks: np.ndarray,
                              min_separation: float = 1.0,
                              min_landmark_separation: float = 0.5) -> Dict[str, Any]:
    """
    ✅ 修复11: Validate world consistency with configurable thresholds
    
    Args:
        trajectories: Robot trajectories
        landmarks: Landmark positions
        min_separation: Minimum required robot separation
        min_landmark_separation: Minimum required landmark separation
        
    Returns:
        Validation report dictionary
    """
    report = {
        "valid": True,
        "warnings": [],
        "statistics": {}
    }
    
    # Normalize trajectory format
    if trajectories.ndim == 2:
        trajectories = trajectories[None, ...]  # Add robot dimension
    
    R, T, _ = trajectories.shape
    L = landmarks.shape[0]
    
    # Check trajectory bounds
    pos_bounds = np.max(np.abs(trajectories[:, :, :2]))
    landmark_bounds = np.max(np.abs(landmarks))
    
    report["statistics"]["trajectory_bounds"] = pos_bounds
    report["statistics"]["landmark_bounds"] = landmark_bounds
    report["statistics"]["num_robots"] = R
    report["statistics"]["num_timesteps"] = T
    report["statistics"]["num_landmarks"] = L
    
    # Check minimum separation between robots
    if R > 1:
        min_robot_distance = float('inf')
        for t in range(T):
            for i in range(R):
                for j in range(i+1, R):
                    dist = np.linalg.norm(trajectories[i, t, :2] - trajectories[j, t, :2])
                    min_robot_distance = min(min_robot_distance, dist)
        
        report["statistics"]["min_robot_separation"] = min_robot_distance
        
        if min_robot_distance < min_separation:
            report["warnings"].append(f"Robots too close: {min_robot_distance:.2f} < {min_separation}")
    
    # ✅ 修复11: 可配置的landmark间距阈值
    min_landmark_distance = float('inf')
    for i in range(L):
        for j in range(i+1, L):
            dist = np.linalg.norm(landmarks[i] - landmarks[j])
            min_landmark_distance = min(min_landmark_distance, dist)
    
    report["statistics"]["min_landmark_separation"] = min_landmark_distance
    
    if min_landmark_distance < min_landmark_separation:
        report["warnings"].append(f"Landmarks too close: {min_landmark_distance:.2f} < {min_landmark_separation}")
    
    # Check for NaN or infinite values
    if not np.isfinite(trajectories).all():
        report["valid"] = False
        report["warnings"].append("Non-finite values in trajectories")
    
    if not np.isfinite(landmarks).all():
        report["valid"] = False
        report["warnings"].append("Non-finite values in landmarks")
    
    return report

# Export main functions
__all__ = [
    'landmarks', 'paths', 'wrap_angle',
    'create_single_robot_world', 'create_multi_robot_world',
    'validate_world_consistency'
]