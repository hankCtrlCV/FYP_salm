"""
Enhanced World Generation for Multi-Robot SLAM
- Robust trajectory generation with proper orientation handling
- Multiple distribution support for landmarks
- Consistent coordinate frame handling
- Full compatibility with measurement and graph building modules

Author: Enhanced for Multi-Robot SLAM
Date: 2025-06-18
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
    **kwargs
) -> np.ndarray:
    """
    Generate landmark positions with various distributions
    
    Args:
        num: Number of landmarks
        limits: Spatial limits (scalar for symmetric, tuple for [x_lim, y_lim])
        distribution: Distribution type ("uniform", "gaussian", "cluster", "grid")
        **kwargs: Distribution-specific parameters
        
    Returns:
        Landmark positions (num, 2)
    """
    if num <= 0:
        raise ValueError("Number of landmarks must be positive")
    
    # Handle limits parameter
    if isinstance(limits, (int, float)):
        x_lim, y_lim = float(limits), float(limits)
    else:
        x_lim, y_lim = float(limits[0]), float(limits[1])
    
    if distribution == "uniform":
        return np.random.uniform([-x_lim, -y_lim], [x_lim, y_lim], size=(num, 2))
    
    elif distribution == "gaussian":
        mu = kwargs.get("mu", [0.0, 0.0])
        sigma = kwargs.get("sigma", [x_lim/3, y_lim/3])
        
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)
        
        if mu.size == 1:
            mu = np.full(2, mu.item())
        if sigma.size == 1:
            sigma = np.full(2, sigma.item())
            
        return np.random.normal(mu, sigma, size=(num, 2))
    
    elif distribution == "cluster":
        num_clusters = kwargs.get("num_clusters", 3)
        cluster_std = kwargs.get("cluster_std", min(x_lim, y_lim) / 10)
        
        # Generate cluster centers
        cluster_centers = np.random.uniform(
            [-x_lim*0.8, -y_lim*0.8], 
            [x_lim*0.8, y_lim*0.8], 
            size=(num_clusters, 2)
        )
        
        # Assign landmarks to clusters
        cluster_assignments = np.random.randint(0, num_clusters, size=num)
        
        landmarks_pos = np.zeros((num, 2))
        for i in range(num_clusters):
            mask = cluster_assignments == i
            count = np.sum(mask)
            if count > 0:
                landmarks_pos[mask] = (
                    cluster_centers[i] + 
                    np.random.normal(0, cluster_std, size=(count, 2))
                )
        
        return landmarks_pos
    
    elif distribution == "grid":
        # Regular grid with optional noise
        grid_noise = kwargs.get("noise", 0.0)
        
        # Determine grid dimensions
        cols = int(np.ceil(np.sqrt(num)))
        rows = int(np.ceil(num / cols))
        
        # Generate grid points
        x_spacing = 2 * x_lim / (cols + 1)
        y_spacing = 2 * y_lim / (rows + 1)
        
        landmarks_pos = []
        for i in range(rows):
            for j in range(cols):
                if len(landmarks_pos) >= num:
                    break
                
                x = -x_lim + (j + 1) * x_spacing
                y = -y_lim + (i + 1) * y_spacing
                
                # Add noise if specified
                if grid_noise > 0:
                    x += np.random.normal(0, grid_noise)
                    y += np.random.normal(0, grid_noise)
                
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
                r += np.random.normal(0, radius_noise)
            
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
    **kwargs
) -> np.ndarray:
    """
    Generate robot trajectories with proper orientation handling
    
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
        **kwargs: Motion-specific parameters
        
    Returns:
        Robot trajectories (R, T, 3) or (T, 3) for single robot
    """
    # Input validation
    if R <= 0 or T <= 0:
        raise ValueError("R and T must be positive")
    
    if radius <= 0:
        raise ValueError("Radius must be positive")
    
    # Initialize trajectory array
    trajectories = np.zeros((R, T, 3))  # [x, y, theta]
    cx, cy = center
    
    # Generate trajectories based on motion type
    if motion_type == "circular":
        _generate_circular_paths(trajectories, R, T, cx, cy, radius, 
                                angular_velocity, robot_separation, time_offset)
    
    elif motion_type == "linear":
        _generate_linear_paths(trajectories, R, T, radius, linear_velocity, **kwargs)
    
    elif motion_type == "figure8":
        _generate_figure8_paths(trajectories, R, T, cx, cy, radius,
                               angular_velocity, robot_separation, time_offset)
    
    elif motion_type == "random":
        _generate_random_paths(trajectories, R, T, radius, linear_velocity, **kwargs)
    
    elif motion_type == "spiral":
        _generate_spiral_paths(trajectories, R, T, cx, cy, radius,
                              angular_velocity, robot_separation, **kwargs)
    
    else:
        raise ValueError(f"Unsupported motion type: {motion_type}")
    
    # Add noise if specified
    if np.any(np.asarray(noise_std) > 0):
        trajectories = _add_trajectory_noise(trajectories, noise_std, R, T)
    
    # Wrap all angles to [-π, π]
    trajectories[:, :, 2] = wrap_angle(trajectories[:, :, 2])
    
    # Return format: (T, 3) for single robot, (R, T, 3) for multiple robots
    if R == 1:
        return trajectories[0]
    
    return trajectories

def _generate_circular_paths(trajectories: np.ndarray, R: int, T: int,
                           cx: float, cy: float, radius: float,
                           angular_velocity: float, robot_separation: float,
                           time_offset: bool):
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
                          radius: float, linear_velocity: float, **kwargs):
    """Generate linear motion trajectories"""
    for r in range(R):
        # Random start position and direction
        start_angle = np.random.uniform(0, 2*np.pi)
        start_radius = np.random.uniform(0, radius)
        
        start_x = start_radius * np.cos(start_angle)
        start_y = start_radius * np.sin(start_angle)
        
        # Movement direction
        direction = np.random.uniform(0, 2*np.pi)
        dx = linear_velocity * np.cos(direction)
        dy = linear_velocity * np.sin(direction)
        
        for t in range(T):
            # Position
            trajectories[r, t, 0] = start_x + t * dx
            trajectories[r, t, 1] = start_y + t * dy
            
            # Orientation (direction of motion)
            trajectories[r, t, 2] = direction

def _generate_figure8_paths(trajectories: np.ndarray, R: int, T: int,
                           cx: float, cy: float, radius: float,
                           angular_velocity: float, robot_separation: float,
                           time_offset: bool):
    """Generate figure-8 motion trajectories with robust orientation calculation"""
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
        
        # Compute orientations using robust finite differences
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
            
            # Store position and orientation
            trajectories[r, t, 0] = positions[t, 0]
            trajectories[r, t, 1] = positions[t, 1]
            trajectories[r, t, 2] = math.atan2(dy, dx)

def _generate_random_paths(trajectories: np.ndarray, R: int, T: int,
                          radius: float, linear_velocity: float, **kwargs):
    """Generate random walk trajectories"""
    turn_frequency = kwargs.get("turn_frequency", 10)  # Steps between direction changes
    max_turn_angle = kwargs.get("max_turn_angle", math.pi/4)  # Maximum turn per step
    
    for r in range(R):
        # Initial conditions
        x = np.random.uniform(-radius/2, radius/2)
        y = np.random.uniform(-radius/2, radius/2)
        theta = np.random.uniform(0, 2*np.pi)
        
        for t in range(T):
            # Store current pose
            trajectories[r, t, 0] = x
            trajectories[r, t, 1] = y
            trajectories[r, t, 2] = theta
            
            # Move forward
            x += linear_velocity * np.cos(theta)
            y += linear_velocity * np.sin(theta)
            
            # Occasional random turns
            if t % turn_frequency == 0 and t > 0:
                turn_angle = np.random.uniform(-max_turn_angle, max_turn_angle)
                theta += turn_angle

def _generate_spiral_paths(trajectories: np.ndarray, R: int, T: int,
                          cx: float, cy: float, radius: float,
                          angular_velocity: float, robot_separation: float,
                          **kwargs):
    """Generate spiral motion trajectories"""
    spiral_rate = kwargs.get("spiral_rate", radius / (T * 2))  # Rate of radius change
    inward = kwargs.get("inward", True)  # True for inward spiral, False for outward
    
    for r in range(R):
        phase_offset = r * 2 * np.pi / R
        
        for t in range(T):
            # Spiral radius
            if inward:
                current_radius = radius * (1 - t / T)
            else:
                current_radius = radius * t / T
            
            angle = phase_offset + angular_velocity * t
            
            # Position
            trajectories[r, t, 0] = cx + current_radius * np.cos(angle)
            trajectories[r, t, 1] = cy + current_radius * np.sin(angle)
            
            # Orientation (tangent to spiral)
            trajectories[r, t, 2] = angle + np.pi/2

def _add_trajectory_noise(trajectories: np.ndarray, noise_std: Union[float, List, np.ndarray],
                         R: int, T: int) -> np.ndarray:
    """Add noise to trajectories with proper handling of different noise specifications"""
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
    
    # Add position noise
    if np.any(pos_std > 0):
        trajectories[:, :, 0] += np.random.normal(0, pos_std[0], size=(R, T))
        trajectories[:, :, 1] += np.random.normal(0, pos_std[1], size=(R, T))
    
    # Add orientation noise
    if angle_std > 0:
        trajectories[:, :, 2] += np.random.normal(0, angle_std, size=(R, T))
    
    return trajectories

# =====================================================================
# Convenience Functions
# =====================================================================

def create_single_robot_world(T: int = 50, motion_type: str = "figure8",
                             num_landmarks: int = 5, world_size: float = 15.0,
                             noise_level: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a simple single-robot world
    
    Args:
        T: Number of time steps
        motion_type: Robot motion pattern
        num_landmarks: Number of landmarks
        world_size: World size limit
        noise_level: Noise level for trajectory
        
    Returns:
        (trajectory, landmarks): Robot path and landmark positions
    """
    # Generate single robot trajectory
    trajectory = paths(
        R=1, T=T,
        motion_type=motion_type,
        radius=world_size * 0.6,
        noise_std=[noise_level, noise_level, noise_level * 0.1]
    )
    
    # Generate landmarks
    landmark_positions = landmarks(
        num=num_landmarks,
        limits=world_size,
        distribution="uniform"
    )
    
    return trajectory, landmark_positions

def create_multi_robot_world(R: int = 2, T: int = 50, motion_type: str = "figure8",
                            num_landmarks: int = 8, world_size: float = 20.0,
                            noise_level: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a multi-robot world
    
    Args:
        R: Number of robots
        T: Number of time steps  
        motion_type: Robot motion pattern
        num_landmarks: Number of landmarks
        world_size: World size limit
        noise_level: Noise level for trajectories
        
    Returns:
        (trajectories, landmarks): Robot paths and landmark positions
    """
    # Generate robot trajectories
    trajectories = paths(
        R=R, T=T,
        motion_type=motion_type,
        radius=world_size * 0.5,
        noise_std=[noise_level, noise_level, noise_level * 0.1],
        time_offset=True  # Stagger robot motions
    )
    
    # Generate landmarks with good coverage
    landmark_positions = landmarks(
        num=num_landmarks,
        limits=world_size,
        distribution="gaussian",
        sigma=[world_size * 0.4, world_size * 0.4]
    )
    
    return trajectories, landmark_positions

def validate_world_consistency(trajectories: np.ndarray, landmarks: np.ndarray,
                              min_separation: float = 1.0) -> Dict[str, Any]:
    """
    Validate world consistency and provide statistics
    
    Args:
        trajectories: Robot trajectories
        landmarks: Landmark positions
        min_separation: Minimum required separation
        
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
    
    # Check landmark separation
    min_landmark_distance = float('inf')
    for i in range(L):
        for j in range(i+1, L):
            dist = np.linalg.norm(landmarks[i] - landmarks[j])
            min_landmark_distance = min(min_landmark_distance, dist)
    
    report["statistics"]["min_landmark_separation"] = min_landmark_distance
    
    if min_landmark_distance < 0.5:
        report["warnings"].append(f"Landmarks too close: {min_landmark_distance:.2f}")
    
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