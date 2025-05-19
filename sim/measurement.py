import numpy as np

def range_bearing(robot_xy: np.ndarray, lm_xy: np.ndarray) -> tuple:
    """计算机器人与地标的距离和方位角（真实值）"""
    delta = lm_xy - robot_xy
    rng = np.linalg.norm(delta)
    bearing = np.arctan2(delta[1], delta[0])
    return rng, bearing

def make_measurements(
    paths: np.ndarray, 
    lms: np.ndarray, 
    sigma_r: float = 0.3, 
    sigma_b: float = np.deg2rad(2),
    max_range: float = 25.0,
    noise_enabled: int = 1,  # 新增参数：0关闭噪声，1开启噪声
    seed: int = None
) -> list:
    """
    参数说明新增:
        noise_enabled: 0=关闭噪声（返回真实值），1=开启噪声（默认）
    """
    # 输入验证
    assert noise_enabled in (0, 1), "noise_enabled 必须是 0 或 1"
    
    if seed is not None:
        np.random.seed(seed)
    
    R, T, _ = paths.shape
    measurements = []
    
    for robot_idx in range(R):
        robot_meas = []
        for t in range(T):
            robot_pos = paths[robot_idx, t, :]
            deltas = lms - robot_pos
            distances = np.linalg.norm(deltas, axis=1)
            visible_mask = distances < max_range
            visible_indices = np.where(visible_mask)[0]
            
            if not visible_indices.size:
                robot_meas.append([])
                continue
                
            # 计算真实测量值
            bearings = np.arctan2(deltas[visible_mask, 1], deltas[visible_mask, 0])
            true_distances = distances[visible_mask]
            
            # 根据噪声开关添加噪声
            if noise_enabled:
                noisy_distances = true_distances + np.random.randn(len(visible_indices)) * sigma_r
                noisy_bearings = bearings + np.random.randn(len(visible_indices)) * sigma_b
            else:
                noisy_distances = true_distances
                noisy_bearings = bearings
            
            obs = list(zip(
                visible_indices.tolist(),
                np.column_stack([noisy_distances, noisy_bearings])
            ))
            robot_meas.append(obs)
        
        measurements.append(robot_meas)
    
    return measurements