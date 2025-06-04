"""
Generate ground-truth world:
  • landmarks() → ndarray (L,2)
  • paths(R,T)  → ndarray (R,T,3)  # 修改为包含(x,y,θ)的3D位姿
"""
import numpy as np
import math
from typing import Union, List, Tuple  # 添加缺失的导入

def landmarks(
    n: int = 2, 
    lim: float = 20.0, 
    distribution: str = "uniform",
    **kwargs  
) -> np.ndarray:
    """支持多种分布的地标生成"""
    if distribution == "uniform":
        return np.random.uniform(-lim, lim, size=(n, 2))
    elif distribution == "gaussian":
        mu = kwargs.get("mu", [0.0, 0.0])
        sigma = kwargs.get("sigma", [lim/3, lim/3])
        return np.random.normal(mu, sigma, size=(n, 2))
    elif distribution == "cluster":
        clusters = np.random.uniform(-lim, lim, size=(3, 2))
        assignments = np.random.randint(0, 3, size=n)
        points = np.zeros((n, 2))
        
        for i in range(3):
            mask = assignments == i
            cluster_size = np.sum(mask)
            if cluster_size > 0:
                points[mask] = clusters[i] + np.random.normal(0, lim/10, size=(cluster_size, 2))
        return points
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

def paths(
    R: int = 2,
    T: int = 100,
    motion_type: str = "circular",
    center: Tuple[float, float] = (0.0, 0.0),
    radius: float = 15.0,
    angular_velocity: float = 0.1,
    linear_velocity: float = 0.15,
    noise_std: Union[float, List[float]] = 0.0,
) -> np.ndarray:
    """生成包含位置(x,y)和朝向(θ)的3D轨迹"""
    if R < 1 or T < 1:
        raise ValueError("R and T must be at least 1")
        
    paths = np.zeros((R, T, 3))  # (R, T, 3) 格式: [x, y, θ]
    
    # ========== 运动类型处理 ==========
    cx, cy = center  # 解构中心点坐标
    
    if motion_type == "circular":
        for r in range(R):
            # 为每个机器人生成单独参数
            r_radius = radius * (1 + (np.random.rand() - 0.5) * 0.4)  # ±20% 变化
            r_omega = angular_velocity * (1 + (np.random.rand() - 0.5) * 0.4)
            phase = r * 2 * np.pi / R  # 相位偏移
            
            for t in range(T):
                angle = phase + r_omega * t
                
                # 位置
                paths[r, t, 0] = cx + r_radius * np.cos(angle)
                paths[r, t, 1] = cy + r_radius * np.sin(angle)
                
                # 朝向 - 切线方向
                paths[r, t, 2] = angle + np.pi/2
    
    elif motion_type == "linear":
        for r in range(R):
            # 随机起点和方向
            start_x = np.random.uniform(-radius, radius)
            start_y = np.random.uniform(-radius, radius)
            direction = np.random.uniform(0, 2*np.pi)
            dx = np.cos(direction) * linear_velocity
            dy = np.sin(direction) * linear_velocity
            
            for t in range(T):
                # 位置
                paths[r, t, 0] = start_x + t * dx
                paths[r, t, 1] = start_y + t * dy
                
                # 朝向 - 运动方向
                paths[r, t, 2] = direction
    
    elif motion_type == "figure8":
        for r in range(R):
            # 为每个机器人生成单独参数
            r_radius = radius * (1 + (np.random.rand() - 0.5) * 0.4)
            phase = r * np.pi / R  # 相位偏移
            
            for t in range(T):
                # 参数化时间 (0-4π)
                u = t / T * 4 * np.pi
                
                # 位置
                paths[r, t, 0] = cx + r_radius * np.sin(u + phase)
                paths[r, t, 1] = cy + r_radius * np.sin(2*(u + phase))
                
                # 朝向 - 数值微分计算切线方向
                if t == 0 and T > 1:
                    # 计算下一帧位置
                    next_u = (t+1) / T * 4 * np.pi
                    dx = cx + r_radius * np.sin(next_u + phase)
                    dy = cy + r_radius * np.sin(2*(next_u + phase))
                    paths[r, t, 2] = np.arctan2(
                        dy - paths[r, t, 1],
                        dx - paths[r, t, 0]
                    )
                elif t == T-1:
                    # 直接使用前一帧的方向
                    paths[r, t, 2] = paths[r, t-1, 2]
                else:
                    # 计算下一帧位置
                    next_u = (t+1) / T * 4 * np.pi
                    dx = cx + r_radius * np.sin(next_u + phase)
                    dy = cy + r_radius * np.sin(2*(next_u + phase))
                    paths[r, t, 2] = np.arctan2(
                        dy - paths[r, t, 1],
                        dx - paths[r, t, 0]
                    )
    
    elif motion_type == "random":
        for r in range(R):
            # 初始位置和方向
            x, y = np.random.uniform(-radius, radius, size=2)
            theta = np.random.uniform(0, 2*np.pi)
            
            for t in range(T):
                # 移动
                dx = linear_velocity * np.cos(theta)
                dy = linear_velocity * np.sin(theta)
                x += dx
                y += dy
                
                # 更新位置和朝向
                paths[r, t, 0] = x
                paths[r, t, 1] = y
                paths[r, t, 2] = theta
                
                # 随机改变方向
                if t > 0 and t % 10 == 0:
                    theta += np.random.normal(0, 0.5)
    
    # ========== 噪声处理（支持向量输入） ==========
    
    # 统一噪声参数格式
    noise_std_arr = np.asarray(noise_std, dtype=float)  # 转成 ndarray
    add_noise = np.any(noise_std_arr > 0)               # 至少有一个分量非零就加噪声
    
    if add_noise:
        # 处理不同维度的噪声输入
        if noise_std_arr.ndim == 0:  # 标量输入
            pos_std = np.full(2, noise_std_arr.item())  # 位置噪声 std
            theta_std = noise_std_arr.item()            # 角度噪声 std
        else:
            # 支持给 [σx, σy, σθ] 或 [σx, σy]
            if noise_std_arr.size == 1:  # 长度1的数组
                pos_std = np.full(2, noise_std_arr.item())
                theta_std = noise_std_arr.item()
            elif noise_std_arr.size == 2:  # [σx, σy]
                pos_std = noise_std_arr
                theta_std = 0.0  # 角度不加噪声
            elif noise_std_arr.size >= 3:  # [σx, σy, σθ]
                pos_std = noise_std_arr[:2]
                theta_std = noise_std_arr[2]
            else:
                raise ValueError("noise_std must have length 1, 2, or 3")
                
        # 给位置(x,y)加噪声
        paths[:, :, 0] += np.random.normal(0, pos_std[0], size=(R, T))
        paths[:, :, 1] += np.random.normal(0, pos_std[1], size=(R, T))
        
        # 给朝向θ加噪声（如果提供非零标准差）
        if theta_std > 0:
            paths[:, :, 2] += np.random.normal(0, theta_std, size=(R, T))
    
    # 单机器人时移除第一维度
    if R == 1:
        return paths[0]
    return paths