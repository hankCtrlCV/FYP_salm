"""
Generate ground-truth world:
  • landmarks() → ndarray (L,2)
  • paths(R,T)  → ndarray (R,T,2)
"""
import numpy as np
import matplotlib.pyplot as plt

def landmarks(
    n: int = 2, 
    lim: float = 20.0, 
    distribution: str = "uniform",  # 新增参数：支持不同分布
    **kwargs  
) -> np.ndarray:
    """支持多种分布的地标生成"""
    if distribution == "uniform":
        return np.random.uniform(-lim, lim, size=(n, 2))
    elif distribution == "gaussian":
        mu = kwargs.get("mu", [0.0, 0.0])       # 默认均值在原点
        sigma = kwargs.get("sigma", [lim/3, lim/3])  # 3σ覆盖大部分区域
        return np.random.normal(mu, sigma, size=(n, 2))
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

def paths(
    R: int = 2,
    T: int = 100,
    radius: float = 15.0, 
    omega: float = 0.1,
    motion_type: str = "circular",  # 新增参数：支持不同运动模式
    random_radius: bool = False,     # 不同机器人轨迹半径随机化
    noise_std: float = 0.0,         # 轨迹噪声
    **kwargs
) -> np.ndarray:
    """生成多样化机器人轨迹"""
    paths = np.zeros((R, T, 2))
    
    # 为每个机器人生成独立半径
    radii = radius * (1 + np.random.randn(R) * 0.2) if random_radius else radius
    
    for r in range(R):
        if motion_type == "circular":
            theta = omega * np.arange(T)
            x = radii[r] * np.cos(theta)
            y = radii[r] * np.sin(theta)
        elif motion_type == "linear":
            x = np.linspace(0, 2*np.pi*radii[r], T)
            y = np.zeros_like(x)
        elif motion_type == "figure8":
            t = np.linspace(0, 2*np.pi, T)
            x = radii[r] * np.sin(t)
            y = radii[r] * np.sin(2*t)
        else:
            raise ValueError(f"Unsupported motion type: {motion_type}")
        
        # 添加噪声
        paths[r, :, 0] = x + np.random.normal(0, noise_std, T)
        paths[r, :, 1] = y + np.random.normal(0, noise_std, T)
    
    return paths

