import numpy as np
import math
from typing import List, Tuple, Dict, Any, Optional

def wrap_angle(theta: float) -> float:
    """将角度归一化到 [-π, π] 范围"""
    while theta > math.pi:
        theta -= 2 * math.pi
    while theta < -math.pi:
        theta += 2 * math.pi
    return theta

def range_bearing(a_xy: np.ndarray, b_xy: np.ndarray) -> Tuple[float, float]:
    """理想量测 (a → b) : 距离 + 方位角（[-π, π]）"""
    dx, dy = b_xy - a_xy
    rng = np.hypot(dx, dy)
    brg = np.arctan2(dy, dx)  # 全局方位角
    return rng, brg

def make_measurements(
    paths: np.ndarray,      # (R, T, 3)  [x,y,theta]
    lms: np.ndarray,       # (L, 2)
    *,
    sigma_r: float = 0.30,
    sigma_b: float = math.radians(2.0),
    max_range_lm: float = 25.0,
    max_range_rr: float = 20.0,
    p_obs_lm: float = 1.0,
    p_obs_rr: float = 0.5,
    noise: bool = True,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:  # 修改返回类型为字典列表
    # 基础检查
    if paths.ndim != 3 or paths.shape[2] != 3:
        raise ValueError("paths must be (R, T, 3) [x,y,theta]")
    if lms.ndim != 2 or lms.shape[1] != 2:
        raise ValueError("lms must be (L, 2)")
    
    R, T, _ = paths.shape
    rng = np.random.default_rng(seed)
    
    # 初始化空列表存储测量结果
    measurements = []
    
    # Robot → Landmark 观测
    for r in range(R):
        for t in range(T):
            p_r = paths[r, t, :2]
            theta_r = paths[r, t, 2]  # 机器人朝向θ
            
            # 计算距离和可见性
            delta = lms - p_r
            dists = np.linalg.norm(delta, axis=1)
            visible = np.where(dists < max_range_lm)[0]
            
            if visible.size == 0:
                continue
                
            # 概率检测
            keep = rng.random(visible.size) < p_obs_lm
            kept = visible[keep]
            
            if kept.size == 0:
                continue
                
            # 计算全局方位角
            dx = delta[kept, 0]
            dy = delta[kept, 1]
            global_brg = np.arctan2(dy, dx)
            
            # 计算相对方位角 = 全局方位角 - 机器人朝向θ
            true_brg = global_brg - theta_r
            
            # 添加噪声
            for idx, lm_id in enumerate(kept):
                if noise:
                    meas_rng = dists[lm_id] + rng.normal(0, sigma_r)
                    meas_brg = true_brg[idx] + rng.normal(0, sigma_b)
                else:
                    meas_rng = dists[lm_id]
                    meas_brg = true_brg[idx]
                
                # 角度归一化
                meas_brg = wrap_angle(meas_brg)
                
                # 创建测量字典
                measurements.append({
                    "type": "robot_lm",
                    "robot": r,
                    "time": t,
                    "id": int(lm_id),
                    "bearing_range": np.array([meas_rng, meas_brg], dtype=float)
                })

    # Robot ↔ Robot 观测
    for r_i in range(R):
        for t in range(T):
            p_i = paths[r_i, t, :2]
            theta_i = paths[r_i, t, 2]  # 观测机器人朝向θ
            
            for r_j in range(R):
                if r_j == r_i:
                    continue
                    
                p_j = paths[r_j, t, :2]
                
                # 计算全局方位角
                rng_ij, brg_ij = range_bearing(p_i, p_j)
                
                # 距离检查
                if rng_ij >= max_range_rr or rng.random() >= p_obs_rr:
                    continue
                
                # 计算相对方位角 = 全局方位角 - 观测者θ_i
                brg_rel = brg_ij - theta_i
                
                # 添加噪声
                if noise:
                    rng_meas = rng_ij + rng.normal(0, sigma_r)
                    brg_meas = brg_rel + rng.normal(0, sigma_b)
                else:
                    rng_meas, brg_meas = rng_ij, brg_rel
                
                # 角度归一化
                brg_meas = wrap_angle(brg_meas)
                
                # 创建测量字典
                measurements.append({
                    "type": "robot_robot",
                    "from_robot": r_i,
                    "from_time": t,
                    "to_robot": r_j,
                    "to_time": t,  # 同一时间
                    "bearing_range": np.array([rng_meas, brg_meas], dtype=float)
                })

    # 共视地标观测 (不需要专门的测量，因为已经通过robot_lm测量捕获)
    # 只需在demo.py中使用测量数据构建因子图即可
    
    return measurements  # 返回字典列表