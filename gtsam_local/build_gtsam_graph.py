# gtsam_local/build_gtsam_graph.py
"""Build a plain-vanilla GTSAM factor-graph for (multi-)robot SLAM.

Graph layout (2-D SE(2)):
    • PriorFactorPose2   —— 每个机器人第 0 帧
    • BetweenFactorPose2 —— 里程计 & inter-robot / loop closures
    • BearingRangeFactor2D —— 机器人-地标观测
"""

from __future__ import annotations
import math, logging, time, hashlib, sys
from typing import List, Dict, Tuple, Any, Sequence

import numpy as np
import gtsam

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# keys helpers
# ---------------------------------------------------------------------------
# 用 "字母 + 整数" (gtsam.symbol) 生成 32-bit Key
def pose_key(rid: int, t: int) -> int:
    """Pose x{rid}_{t} → Key."""
    return gtsam.symbol('x', rid * 10000 + t)  # 假设 t < 10000，提高扩展性
    
def landmark_key(lid: int) -> int:
    """Landmark l_{lid} → Key."""
    return gtsam.symbol('l', lid)

def diag_sigmas(sig: Sequence[float]) -> gtsam.noiseModel.Diagonal:
    """Convenience wrapper."""
    return gtsam.noiseModel.Diagonal.Sigmas(np.asarray(sig, float))

# ---------------------------------------------------------------------------
# 相对位姿助手（与 world.paths() 的实现保持一致）
# ---------------------------------------------------------------------------
def compute_relative_pose2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return [dx, dy, dtheta] such that  a ∘ (dx,dy,dth) = b  .
    a, b: [x, y, theta]
    """
    dx, dy = b[:2] - a[:2]
    ca, sa = math.cos(a[2]), math.sin(a[2])
    dx_l =  ca * dx + sa * dy
    dy_l = -sa * dx + ca * dy
    dth  = ((b[2] - a[2] + math.pi) % (2 * math.pi)) - math.pi
    return np.array([dx_l, dy_l, dth], float)

# ---------------------------------------------------------------------------
# 通信量估计 - 改进版，使用更精确的方法
# ---------------------------------------------------------------------------
def estimate_gtsam_comm_bytes(graph: gtsam.NonlinearFactorGraph, values: gtsam.Values) -> int:
    """
    估计集中式GTSAM的通信量（假设所有数据需传到中心服务器）
    
    Parameters
    ----------
    graph : gtsam.NonlinearFactorGraph
        因子图
    values : gtsam.Values
        初始值
        
    Returns
    -------
    int
        估计的通信字节数
    """
    # 因子传输量（每个因子的变量索引、噪声模型、测量值）
    factor_bytes = 0
    for i in range(graph.size()):
        factor = graph.at(i)
        factor_type = type(factor).__name__
        
        # 使用len()而不是.size()
        n_keys = len(factor.keys())
        
        # 估计因子大小：键数量 * 4字节 + 噪声模型 + 测量值
        if 'Prior' in factor_type:
            # PriorFactor: 键(4) + 测量值(24) + 噪声模型(24)
            factor_bytes += 4 + 24 + 24
        elif 'Between' in factor_type:
            # BetweenFactor: 2键(8) + 测量值(24) + 噪声模型(24)
            factor_bytes += 8 + 24 + 24
        elif 'BearingRange' in factor_type:
            # BearingRangeFactor: 2键(8) + 测量值(16) + 噪声模型(16)
            factor_bytes += 8 + 16 + 16
        else:
            # 默认估计: 键 + 测量值 + 噪声模型
            factor_bytes += n_keys * 4 + 16 + 16
    
    # 变量初值传输量
    values_bytes = values.size() * 24  # 每个变量平均24字节(Pose2=3*8字节)
    
    return factor_bytes + values_bytes

# ---------------------------------------------------------------------------
# 配置过滤工具函数 - 可以供GBP重用
# ---------------------------------------------------------------------------
def filter_config(config: Dict[str, Any], 
                  allowed_keys: set, 
                  defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    从配置中提取允许的键值，缺失项使用默认值
    
    Parameters
    ----------
    config : Dict[str, Any]
        输入配置
    allowed_keys : set
        允许的键集合
    defaults : Dict[str, Any]
        默认值
        
    Returns
    -------
    Dict[str, Any]
        过滤后的配置
    """
    return {k: config.get(k, defaults.get(k)) for k in allowed_keys}

# ---------------------------------------------------------------------------
# main builder
# ---------------------------------------------------------------------------
_ALLOWED_KEYS = {
    # prior / odom
    "prior_sigma_xy", "prior_sigma_theta",
    "odom_sigma_xy",  "odom_sigma_theta",
    # landmark / bearing-range
    "obs_sigma_bearing", "obs_sigma_range",
    # inter-robot
    "inter_robot_obs_sigma_xy", "inter_robot_obs_sigma_theta",
    # loop closure
    "loop_closure_sigma_xy", "loop_closure_sigma_theta",
    # 可以添加新的键，例如：
    # "odom_bias_sigma",
}

_DEFAULTS = dict(
    prior_sigma_xy      = 0.05,
    prior_sigma_theta   = 0.02,
    odom_sigma_xy       = 0.08,
    odom_sigma_theta    = 0.015,
    obs_sigma_bearing   = math.radians(3.0),
    obs_sigma_range     = 0.12,
    inter_robot_obs_sigma_xy   = 0.15,
    inter_robot_obs_sigma_theta= 0.03,
    loop_closure_sigma_xy      = 0.16,
    loop_closure_sigma_theta   = 0.03,
    # 同样添加新键的默认值
    # odom_bias_sigma    = 0.01,
)

def build_gtsam_graph(
        robot_trajectories: np.ndarray | List[np.ndarray],
        landmark_positions: np.ndarray,
        measurements:       List[Dict[str, Any]],
        *,
        loop_closures:  List[Dict[str, Any]] | None = None,
        cfg: Dict[str, Any] | None = None,
) -> Tuple[gtsam.NonlinearFactorGraph,
           gtsam.Values,
           Dict[str, int],
           Dict[str, Any]]:
    """
    Parameters
    ----------
    robot_trajectories :  (R,T,3)   or list of (T_i,3)
    landmark_positions :  (L,2)
    measurements       :  list of dicts (robot-lm / inter_robot)
    loop_closures      :  optional list[dict]
    cfg                :  *flat* dict from load_common(); will be filtered.

    Returns
    -------
    graph, initial_values, key_map(str → Key), stats
    """
    # 性能计时开始
    t_start = time.perf_counter()
    
    # -------------------------------------------------------------------
    # 0.   cfg 过滤 + 缺省
    # -------------------------------------------------------------------
    cfg_in = cfg or {}
    cfg_use = filter_config(cfg_in, _ALLOWED_KEYS, _DEFAULTS)

    # -------------------------------------------------------------------
    # 1.   标准化 trajectory 形状
    # -------------------------------------------------------------------
    if isinstance(robot_trajectories, np.ndarray):
        robot_paths = (
            [robot_trajectories] if robot_trajectories.ndim == 2
            else [robot_trajectories[i] for i in range(robot_trajectories.shape[0])]
        )
    elif isinstance(robot_trajectories, list):
        robot_paths = [np.asarray(p, float) for p in robot_trajectories]
    else:
        raise TypeError("robot_trajectories must be ndarray or list")
    R = len(robot_paths)

    # -------------------------------------------------------------------
    # 2.   graph / initial container
    # -------------------------------------------------------------------
    graph   = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    key_map: Dict[str, int] = {}
    
    # 如果需要暂时禁用inter-robot测量进行调试，取消下面的注释
    # measurements = [m for m in measurements if m.get("type") != "inter_robot"]

    # ---- noise models ----
    prior_noise = diag_sigmas([cfg_use["prior_sigma_xy"],
                               cfg_use["prior_sigma_xy"],
                               cfg_use["prior_sigma_theta"]])
    odom_noise  = diag_sigmas([cfg_use["odom_sigma_xy"],
                               cfg_use["odom_sigma_xy"],
                               cfg_use["odom_sigma_theta"]])
    obs_noise   = diag_sigmas([cfg_use["obs_sigma_bearing"],
                               cfg_use["obs_sigma_range"]])
    ir_noise    = diag_sigmas([cfg_use["inter_robot_obs_sigma_xy"],
                               cfg_use["inter_robot_obs_sigma_xy"],
                               cfg_use["inter_robot_obs_sigma_theta"]])
    lc_noise    = diag_sigmas([cfg_use["loop_closure_sigma_xy"],
                               cfg_use["loop_closure_sigma_xy"],
                               cfg_use["loop_closure_sigma_theta"]])

    # -------------------------------------------------------------------
    # 3.   插入 Pose 变量 + 先验
    # -------------------------------------------------------------------
    for rid, path in enumerate(robot_paths):
        for t, pose in enumerate(path):
            k = pose_key(rid, t)
            key_map[f"x{rid}_{t}"] = k
            initial.insert(k, gtsam.Pose2(*pose.tolist()))
        # prior on first pose
        graph.add(gtsam.PriorFactorPose2(
            pose_key(rid, 0),
            gtsam.Pose2(*path[0].tolist()),
            prior_noise))

        # odometry chain
        for t in range(len(path) - 1):
            delta = compute_relative_pose2(path[t], path[t+1])
            graph.add(gtsam.BetweenFactorPose2(
                pose_key(rid, t),
                pose_key(rid, t+1),
                gtsam.Pose2(*delta),
                odom_noise))

    # -------------------------------------------------------------------
    # 4.   Landmarks 变量
    # -------------------------------------------------------------------
    for lid, pt in enumerate(landmark_positions):
        k = landmark_key(lid)
        key_map[f"l_{lid}"] = k
        initial.insert(k, gtsam.Point2(*pt.tolist()))

    # -------------------------------------------------------------------
    # 5.   Measurements
    # -------------------------------------------------------------------
    for m in measurements:
        mtype = m.get("type")
        if mtype == "robot_lm":
            rid = m["robot"];   t = m["time"];   lid = m["id"]
            bearing, rng = m["bearing_range"]
            graph.add(gtsam.BearingRangeFactor2D(
                pose_key(rid, t),
                landmark_key(lid),
                gtsam.Rot2.fromAngle(float(bearing)),
                float(rng),
                obs_noise))
        elif mtype == "inter_robot":
            r1 = m["observer_robot"]; t1 = m["observer_time"]
            r2 = m["observed_robot"]; t2 = m["observed_time"]
            br, rng = m["bearing_range"]
            
            # 尝试使用专用BearingRangeFactor (如果GTSAM版本支持)
            bearing_range_factor = getattr(gtsam, 'BearingRangeFactorPose2', 
                               getattr(gtsam, 'BearingRangeFactor2D', None))
            
            if bearing_range_factor is not None:
                # 最严谨的方法：直接使用bearing-range因子
                try:
                    graph.add(bearing_range_factor(
                        pose_key(r1, t1),
                        pose_key(r2, t2),
                        gtsam.Rot2.fromAngle(float(br)),
                        float(rng),
                        obs_noise))  # 使用与landmark观测相同的噪声模型
                    continue  # 成功添加，跳过下面的降级处理
                except Exception as e:
                    log.warning(f"Failed to use BearingRangeFactor: {e}, falling back to BetweenFactor")
            
            # 降级方案：改进的BetweenFactor实现
            # 获取观测者和目标的真实位姿
            observer_pose = robot_paths[r1][t1]
            target_pose = robot_paths[r2][t2]
            
            # 计算精确的相对位姿
            rel = gtsam.Pose2(*compute_relative_pose2(observer_pose, target_pose))
            
            # 创建一个调整后的噪声模型，降低角度Jacobian权重，而不是直接放大噪声
            # 创建一个更合理的噪声模型
            sigma_xy = cfg_use["inter_robot_obs_sigma_xy"]
            sigma_theta = cfg_use["inter_robot_obs_sigma_theta"]
            
            # 方法1：直接放大角度噪声
            # adjusted_noise = diag_sigmas([sigma_xy, sigma_xy, sigma_theta * 3.0])
            
            # 方法2：使用Information矩阵调整权重（更精确的方法）
            # 生成一个信息矩阵，降低角度分量的权重
            info_matrix = np.diag([1/(sigma_xy**2), 1/(sigma_xy**2), 1/(sigma_theta**2)/9.0])
            adjusted_noise = gtsam.noiseModel.Gaussian.Information(info_matrix)
            
            graph.add(gtsam.BetweenFactorPose2(
                pose_key(r1, t1), 
                pose_key(r2, t2),
                rel,
                adjusted_noise))
        else:
            log.debug("skip unknown measurement type %s", mtype)

    # -------------------------------------------------------------------
    # 6.   Loop closures (BetweenFactorPose2)
    # -------------------------------------------------------------------
    if loop_closures:
        for lc in loop_closures:
            k1 = pose_key(lc["robot1"], lc["time1"])
            k2 = pose_key(lc["robot2"], lc["time2"])
            rel = gtsam.Pose2(*lc["relative_pose"])
            graph.add(gtsam.BetweenFactorPose2(k1, k2, rel, lc_noise))

    # -------------------------------------------------------------------
    # 7.   性能统计
    # -------------------------------------------------------------------
    build_time = time.perf_counter() - t_start
    comm_bytes = estimate_gtsam_comm_bytes(graph, initial)
    
    # 计算平均度数（每个变量的平均因子数）
    avg_degree = float(graph.size()) / max(1, initial.size())
    
    # 创建一个简单的图哈希值，用于结果校验
    graph_hash = ""
    try:
        # 简单计算因子数和变量数的哈希
        hash_input = f"{graph.size()}_{initial.size()}_{len(measurements)}"
        graph_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    except:
        pass
    
    stats = {
        "build_time": build_time,
        "comm_bytes": comm_bytes,
        "num_factors": graph.size(),
        "num_variables": initial.size(),
        "avg_degree": avg_degree,
        "graph_hash": graph_hash,
    }

    log.info("✔ GTSAM graph built:  %d factors  /  %d variables  /  %.3f s",
             graph.size(), initial.size(), build_time)
    log.info("  Estimated communication: %.2f KB", comm_bytes / 1024)
    log.info("  Average degree: %.2f factors/var", avg_degree)
    
    return graph, initial, key_map, stats


# ---------------------------------------------------------------------------
# 优化器包装函数，添加时间计量和迭代次数
# ---------------------------------------------------------------------------
def run_gtsam_optimizer(graph: gtsam.NonlinearFactorGraph, 
                        initial: gtsam.Values, 
                        optimizer_type: str = "GaussNewton",
                        max_iterations: int = 100) -> Tuple[gtsam.Values, Dict[str, Any]]:
    """
    运行GTSAM优化器并计时
    
    Parameters
    ----------
    graph : gtsam.NonlinearFactorGraph
        因子图
    initial : gtsam.Values
        初始值
    optimizer_type : str, optional
        优化器类型，可选 "GaussNewton", "LevenbergMarquardt", "iSAM2"
    max_iterations : int, optional
        最大迭代次数，默认100
        
    Returns
    -------
    result : gtsam.Values
        优化结果
    stats : Dict[str, Any]
        性能统计
    """
    t_start = time.perf_counter()
    iterations = 0
    
    if optimizer_type == "GaussNewton":
        params = gtsam.GaussNewtonParams()
        params.setMaxIterations(max_iterations)
        params.setVerbosity('ERROR')  # 可选: SILENT, ERROR, VALUES
        optimizer = gtsam.GaussNewtonOptimizer(graph, initial, params)
        result = optimizer.optimize()
        iterations = optimizer.iterations()
    elif optimizer_type == "LevenbergMarquardt":
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(max_iterations)
        params.setVerbosity('ERROR')
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        result = optimizer.optimize()
        iterations = optimizer.iterations()
    elif optimizer_type == "iSAM2":
        params = gtsam.ISAM2Params()
        isam = gtsam.ISAM2(params)
        isam.update(graph, initial)
        result = isam.calculateEstimate()
        iterations = 1  # 对于iSAM2，这表示更新次数，而不是内部迭代
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    opt_time = time.perf_counter() - t_start
    
    stats = {
        "opt_time": opt_time,
        "iterations": iterations,
        "optimizer": optimizer_type
    }
    
    log.info("✔ GTSAM optimization complete:  %.3f s  /  %d iterations",
             opt_time, iterations)
    
    return result, stats