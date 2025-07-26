#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verify the whole GTSAM pipeline:
 world.py  ---------->  trajectories, landmarks
 measurement.py ---->  measurements, loop_closures
 build_gtsam_graph.py --> factor-graph & initial guess
 GTSAM optimizer ----> optimized Values
 RMSE report --------> quick sanity check
"""
import time, math, logging, sys, os
import numpy as np
import gtsam
import matplotlib.pyplot as plt
import argparse
import psutil
from pathlib import Path
from datetime import datetime

# ------------------------------------------------------------------------
# 保证可以 import 项目根目录下的模块 | Ensure modules from project root can be imported
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from sim.world import create_multi_robot_world
from sim.measurement import generate_multi_robot_measurements
from gtsam_local.build_gtsam_graph import build_gtsam_graph, run_gtsam_optimizer
from utils.cfg_loader import load_common  # 导入统一配置加载函数 | Import unified configuration loader

# ------------------------------------------------------------------------
# 配置日志 | Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("GTSAMTest")

# ------------------------------------------------------------------------
def parse_args():
    """解析命令行参数 | Parse command line arguments"""
    parser = argparse.ArgumentParser(description='GTSAM SLAM Runner')
    
    # 场景参数 | Scene parameters
    parser.add_argument('--robots', type=int, default=2, help='Number of robots')
    parser.add_argument('--steps', type=int, default=40, help='Number of timesteps')
    parser.add_argument('--landmarks', type=int, default=15, help='Number of landmarks')
    parser.add_argument('--world-size', type=float, default=20.0, help='World size in meters')
    parser.add_argument('--motion', type=str, default="figure8", choices=["figure8", "circle", "random"], 
                      help='Robot motion pattern')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # 测量参数 | Measurement parameters
    parser.add_argument('--max-range', type=float, default=8.0, help='Maximum landmark detection range')
    parser.add_argument('--bearing-noise', type=float, default=0.05, help='Bearing measurement noise std')
    parser.add_argument('--range-noise', type=float, default=0.20, help='Range measurement noise std')
    parser.add_argument('--noise-mult', type=float, default=1.0, help='Noise multiplier for stress testing')
    parser.add_argument('--no-interrobot', action='store_false', dest='interrobot', 
                      help='Disable inter-robot measurements')
    parser.add_argument('--enable-loop', action='store_true', help='Enable loop closures')
    
    # 优化器参数 | Optimizer parameters
    parser.add_argument('--optimizer', choices=['gn', 'lm', 'isam2'], default='gn', 
                      help='Optimizer type (GaussNewton, LevenbergMarquardt, iSAM2)')
    parser.add_argument('--max-iterations', type=int, default=100, help='Maximum optimization iterations')
    
    # 输出参数 | Output parameters
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--save-json', action='store_true', help='Save results to JSON')
    parser.add_argument('--save-plot', action='store_true', help='Save plot to file')
    parser.add_argument('--no-plot', action='store_false', dest='show_plot', help='Do not show plot')
    
    # 故障测试参数 | Fault testing parameters
    parser.add_argument('--drop-node-rate', type=float, default=0.0, 
                      help='Rate of randomly dropping nodes (0.0 = none)')
    
    parser.add_argument('--truth-init', action='store_true',
                        help='Use ground-truth poses / landmarks as initial guess')
    
    return parser.parse_args()

# ------------------------------------------------------------------------
def rmse(truth, est):
    """计算RMSE，truth & est 都是 (R,T,3) ndarray | Calculate RMSE, truth & est are both (R,T,3) ndarray"""
    diff = truth - est
    diff[:,:,2] = (diff[:,:,2] + math.pi) % (2*math.pi) - math.pi   # wrap yaw
    return np.sqrt(np.mean(diff**2, axis=(0,1)))

# ------------------------------------------------------------------------
def plot_results(tra, est, lms_gt, lms_est=None,
                 title="GTSAM SLAM", save_path=None):
    """
    tra      : (R,T,3) ground-truth trajectories
    est      : (R,T,3) estimated trajectories
    lms_gt   : (L,2)   ground-truth landmarks
    lms_est  : (L,2)   estimated landmarks  (可选 | optional)
    """
    plt.figure(figsize=(10, 8))
    plt.gca().set_aspect('equal', adjustable='box')

    # 1️⃣ 地标：真值 × | Landmarks: ground truth ×
    plt.scatter(lms_gt[:, 0], lms_gt[:, 1],
                c='black', marker='x', label='Landmarks GT', alpha=0.7)

    # 2️⃣ 地标：估计 + | Landmarks: estimates +
    if lms_est is not None:
        plt.scatter(lms_est[:, 0], lms_est[:, 1],
                    c='cyan', marker='+', label='Landmarks Est', alpha=0.9, zorder=3)
        # 虚线连 GT 与 Est（误差向量） | Dotted lines connecting GT and Est (error vectors)
        for (gx, gy), (ex, ey) in zip(lms_gt, lms_est):
            plt.plot([gx, ex], [gy, ey],
                     linestyle=':', color='gray', linewidth=0.6, alpha=0.6)

    # 3️⃣ 机器人轨迹 | Robot trajectories
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange']
    for r in range(len(tra)):
        c = colors[r % len(colors)]
        # ground-truth
        plt.plot(tra[r, :, 0], tra[r, :, 1],
                 color=c, linewidth=2, label=f'R{r} GT')
        # estimate
        plt.plot(est[r, :, 0], est[r, :, 1],
                 color=c, linewidth=1, linestyle='--', label=f'R{r} Est')
        # 起点/终点 | Start/end points
        plt.scatter(tra[r, 0, 0], tra[r, 0, 1], color=c, marker='o')
        plt.scatter(tra[r, -1, 0], tra[r, -1, 1], color=c, marker='s')

    plt.title(title)
    plt.xlabel('X (m)'); plt.ylabel('Y (m)')
    plt.grid(alpha=0.3); plt.legend(fontsize=8)
    plt.tight_layout()

    # if save_path:
    #     Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     logger.info(f"Result image saved as {save_path}")

    plt.show()


# ------------------------------------------------------------------------
def simulate_node_failure(measurements, drop_rate):
    """模拟节点失效，随机丢弃一部分测量 | Simulate node failure by randomly dropping measurements"""
    if drop_rate <= 0:
        return measurements
    
    # 设置随机种子以确保可重复性 | Set random seed to ensure reproducibility
    np.random.seed(42)
    
    # 计算要丢弃的测量数量 | Calculate number of measurements to drop
    n_drop = int(len(measurements) * drop_rate)
    if n_drop == 0:
        return measurements
    
    # 随机选择要保留的测量 | Randomly select measurements to keep
    keep_indices = np.random.choice(len(measurements), 
                                   len(measurements) - n_drop, 
                                   replace=False)
    
    # 返回保留的测量 | Return kept measurements
    logger.info(f"Dropped {n_drop} measurements ({drop_rate*100:.1f}%)")
    return [measurements[i] for i in keep_indices]

# ------------------------------------------------------------------------
def main():
    # 解析命令行参数 | Parse command line arguments
    args = parse_args()
    
    # 设置随机种子 | Set random seed
    np.random.seed(args.seed)
    
    # 开始内存监控 | Start memory monitoring
    process = psutil.Process(os.getpid())
    mem_start = process.memory_info().rss
    
    # 1. 造世界 & 量测 | Create world & measurements ----------------------------------------------------
    world_start = time.perf_counter()
    
    # 创建多机器人世界 | Create multi-robot world
    tra, lms = create_multi_robot_world(
        R=args.robots, 
        T=args.steps, 
        motion_type=args.motion,
        num_landmarks=args.landmarks, 
        world_size=args.world_size,
        noise_level=0.05
    )
    
    # 配置测量生成 | Configure measurement generation
    meas_cfg = dict(
        max_landmark_range = args.max_range,
        bearing_noise_std  = args.bearing_noise * args.noise_mult,
        range_noise_std    = args.range_noise * args.noise_mult,
        enable_inter_robot_measurements = args.interrobot,
        enable_loop_closure = args.enable_loop,
        landmark_detection_prob = 1.0,
        robot_detection_prob    = 1.0,
    )
    
    # 生成测量 | Generate measurements
    measurements, loop_closures = generate_multi_robot_measurements(tra, lms, meas_cfg)
    
    # 模拟节点失效（如果需要） | Simulate node failure (if needed)
    if args.drop_node_rate > 0:
        measurements = simulate_node_failure(measurements, args.drop_node_rate)
    
    world_time = time.perf_counter() - world_start
    
    logger.info(f"Generated world: {args.robots} robots, {args.steps} timesteps, {len(lms)} landmarks")
    logger.info(f"Generated measurements: {len(measurements)} measurements, {len(loop_closures) if loop_closures else 0} loop closures")
    logger.info(f"World generation time: {world_time:.3f} s")
    
    # 2. 调用 build_gtsam_graph() | Call build_gtsam_graph() ------------------------------------------
    logger.info("Building GTSAM factor graph...")
    graph, initial, key_map, build_stats = build_gtsam_graph(
        tra, lms,
        measurements   = measurements,
        loop_closures  = loop_closures,
        cfg            = load_common(),
        use_truth_init = args.truth_init  
    )
    
    logger.info(f"Graph size: {build_stats['num_factors']} factors / {build_stats['num_variables']} variables")
    logger.info(f"Graph build time: {build_stats['build_time']:.3f} s")
    logger.info(f"Communication: {build_stats['comm_bytes']/1024:.2f} KB")
    logger.info(f"Average degree: {build_stats['avg_degree']:.2f} factors/var")
    
    
    # ---------- 2a. 评估 *初始* 误差（在优化之前） | Evaluate *initial* error (before optimization) ----------------
    init_est = np.zeros_like(tra)
    for r in range(args.robots):
        for t in range(args.steps):
            k = key_map.get(f"x{r}_{t}")
            pose0 = initial.atPose2(k)
            init_est[r, t] = np.array([pose0.x(), pose0.y(), pose0.theta()])

    init_lms = np.zeros_like(lms)
    for lid in range(lms.shape[0]):
        k = key_map.get(f"l_{lid}")
        pt0 = initial.atPoint2(k)
        if hasattr(pt0, "x"):      # Point2
            init_lms[lid] = np.array([pt0.x(), pt0.y()])
        else:                      # ndarray
            init_lms[lid] = np.asarray(pt0)

    init_pose_rmse = rmse(tra, init_est)[:2].mean()          # xy 均值 | xy mean
    init_lm_rmse   = np.sqrt(np.mean((lms - init_lms) ** 2)) # 地标 | landmarks

    logger.info(f"Initial pose-RMSE     : {init_pose_rmse:.3f} m")
    logger.info(f"Initial landmark-RMSE : {init_lm_rmse:.3f} m")

    # 3. 运行优化 | Run optimization ----------------------------------------------------------
    # 优化器类型映射 | Optimizer type mapping
    optimizer_map = {
        'gn': 'GaussNewton',
        'lm': 'LevenbergMarquardt',
        'isam2': 'iSAM2'
    }
    
    # 获取完整的优化器名称 | Get full optimizer name
    optimizer_type = optimizer_map.get(args.optimizer, args.optimizer)
    
    logger.info(f"Running GTSAM optimizer ({args.optimizer})...")
    result, opt_stats = run_gtsam_optimizer(
        graph, initial, 
        optimizer_type=optimizer_type,
        max_iterations=args.max_iterations
    )
    
    logger.info(f"Optimization time: {opt_stats['opt_time']:.3f} s")
    logger.info(f"Iterations: {opt_stats['iterations']}")
    
    # 4. 把 result 还原成 (R,T,3) | Convert result back to (R,T,3) -------------------------------------------
    est = np.zeros_like(tra)
    for r in range(args.robots):
        for t in range(args.steps):
            key_str = f"x{r}_{t}"
            if key_str in key_map:
                k = key_map[key_str]
                pose = result.atPose2(k)
                est[r, t] = np.array([pose.x(), pose.y(), pose.theta()])

    # ------------------------------------------------------------------
    # 4b. 读取地标估计 (Point2 或 ndarray) 并计算 RMSE | Read landmark estimates (Point2 or ndarray) and calculate RMSE
    # ------------------------------------------------------------------
    est_lms = np.zeros_like(lms)
    for lid in range(lms.shape[0]):
        key_str = f"l_{lid}"
        if key_str in key_map:
            k = key_map[key_str]
            pt = result.atPoint2(k)          # 可能是 Point2，也可能是 ndarray | May be Point2 or ndarray
            if hasattr(pt, "x"):             # 老版本：Point2 | Old version: Point2
                est_lms[lid] = np.array([pt.x(), pt.y()])
            else:                            # 新包装：ndarray | New wrapper: ndarray
                est_lms[lid] = np.asarray(pt)
        else:
            logger.warning(f"Landmark key {key_str} not found in key_map")
    
    lm_rmse = np.sqrt(np.mean((lms - est_lms) ** 2))
    logger.info(f"Landmark RMSE: {lm_rmse:.3f} m")


    
    # 5. 计算 RMSE | Calculate RMSE ----------------------------------------------------------
    e_xyz = rmse(tra, est)
    logger.info(f"RMSE [x y theta] = {e_xyz[0]:.3f} m, {e_xyz[1]:.3f} m, {e_xyz[2]:.3f} rad")
    
    # 计算内存使用 | Calculate memory usage
    mem_end = process.memory_info().rss
    mem_used = (mem_end - mem_start) / (1024 * 1024)  # MB
    
    # 6. 可视化结果 | Visualize results ---------------------------------------------------------
    if args.show_plot:
        plot_title = f"GTSAM SLAM: {args.robots} Robots, {args.landmarks} Landmarks"
        
        # 如果需要保存图像 | If need to save image
        plot_path = None
        if args.save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = Path(args.output_dir) / f"gtsam_r{args.robots}_t{args.steps}_l{args.landmarks}_{timestamp}.png"
        
        plot_results(tra, est, lms, lms_est=est_lms,
                     title=plot_title, save_path=plot_path)

    
    # 7. 性能总结 | Performance summary -----------------------------------------------------------
    total_time = world_time + build_stats['build_time'] + opt_stats['opt_time']
    print("\n=== GTSAM Performance Metrics ===")
    print(f"Scenario size   :  {args.robots} robots × {args.steps} timesteps × {args.landmarks} landmarks")
    print(f"Number of factors: {build_stats['num_factors']}")
    print(f"Number of variables: {build_stats['num_variables']}")
    print(f"Average degree  : {build_stats['avg_degree']:.2f} factors/variable")
    print(f"Graph build time: {build_stats['build_time']:.3f} s")
    print(f"Optimisation time: {opt_stats['opt_time']:.3f} s  ({opt_stats['iterations']} iterations)")
    print(f"Total runtime   : {total_time:.3f} s")
    print(f"Memory usage    : {mem_used:.2f} MB")
    print(f"Communication   : {build_stats['comm_bytes']/1024:.2f} KB")
    print(f"Final position RMSE  : {(e_xyz[0] + e_xyz[1]) / 2:.3f} m")
    print(f"Initial position RMSE: {init_pose_rmse:.3f} m")
    print(f"Initial landmark RMSE: {init_lm_rmse:.3f} m")
    print(f"Final yaw RMSE       : {e_xyz[2]:.3f} rad")
    print(f"Final landmark RMSE  : {lm_rmse:.3f} m")
    print(f"Graph hash      : {build_stats['graph_hash']}")

    print("\n✅  GTSAM pipeline test finished OK\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())