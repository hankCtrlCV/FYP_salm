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
# 保证可以 import 项目根目录下的模块
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from sim.world import create_multi_robot_world
from sim.measurement import generate_multi_robot_measurements
from gtsam_local.build_gtsam_graph import build_gtsam_graph, run_gtsam_optimizer
from utils.cfg_loader import load_common  # 导入统一配置加载函数

# ------------------------------------------------------------------------
# 配置日志
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("GTSAMTest")

# ------------------------------------------------------------------------
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='GTSAM SLAM Runner')
    
    # 场景参数
    parser.add_argument('--robots', type=int, default=2, help='Number of robots')
    parser.add_argument('--steps', type=int, default=40, help='Number of timesteps')
    parser.add_argument('--landmarks', type=int, default=15, help='Number of landmarks')
    parser.add_argument('--world-size', type=float, default=20.0, help='World size in meters')
    parser.add_argument('--motion', type=str, default="figure8", choices=["figure8", "circle", "random"], 
                      help='Robot motion pattern')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # 测量参数
    parser.add_argument('--max-range', type=float, default=8.0, help='Maximum landmark detection range')
    parser.add_argument('--bearing-noise', type=float, default=0.05, help='Bearing measurement noise std')
    parser.add_argument('--range-noise', type=float, default=0.20, help='Range measurement noise std')
    parser.add_argument('--noise-mult', type=float, default=1.0, help='Noise multiplier for stress testing')
    parser.add_argument('--no-interrobot', action='store_false', dest='interrobot', 
                      help='Disable inter-robot measurements')
    parser.add_argument('--enable-loop', action='store_true', help='Enable loop closures')
    
    # 优化器参数
    parser.add_argument('--optimizer', choices=['gn', 'lm', 'isam2'], default='gn', 
                      help='Optimizer type (GaussNewton, LevenbergMarquardt, iSAM2)')
    parser.add_argument('--max-iterations', type=int, default=100, help='Maximum optimization iterations')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--save-json', action='store_true', help='Save results to JSON')
    parser.add_argument('--save-plot', action='store_true', help='Save plot to file')
    parser.add_argument('--no-plot', action='store_false', dest='show_plot', help='Do not show plot')
    
    # 故障测试参数
    parser.add_argument('--drop-node-rate', type=float, default=0.0, 
                      help='Rate of randomly dropping nodes (0.0 = none)')
    
    return parser.parse_args()

# ------------------------------------------------------------------------
def rmse(truth, est):
    """计算RMSE，truth & est 都是 (R,T,3) ndarray"""
    diff = truth - est
    diff[:,:,2] = (diff[:,:,2] + math.pi) % (2*math.pi) - math.pi   # wrap yaw
    return np.sqrt(np.mean(diff**2, axis=(0,1)))

# ------------------------------------------------------------------------
def plot_results(tra, est, lms, title="GTSAM SLAM", save_path=None):
    """绘制真实轨迹、估计轨迹和地标位置"""
    plt.figure(figsize=(10, 8))
    
    # 绘制地标
    plt.scatter(lms[:, 0], lms[:, 1], c='black', marker='x', label='Landmarks', alpha=0.6)
    
    # 为每个机器人使用不同颜色
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    # 绘制真实轨迹和估计轨迹
    for r in range(len(tra)):
        color = colors[r % len(colors)]
        
        # 真实轨迹
        plt.plot(tra[r, :, 0], tra[r, :, 1], color=color, linestyle='-', 
                 linewidth=2, label=f'Robot {r} Ground Truth')
        
        # 估计轨迹
        plt.plot(est[r, :, 0], est[r, :, 1], color=color, linestyle='--', 
                 linewidth=1.5, label=f'Robot {r} Estimate')
        
        # 起点和终点标记
        plt.scatter(tra[r, 0, 0], tra[r, 0, 1], color=color, marker='o')
        plt.scatter(tra[r, -1, 0], tra[r, -1, 1], color=color, marker='s')
    
    plt.title(title)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.axis('equal')
    plt.tight_layout()
    
    # 保存图像（如果提供了路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"结果图像已保存为 {save_path}")
    
    plt.show()

# ------------------------------------------------------------------------
def simulate_node_failure(measurements, drop_rate):
    """模拟节点失效，随机丢弃一部分测量"""
    if drop_rate <= 0:
        return measurements
    
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    # 计算要丢弃的测量数量
    n_drop = int(len(measurements) * drop_rate)
    if n_drop == 0:
        return measurements
    
    # 随机选择要保留的测量
    keep_indices = np.random.choice(len(measurements), 
                                   len(measurements) - n_drop, 
                                   replace=False)
    
    # 返回保留的测量
    logger.info(f"丢弃了 {n_drop} 个测量 ({drop_rate*100:.1f}%)")
    return [measurements[i] for i in keep_indices]

# ------------------------------------------------------------------------
def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 开始内存监控
    process = psutil.Process(os.getpid())
    mem_start = process.memory_info().rss
    
    # 1. 造世界 & 量测 ----------------------------------------------------
    world_start = time.perf_counter()
    
    # 创建多机器人世界
    tra, lms = create_multi_robot_world(
        R=args.robots, 
        T=args.steps, 
        motion_type=args.motion,
        num_landmarks=args.landmarks, 
        world_size=args.world_size,
        noise_level=0.05
    )
    
    # 配置测量生成
    meas_cfg = dict(
        max_landmark_range = args.max_range,
        bearing_noise_std  = args.bearing_noise * args.noise_mult,
        range_noise_std    = args.range_noise * args.noise_mult,
        enable_inter_robot_measurements = args.interrobot,
        enable_loop_closure = args.enable_loop,
        landmark_detection_prob = 1.0,
        robot_detection_prob    = 1.0,
    )
    
    # 生成测量
    measurements, loop_closures = generate_multi_robot_measurements(tra, lms, meas_cfg)
    
    # 模拟节点失效（如果需要）
    if args.drop_node_rate > 0:
        measurements = simulate_node_failure(measurements, args.drop_node_rate)
    
    world_time = time.perf_counter() - world_start
    
    logger.info(f"生成世界: {args.robots}个机器人, {args.steps}个时间步, {len(lms)}个地标")
    logger.info(f"生成测量: {len(measurements)}个测量, {len(loop_closures) if loop_closures else 0}个回环")
    logger.info(f"世界生成时间: {world_time:.3f} s")
    
    # 2. 调用 build_gtsam_graph() ------------------------------------------
    logger.info("构建GTSAM因子图...")
    graph, initial, key_map, build_stats = build_gtsam_graph(
        tra, lms,
        measurements   = measurements,
        loop_closures  = loop_closures,
        cfg            = load_common()
    )
    
    logger.info(f"图大小: {build_stats['num_factors']}因子 / {build_stats['num_variables']}变量")
    logger.info(f"构图时间: {build_stats['build_time']:.3f} s")
    logger.info(f"通信量: {build_stats['comm_bytes']/1024:.2f} KB")
    logger.info(f"平均度数: {build_stats['avg_degree']:.2f} factors/var")
    
    # 3. 运行优化 ----------------------------------------------------------
    # 优化器类型映射
    optimizer_map = {
        'gn': 'GaussNewton',
        'lm': 'LevenbergMarquardt',
        'isam2': 'iSAM2'
    }
    
    # 获取完整的优化器名称
    optimizer_type = optimizer_map.get(args.optimizer, args.optimizer)
    
    logger.info(f"运行GTSAM优化器 ({args.optimizer})...")
    result, opt_stats = run_gtsam_optimizer(
        graph, initial, 
        optimizer_type=optimizer_type,
        max_iterations=args.max_iterations
    )
    
    logger.info(f"优化时间: {opt_stats['opt_time']:.3f} s")
    logger.info(f"迭代次数: {opt_stats['iterations']}")
    
    # 4. 把 result 还原成 (R,T,3) -------------------------------------------
    est = np.zeros_like(tra)
    for r in range(args.robots):
        for t in range(args.steps):
            key_str = f"x{r}_{t}"
            if key_str in key_map:
                k = key_map[key_str]
                pose = result.atPose2(k)
                est[r,t] = np.array([pose.x(), pose.y(), pose.theta()])
    
    # 5. 计算 RMSE ----------------------------------------------------------
    e_xyz = rmse(tra, est)
    logger.info(f"RMSE [x y theta] = {e_xyz[0]:.3f} m, {e_xyz[1]:.3f} m, {e_xyz[2]:.3f} rad")
    
    # 计算内存使用
    mem_end = process.memory_info().rss
    mem_used = (mem_end - mem_start) / (1024 * 1024)  # MB
    
    # 6. 可视化结果 ---------------------------------------------------------
    if args.show_plot:
        plot_title = f"GTSAM SLAM: {args.robots} Robots, {args.landmarks} Landmarks"
        
        # 如果需要保存图像
        plot_path = None
        if args.save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = Path(args.output_dir) / f"gtsam_r{args.robots}_t{args.steps}_l{args.landmarks}_{timestamp}.png"
        
        plot_results(tra, est, lms, title=plot_title, save_path=plot_path)
    
    # 7. 性能总结 -----------------------------------------------------------
    total_time = world_time + build_stats['build_time'] + opt_stats['opt_time']
    
    print("\n=== GTSAM 性能指标 ===")
    print(f"场景规模:  {args.robots}机器人 × {args.steps}时间步 × {args.landmarks}地标")
    print(f"因子数量:  {build_stats['num_factors']}")
    print(f"变量数量:  {build_stats['num_variables']}")
    print(f"平均度数:  {build_stats['avg_degree']:.2f} factors/var")
    print(f"构图时间:  {build_stats['build_time']:.3f} s")
    print(f"优化时间:  {opt_stats['opt_time']:.3f} s ({opt_stats['iterations']}次迭代)")
    print(f"总时间:    {total_time:.3f} s")
    print(f"内存使用:  {mem_used:.2f} MB")
    print(f"通信量:    {build_stats['comm_bytes']/1024:.2f} KB")
    print(f"位置RMSE:  {(e_xyz[0] + e_xyz[1])/2:.3f} m")
    print(f"角度RMSE:  {e_xyz[2]:.3f} rad")
    print(f"图哈希:    {build_stats['graph_hash']}")
    
    print("\n✅  GTSAM pipeline test finished OK\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())