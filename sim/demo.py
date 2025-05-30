#!/usr/bin/env python
# coding: utf-8
"""
Gaussian / Sigma-Point Belief Propagation   Demo
author : ChatGPT (refactored)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import logging

# ---------- 你的框架 ----------
from sim.world import paths, landmarks
from sim.measurement import make_measurements
from sim.graph_build import GBPGraphBuilder
from algorithm.frontend.gbp import GBPGraph
from algorithm.frontend.factor_ut import BearingRangeUTFactor
# ------------------------------

# ----------- 日志 -------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger("SLAM-Demo")


# =========================================================
# 工具：协方差椭圆
# =========================================================
def plot_cov_ellipse(ax, mean, cov, n_std=2, **kw):
    """mean: (2,), cov: (2,2)"""
    if cov is None or cov.size == 0:
        return
    try:
        vals, vecs = np.linalg.eigh(cov[:2, :2])
        vals = np.maximum(vals, 0)
        width, height = 2 * n_std * np.sqrt(vals)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        e = Ellipse(mean, width, height, angle=angle, **kw)
        ax.add_patch(e)
    except Exception as exc:
        log.debug(f"ellipse fail: {exc}")


# =========================================================
# 主函数
# =========================================================
def main():

    # ---------- 0) 生成仿真数据 ----------
    np.random.seed(0)
    LMS  = landmarks(5, distribution="gaussian", sigma=[7, 7])
    PATH = paths     (2, 50, motion_type="figure8",
                      random_radius=True, noise_std=0.0)

    MEAS = make_measurements(
        PATH, LMS,
        p_obs=0.6,                  # ← 覆盖率 60 %
        sigma_r=0.3,
        sigma_b=np.deg2rad(2),
        max_range=25.0,
        seed=0)

    log.info(f"Landmarks={len(LMS)}, Poses={PATH.shape[1]}, "
             f"Measurements={len(MEAS)}")

    # ---------- 1) 构建因子图 ----------
    builder = GBPGraphBuilder({
        "ut_mode": "auto",
        "ut_alpha": 0.15,
        "ut_kappa": 0.0,
        "odom_sigma": 0.15
    })
    factors, variables = builder.build(PATH, LMS, MEAS)
    log.info(f"Graph built: {len(factors)} factors, "
             f"{len(variables)} variables")

    # ---------- 2) 初始化 GBP ----------
    graph = GBPGraph(damping=0.8)                # 较大阻尼更稳

    for key, init in variables.items():
        if key == "x0_0":
            graph.add_variable(key, prior=init, weight=1e6)   # 强先验
        else:
            graph.add_variable(key, prior=init, weight=1e-4)  # 极弱先验

    for f in factors:
        graph.add_factor(f)

    # ---------- 3) 求解 ----------
    log.info("Optimising ...")
    energy_log = graph.solve(n_iters=50, verbose=True)
    log.info(f"Energy final = {energy_log[-1]:.3e}")

    # ---------- 4) 结果 ----------
    traj_est = np.array([graph.get_mean(f"x0_{t}")
                         for t in range(PATH.shape[1])])
    lms_est  = np.array([graph.get_mean(f"l_{i}")
                         for i in range(LMS.shape[0])])

    rmse = np.sqrt(np.mean(np.sum((PATH[0] - traj_est)**2, axis=1)))
    log.info(f"Trajectory RMSE = {rmse:.3f} m")

    # mode 统计
    gbp_cnt = sum(isinstance(f, BearingRangeUTFactor) and
                  getattr(f, "_current_mode", "gbp") == "gbp"
                  for f in graph.factors)
    spbp_cnt = sum(isinstance(f, BearingRangeUTFactor) and
                   getattr(f, "_current_mode", "gbp") == "spbp"
                   for f in graph.factors)

    # ---------- 5) 可视化 ----------
    plt.figure(figsize=(14, 5))

    # (a) 轨迹
    ax1 = plt.subplot(131)
    ax1.scatter(LMS[:, 0], LMS[:, 1],
                c='k', marker='*', s=120, label='Landmark GT')
    ax1.scatter(lms_est[:, 0], lms_est[:, 1],
                c='r', marker='x', s=80,  label='Landmark EST')
    for i in range(len(LMS)):
        plot_cov_ellipse(ax1, lms_est[i],
                         graph.get_cov(f"l_{i}"), color='r', alpha=0.2)

    ax1.plot(PATH[0, :, 0], PATH[0, :, 1],
             '--b', label='Robot GT')
    ax1.plot(traj_est[:, 0], traj_est[:, 1],
             '-r', lw=2, label='Robot EST')
    ax1.scatter(traj_est[0, 0], traj_est[0, 1],
                s=100, c='g', marker='o', label='Start')

    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.legend()
    ax1.set_title("Trajectory & Landmarks")

    # (b) 能量曲线
    plt.subplot(132)
    plt.plot(energy_log, 'o-')
    plt.yscale('log')
    plt.title(f"Convergence (RMSE={rmse:.3f} m)")
    plt.xlabel("Iteration")
    plt.ylabel("Total Energy")
    plt.grid(True)

    # (c) 模式
    plt.subplot(133)
    plt.bar(["GBP", "SPBP"], [gbp_cnt, spbp_cnt],
            color=['blue', 'red'])
    plt.ylabel("Count")
    plt.title("Factor Mode Distribution")

    plt.tight_layout()
    plt.savefig("slam_result.png")
    log.info("Figure saved →  slam_result.png")
    plt.show()


# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback, sys
        traceback.print_exc(file=sys.stderr)
