#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Robot SLAM - factor-graph animation (依赖 sim/measurement.py)
保持原有可视化 / 动画全部功能
Run:  python sim/factor_graph_animation.py
"""
from __future__ import annotations
import os, sys, math, logging, copy
import numpy as np
import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# ---------- 环境 & 日志 ----------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("SLAMAnimation")

# ---------- 外部模块 ----------
from sim.world              import create_multi_robot_world, validate_world_consistency
from sim.measurement        import generate_multi_robot_measurements           
from sim.graph_build        import GBPGraphBuilder
from algorithm.frontend.factor_ut import (wrap_angle, PriorFactor, OdometryFactor, BearingRangeUTFactor, PoseToPoseUTFactor, LoopClosureFactor
)

# ----------------------------------------------------------------------
# Builder config dataclass  (与 GBPGraphBuilder 选项同步即可)
# ----------------------------------------------------------------------
@dataclass
class BuilderConfig:
    enable_inter_robot_observations: bool = True
    enable_pose_to_pose_factors:    bool = True
    ut_mode:                        str  = "auto"
    enable_true_vectorization:      bool = True
    enable_batch_processing:        bool = True
    enable_cache_coordination:      bool = True

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def se2_relative_pose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """T_ab = a⁻¹ ∘ b  (a,b = [x,y,θ])"""
    dx, dy = b[0]-a[0], b[1]-a[1]
    ca, sa = math.cos(a[2]), math.sin(a[2])
    dx_l =  ca*dx + sa*dy
    dy_l = -sa*dx + ca*dy
    return np.array([dx_l, dy_l, wrap_angle(b[2]-a[2])], dtype=float)

# ================================ 可视化 / 动画 ===============================
def visualize_factor_graph(factors, variables, title="Multi-Robot SLAM Factor Graph"):
    plt.figure(figsize=(12, 10))
    plt.title(title)
    plt.grid(True, ls="--", alpha=.7)

    pose_vars, lm_vars = {}, {}
    for k, v in variables.items():
        (pose_vars if k.startswith("x") else lm_vars)[k] = v

    # 地标
    plt.scatter([p[0] for p in lm_vars.values()], [p[1] for p in lm_vars.values()],
                c="green", s=100, marker="*", label="Landmarks")

    # 机器人轨迹
    colors = ["blue", "red", "purple", "orange", "cyan"]
    traj_by_robot: Dict[str, Dict[int, np.ndarray]] = {}
    for key, p in pose_vars.items():
        rid, t = key.split("_"); t = int(t)
        traj_by_robot.setdefault(rid, {})[t] = p

    for i, (rid, dic) in enumerate(traj_by_robot.items()):
        pts = np.array([dic[t] for t in sorted(dic)])
        segs = np.stack([pts[:-1, :2], pts[1:, :2]], axis=1)
        plt.gca().add_collection(LineCollection(segs, colors=colors[i%5], lw=1.5,
                                                label=f"Robot {rid}"))
        # 朝向箭头(每 5 帧)
        for t in sorted(dic)[::5]:
            p = dic[t]; dx, dy = .5*math.cos(p[2]), .5*math.sin(p[2])
            plt.arrow(p[0], p[1], dx, dy, head_width=.2, head_length=.2,
                      fc=colors[i%5], ec=colors[i%5], alpha=.7)

    # 因子
    odom, obs, inter = [], [], []
    for f in factors:
        if isinstance(f, OdometryFactor) and f.v1 in pose_vars and f.v2 in pose_vars:
            odom.append([(pose_vars[f.v1][0], pose_vars[f.v1][1]),
                         (pose_vars[f.v2][0], pose_vars[f.v2][1])])
        elif isinstance(f, BearingRangeUTFactor) and \
             f.pose_key in pose_vars and f.lm_key in lm_vars:
            obs.append ([(pose_vars[f.pose_key][0], pose_vars[f.pose_key][1]),
                         (lm_vars[f.lm_key][0],  lm_vars[f.lm_key][1])])
        elif isinstance(f, PoseToPoseUTFactor) and \
             f.pose1_key in pose_vars and f.pose2_key in pose_vars:
            inter.append([(pose_vars[f.pose1_key][0], pose_vars[f.pose1_key][1]),
                          (pose_vars[f.pose2_key][0], pose_vars[f.pose2_key][1])])

    if odom:  plt.gca().add_collection(LineCollection(odom,  colors="black",  lw=.8,  alpha=.3, label="Odometry"))
    if obs:   plt.gca().add_collection(LineCollection(obs,   colors="green",  lw=.5,  ls="--", alpha=.3, label="Obs"))
    if inter: plt.gca().add_collection(LineCollection(inter, colors="red",    lw=1.5, ls="--", alpha=.4, label="Inter-Robot"))

    plt.legend()
    plt.axis("equal")
    plt.tight_layout()

def create_slam_animation(tra, lms, meas, factors, variables,
                          max_range=8.0, fps=10):
    """与旧版本完全一致，仅把变量名对齐"""
    R, T = tra.shape[:2]
    # --- pre-index measurement per time ---
    meas_by_t = [[] for _ in range(T)]
    for m in meas:
        if m["type"] == "robot_lm":
            if m["time"] < T:
                meas_by_t[m["time"]].append(m)

    seen_lm: List[set[int]] = [set() for _ in range(T)]
    for t in range(T):
        if t: seen_lm[t] = seen_lm[t-1].copy()
        for m in meas_by_t[t]:
            seen_lm[t].add(m["id"])

    # ---------- figure ----------
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title("Multi-Robot SLAM Animation"); ax.grid(True, ls="--", alpha=.7)

    pose_vars, lm_vars = {}, {}
    for k, v in variables.items():
        (pose_vars if k.startswith("x") else lm_vars)[k] = v

    colors = ["blue", "red", "purple", "orange", "cyan"]
    robot_plots = [ax.plot([], [], "o-", color=colors[i%5], label=f"Robot {i}")[0]
                   for i in range(R)]
    arrows: list[list[Any]] = [[] for _ in range(R)]
    range_circles = [ax.add_patch(plt.Circle((0,0), max_range, color=colors[i%5],
                                             fill=False, ls=":", alpha=.2))
                     for i in range(R)]
    ax.scatter([p[0] for p in lms], [p[1] for p in lms], c="gray", s=50, alpha=.3)
    obs_scatter = ax.scatter([], [], c="green", s=100, marker="*")

    # factor collections (updated in each frame)
    odom_lc = LineCollection([], colors="k", lw=.8, alpha=.3); ax.add_collection(odom_lc)
    obs_lc  = LineCollection([], colors="green", ls="--", lw=.5, alpha=.3); ax.add_collection(obs_lc)
    ir_lc   = LineCollection([], colors="red",   ls="--", lw=1.5, alpha=.4); ax.add_collection(ir_lc)
    
    odom_lc.set_label("Odometry")
    obs_lc.set_label("Obs")
    ir_lc.set_label("Inter-Robot")
    
    handles = [odom_lc, obs_lc, ir_lc, *robot_plots]
    ax.legend(handles=handles, loc="best")


    txt_t  = ax.text(.02,.98,"", transform=ax.transAxes, va="top", fontsize=12)
    txt_st = ax.text(.02,.94,"", transform=ax.transAxes, va="top", fontsize=10)

    # factor time map
    f_t = {}
    for f in factors:
        if isinstance(f, OdometryFactor):
            f_t[f] = max(int(f.v1.split("_")[1]), int(f.v2.split("_")[1]))
        elif isinstance(f, BearingRangeUTFactor):
            f_t[f] = int(f.pose_key.split("_")[1])
        elif isinstance(f, PoseToPoseUTFactor):
            f_t[f] = max(int(f.pose1_key.split("_")[1]), int(f.pose2_key.split("_")[1]))

    # bounds
    all_x = [p[0] for p in lms] + list(tra[:,:,0].ravel())
    all_y = [p[1] for p in lms] + list(tra[:,:,1].ravel())
    pad = .1*max(max(all_x)-min(all_x), max(all_y)-min(all_y))
    ax.set_xlim(min(all_x)-pad, max(all_x)+pad)
    ax.set_ylim(min(all_y)-pad, max(all_y)+pad); ax.axis("equal")

    def update(frame:int):
        if frame>=T: return robot_plots
        txt_t.set_text(f"t = {frame}")
        txt_st.set_text(f"obs: {sum(len(meas_by_t[t]) for t in range(frame+1))}   "
                        f"LM seen: {len(seen_lm[frame])}")

        # robots & circles & arrows
        for r in range(R):
            robot_plots[r].set_data(tra[r,:frame+1,0], tra[r,:frame+1,1])
            range_circles[r].center = (tra[r,frame,0], tra[r,frame,1])
            # arrows
            for a in arrows[r]: a.remove()
            arrows[r].clear()
            dx, dy = .5*math.cos(tra[r,frame,2]), .5*math.sin(tra[r,frame,2])
            arrows[r].append(ax.arrow(tra[r,frame,0], tra[r,frame,1], dx, dy,
                                      head_width=.2, head_length=.2,
                                      fc=colors[r%5], ec=colors[r%5], alpha=.7))
        # landmark seen
        ids = list(seen_lm[frame])
        if ids:
            obs_scatter.set_offsets(np.c_[ [lms[i][0] for i in ids],
                                            [lms[i][1] for i in ids] ])
        else: obs_scatter.set_offsets(np.zeros((0,2)))

        # factors until now
        od, ob, ir = [], [], []
        for f, tf in f_t.items():
            if tf>frame: continue
            if isinstance(f, OdometryFactor):
                od.append([(pose_vars[f.v1][0], pose_vars[f.v1][1]),
                           (pose_vars[f.v2][0], pose_vars[f.v2][1])])
            elif isinstance(f, BearingRangeUTFactor):
                ob.append([(pose_vars[f.pose_key][0], pose_vars[f.pose_key][1]),
                           (lm_vars[f.lm_key][0],  lm_vars[f.lm_key][1])])
            elif isinstance(f, PoseToPoseUTFactor):
                ir.append([(pose_vars[f.pose1_key][0], pose_vars[f.pose1_key][1]),
                           (pose_vars[f.pose2_key][0], pose_vars[f.pose2_key][1])])

        odom_lc.set_segments(od); obs_lc.set_segments(ob); ir_lc.set_segments(ir)
        return robot_plots + [obs_scatter, odom_lc, obs_lc, ir_lc, txt_t, txt_st]

    ani = animation.FuncAnimation(fig, update, frames=T+10, interval=1000/fps, blit=False)
    plt.show()
    plt.tight_layout()
    return ani, fig
# ============================  主流程  =========================================
def run_factor_graph_test() -> bool:
    logger.info("🚀  Multi-Robot SLAM Factor-Graph Animation Test start")
    try:
        np.random.seed(42)

        # 1. 世界 -----------------------------------------------------------------
        tra, lms = create_multi_robot_world(R=2, T=50, motion_type="figure8",
                                            num_landmarks=15, world_size=20.0,
                                            noise_level=0.05)
        validate_world_consistency(tra, lms)

        # 2. 一次性生成量测 --------------------------------------------------------
        meas_cfg = {
            "max_landmark_range": 8.0,
            "bearing_noise_std":  0.05,
            "range_noise_std":    0.20,
            "landmark_detection_prob": 1.0,
            "robot_detection_prob":    1.0,
            "enable_inter_robot_measurements": True,
            "enable_loop_closure": True,
        }
        measurements, loop_closures= generate_multi_robot_measurements(tra, lms, meas_cfg)
        lm_meas = [m for m in measurements if m["type"]=="robot_lm"]
        ir_meas = [m for m in measurements if m["type"]=="inter_robot"]
        logger.info("  -> %d landmark obs, %d inter-robot obs",
                    len(lm_meas), len(ir_meas))

        # 3. 构建图 ---------------------------------------------------------------
        builder = GBPGraphBuilder(BuilderConfig().to_dict())
        factors, variables = builder.build(tra, lms, lm_meas, inter_robot_obs=ir_meas, loop_closures=loop_closures)
        logger.info("  -> graph: %d factors / %d variables", len(factors), len(variables))

        # 4. 可视化 ---------------------------------------------------------------
        visualize_factor_graph(factors, variables)
        ani, _ = create_slam_animation(tra, lms, measurements, factors, variables, max_range=8.0)
        logger.info("🎉  Test finished OK")
        return True

    except Exception as exc:
        logger.exception("❌  Test failed")
        return False

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(0 if run_factor_graph_test() else 1)
