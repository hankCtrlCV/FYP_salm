#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal *stable* centralized GBP-SLAM demo
-----------------------------------------
• 1 robot, figure-8 trajectory (SE(2))
• random 2-D landmarks
• bearing-range measurements
"""

from __future__ import annotations
import math, argparse, logging, sys, time
from typing import Dict, Any, List

import numpy as np

# ─── external modules ──────────────────────────────────────
from sim.world import paths, landmarks
from algorithm.frontend.graph_build import GBPGraphBuilder, export_to_gbp
from algorithm.frontend.gbp          import GBPGraph

# ─── logging ───────────────────────────────────────────────
import logging
import sys

# PATCH BEGIN: 双通道日志输出（终端+文件）
logfile = "sim/slam_demo_full.log"   # 可以按需自定义路径

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)5s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(logfile, mode="w", encoding="utf-8"),  # 输出到文件
        logging.StreamHandler(sys.stdout)                          # 输出到终端
    ]
)
log = logging.getLogger("DEMO")
logging.getLogger("GaBP").setLevel(logging.INFO)   # 控制迭代内部日志等级
# PATCH END


wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi      # angle wrap


# ─── measurement generator ─────────────────────────────────
def make_meas(traj: np.ndarray,
              lms : np.ndarray,
              rng : np.random.Generator,
              max_r : float = 8.0,
              p_det : float = .9,
              σ_r  : float = .2,                      # → 更大噪声 ⇒ 信息矩阵更稳
              σ_b  : float = math.radians(2.0)         #   ~2°
             ) -> List[Dict[str,Any]]:
    """Generate noisy bearing-range measurements."""
    R, T, _ = traj.shape
    out: List[Dict[str,Any]] = []
    for r in range(R):
        for t in range(T):
            pose = traj[r, t]
            dxy  = lms - pose[:2]
            dist = np.linalg.norm(dxy, axis=1)
            vis  = (dist < max_r) & (rng.random(len(dist)) < p_det)
            for i in np.where(vis)[0]:
                bearing = wrap(math.atan2(dxy[i,1], dxy[i,0]) - pose[2]
                               + rng.normal(0, σ_b))
                rng_dist = max(.1, dist[i] + rng.normal(0, σ_r))
                out.append({
                    "robot"         : r,
                    "time"          : t,
                    "id"            : i,
                    "bearing_range" : np.array([bearing, rng_dist]),
                })
    log.info("measurements: %d", len(out))
    return out


# ─── main ─────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-T", type=int, default=30, help="trajectory length")
    ap.add_argument("-L", type=int, default=8,  help="# landmarks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iter", type=int, default=80, help="GBP max iterations")
    ap.add_argument("--tol",  type=float, default=3e-5, help="convergence tol")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)

    # 1) world --------------------------------------------------------
    lm  = landmarks(num=args.L, distribution="gaussian", sigma=[7, 7])
    trj = paths(R=1, T=args.T, motion_type="figure8",
                noise_std=np.array([.05, .05, .015]))

    if trj.ndim == 2:                       # shape (T,3) → (1,T,3)
        trj = trj[None, ...]

    # 2) measurements -------------------------------------------------
    meas = make_meas(trj, lm, rng)

    # 3) factor-graph builder cfg ------------------------------------
    
    cfg = {
        # --- 观测 / 里程计 / 先验 σ ---
        "obs_sigma_bearing": math.radians(2.0),   # 4°
        "obs_sigma_range"  : 0.20,                # 20 cm
        "odom_sigma_xy"    : 0.05,                # 5 cm
        "odom_sigma_theta" : math.radians(3.0),   # 3°
        "prior_sigma_xy"   : 0.10,                # 10 cm
        "prior_sigma_theta": math.radians(4.0),   # 4°

        # --- 数值稳健参数 ---
        "numerical_config": {
            "regularization":        1e-2,   # 0.02
            "min_eigenvalue":        1e-3,
            "max_condition_number":  1e6,
            "jacobian_clip_threshold": 2e5,
        },

        "ut_mode": "gbp",
        "enable_strict_validation": False,
        "cache_warmup": True,
        "clear_caches_on_init": True,
    }

    builder = GBPGraphBuilder(cfg)
    builder.build(trj, lm, meas)            # 单机器人，无环闭

    # 4) 导出给求解器 ＋ 强锚点先验 -------------------------------
    variables, factors, priors, angle_map = export_to_gbp(builder)


    # 5) 运行 GBP（两级阻尼尝试） -------------------------------
    damping_schemes = [(0.85, 0.75), (0.80, 0.70)]
    for attempt, (bel_d, msg_d) in enumerate(damping_schemes, 1):
        g = GBPGraph(factors, variables, priors,
                     angle_vars=angle_map,
                     bel_damp=bel_d, msg_damp=msg_d)

        tic = time.perf_counter()
        hist, stats = g.run(max_iter=args.iter, tol=args.tol, verbose=True)
        toc = time.perf_counter()

        log.info("GBP: Δ=%.2e | %2d iters | converged=%s | %.3fs",
                 stats["final_delta"], stats["iterations"],
                 stats["converged"], toc - tic)

        if stats["converged"]:
            est = g.get_means()["x0_0"]
            log.info("x0_0 estimate: %s", np.round(est, 3))
            break
        elif attempt == len(damping_schemes):
            log.error("GBP failed to converge after %d attempts", attempt)

# entry-point ---------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
