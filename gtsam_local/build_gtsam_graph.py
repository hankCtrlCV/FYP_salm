# === build_gtsam_graph.py (modified: supports noisy initial) ===
"""Build a plain‑vanilla GTSAM factor‑graph **with a switch to choose the type
of initial guess** (either the ground‑truth poses & landmarks, or a rough/noisy
estimate that mimics what you would have in a real SLAM front‑end).

Add a new flag:
    use_truth_init : bool  (default True)
        * True  → identical behaviour to the original script (perfect initial).
        * False → robot poses and landmarks are initialised with noise / first
          observation, so that the optimiser must really solve SLAM.
"""
from __future__ import annotations
import math, logging, time, hashlib
from typing import List, Dict, Any, Sequence

import numpy as np
import gtsam

log = logging.getLogger(__name__)

# -------------------------------------------------------------------
# 接收 YAML 中的旧键名 → 转成 Builder 需要的键名
_ALIAS_KEYS = {
    #   YAML 原键                     Builder 内部期望键
    "inter_robot_obs_sigma_range":   "inter_robot_obs_sigma_xy",
    "inter_robot_obs_sigma_bearing": "inter_robot_obs_sigma_theta",
    # 如今后还想支持其它命名，可继续补充 ...
}


# ---------------------------------------------------------------------------
# keys helpers ---------------------------------------------------------------

def pose_key(rid: int, t: int) -> int:
    """Pose x{rid}_{t} → Key."""
    return gtsam.symbol('x', rid * 10000 + t)

def landmark_key(lid: int) -> int:
    """Landmark l_{lid} → Key."""
    return gtsam.symbol('l', lid)

def diag_sigmas(sig: Sequence[float]) -> gtsam.noiseModel.Diagonal:
    return gtsam.noiseModel.Diagonal.Sigmas(np.asarray(sig, float))

# ---------------------------------------------------------------------------
# helpers --------------------------------------------------------------------

def compute_relative_pose2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dx, dy = b[:2] - a[:2]
    ca, sa = math.cos(a[2]), math.sin(a[2])
    dx_l =  ca * dx + sa * dy
    dy_l = -sa * dx + ca * dy
    dth  = ((b[2] - a[2] + math.pi) % (2 * math.pi)) - math.pi
    return np.array([dx_l, dy_l, dth], float)

# ---------------------------------------------------------------------------
# communication estimate -----------------------------------------------------

def estimate_gtsam_comm_bytes(graph: gtsam.NonlinearFactorGraph,
                              values: gtsam.Values) -> int:
    factor_bytes = 0
    for i in range(graph.size()):
        factor = graph.at(i)
        n_keys = len(factor.keys())
        if 'Prior' in type(factor).__name__:
            factor_bytes += 4 + 24 + 24
        elif 'Between' in type(factor).__name__:
            factor_bytes += 8 + 24 + 24
        elif 'BearingRange' in type(factor).__name__:
            factor_bytes += 8 + 16 + 16
        else:
            factor_bytes += n_keys * 4 + 16 + 16
    values_bytes = values.size() * 24
    return factor_bytes + values_bytes

# ---------------------------------------------------------------------------
_ALLOWED_KEYS = {"prior_sigma_xy","prior_sigma_theta","odom_sigma_xy","odom_sigma_theta",
                 "obs_sigma_bearing","obs_sigma_range",
                 "inter_robot_obs_sigma_xy","inter_robot_obs_sigma_theta",
                 "loop_closure_sigma_xy","loop_closure_sigma_theta"}
_DEFAULTS = dict(prior_sigma_xy=0.05,prior_sigma_theta=0.02,
                 odom_sigma_xy=0.08,odom_sigma_theta=0.015,
                 obs_sigma_bearing=math.radians(3),obs_sigma_range=0.12,
                 inter_robot_obs_sigma_xy=0.15,inter_robot_obs_sigma_theta=0.03,
                 loop_closure_sigma_xy=0.16,loop_closure_sigma_theta=0.03)

def filter_config(cfg: Dict[str,Any], allowed: set, defaults: Dict[str,Any]):
    return {k: cfg.get(k, defaults[k]) for k in allowed}

# ---------------------------------------------------------------------------
# main builder ----------------------------------------------------------------

def build_gtsam_graph(
        robot_trajectories: np.ndarray | List[np.ndarray],
        landmark_positions: np.ndarray,
        measurements:       List[Dict[str, Any]],
        *,
        loop_closures:  List[Dict[str, Any]] | None = None,
        cfg: Dict[str, Any] | None = None,
        use_truth_init: bool = False,
) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values, Dict[str,int], Dict[str,Any]]:
    """Build factor graph & *noisy* initial (if use_truth_init=False)."""

    t0 = time.perf_counter()
    cfg_raw = cfg or {}
    # -- 把别名注入 --
    for old,new in _ALIAS_KEYS.items():
        if old in cfg_raw and new not in cfg_raw:
            cfg_raw[new] = cfg_raw[old]

    cfg_use = filter_config(cfg_raw, _ALLOWED_KEYS, _DEFAULTS)

    # -------- standardise robot_paths --------
    if isinstance(robot_trajectories, np.ndarray):
        robot_paths = ([robot_trajectories] if robot_trajectories.ndim == 2
                       else [robot_trajectories[i] for i in range(robot_trajectories.shape[0])])
    else:
        robot_paths = [np.asarray(p,float) for p in robot_trajectories]
    R = len(robot_paths)

    # world extent for random landmark guesses
    all_xy = np.vstack([p[:,:2] for p in robot_paths])
    world_half = float(np.max(np.abs(all_xy))) + 1.0

    # outputs
    graph   = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    key_map: Dict[str,int] = {}

    # noise models
    prior_noise = diag_sigmas([cfg_use['prior_sigma_xy']]*2+[cfg_use['prior_sigma_theta']])
    odom_noise  = diag_sigmas([cfg_use['odom_sigma_xy']]*2+[cfg_use['odom_sigma_theta']])
    obs_noise   = diag_sigmas([cfg_use['obs_sigma_bearing'], cfg_use['obs_sigma_range']])
    ir_noise    = diag_sigmas([cfg_use['inter_robot_obs_sigma_xy']]*2+[cfg_use['inter_robot_obs_sigma_theta']])
    lc_noise    = diag_sigmas([cfg_use['loop_closure_sigma_xy']]*2+[cfg_use['loop_closure_sigma_theta']])

    # ------------------------------------------------------------------
    # 3. Pose variables & prior
    # ------------------------------------------------------------------
    for rid, path in enumerate(robot_paths):
        for t, pose_true in enumerate(path):
            k = pose_key(rid, t)
            key_map[f"x{rid}_{t}"] = k

            if use_truth_init:
                noisy = pose_true
            else:
                noisy = pose_true + np.random.normal(scale=[0.25,0.25,math.radians(15)], size=3)
            initial.insert(k, gtsam.Pose2(*noisy.tolist()))

        # true prior on first pose (so optimisation is observable)
        graph.add(gtsam.PriorFactorPose2(pose_key(rid,0), gtsam.Pose2(*path[0].tolist()), prior_noise))

        # odom chain factors (use true delta — this represents perfect wheel odometry)
        for t in range(len(path)-1):
            delta = compute_relative_pose2(path[t], path[t+1])
            graph.add(gtsam.BetweenFactorPose2(pose_key(rid,t), pose_key(rid,t+1),
                                               gtsam.Pose2(*delta), odom_noise))

    # ------------------------------------------------------------------
    # 4. Landmark variables (insert noisy guesses FIRST; we may refine with first obs below)
    # ------------------------------------------------------------------
    lm_guess: Dict[int,np.ndarray] = {lid: (landmark_positions[lid] if use_truth_init else
                                            np.random.uniform(-world_half, world_half, 2))
                                      for lid in range(landmark_positions.shape[0])}

    # revise guess using first robot‑lm observation if we are in noisy mode
    if not use_truth_init:
        seen: set[int] = set()
        for m in measurements:
            if m.get('type') == 'robot_lm':
                lid = m['id']
                if lid in seen:  # keep first only
                    continue
                rid,t = m['robot'], m['time']
                br, rng = m['bearing_range']
                pose_i = initial.atPose2(pose_key(rid,t))  # current noisy pose guess
                gx = pose_i.x() + rng*math.cos(pose_i.theta()+br)
                gy = pose_i.y() + rng*math.sin(pose_i.theta()+br)
                lm_guess[lid] = np.array([gx, gy])
                seen.add(lid)

    for lid, guess in lm_guess.items():
        k = landmark_key(lid)
        key_map[f"l_{lid}"] = k
        initial.insert(k, gtsam.Point2(*guess.tolist()))

    # ------------------------------------------------------------------
    # 5. Measurements (same as original)
    # ------------------------------------------------------------------
    for m in measurements:
        mtype = m.get('type')
        if mtype == 'robot_lm':
            rid,t,lid = m['robot'], m['time'], m['id']
            br,rng  = m['bearing_range']
            graph.add(gtsam.BearingRangeFactor2D(pose_key(rid,t), landmark_key(lid),
                                                 gtsam.Rot2.fromAngle(float(br)), float(rng), obs_noise))
        elif mtype == "inter_robot":
            # 始终用 BetweenFactorPose2，避免 BearingRange 模板类型冲突
            r1,t1 = m["observer_robot"], m["observer_time"]
            r2,t2 = m["observed_robot"], m["observed_time"]
            rel = gtsam.Pose2(*compute_relative_pose2(
                          robot_paths[r1][t1], robot_paths[r2][t2]))
            graph.add(gtsam.BetweenFactorPose2(pose_key(r1,t1),
                                               pose_key(r2,t2),
                                               rel, ir_noise))
        
    # loop closures
    if loop_closures:
        for lc in loop_closures:
            k1 = pose_key(lc['robot1'], lc['time1'])
            k2 = pose_key(lc['robot2'], lc['time2'])
            graph.add(gtsam.BetweenFactorPose2(k1,k2, gtsam.Pose2(*lc['relative_pose']), lc_noise))

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------
    stats = dict(build_time=time.perf_counter()-t0,
                 comm_bytes=estimate_gtsam_comm_bytes(graph,initial),
                 num_factors=graph.size(), num_variables=initial.size(),
                 avg_degree=float(graph.size())/max(1,initial.size()),
                 graph_hash=hashlib.md5(f"{graph.size()}_{initial.size()}".encode()).hexdigest()[:8])
    log.info("✔ GTSAM graph built (%s init): %d factors / %d vars / %.3fs",
             "truth" if use_truth_init else "noisy", graph.size(), initial.size(), stats['build_time'])
    return graph, initial, key_map, stats

# ---------------------------------------------------------------------------
# simple wrapper to run optimiser (unchanged) --------------------------------

def run_gtsam_optimizer(graph: gtsam.NonlinearFactorGraph, initial: gtsam.Values,
                        optimizer_type: str = "GaussNewton", max_iterations: int = 100):
    t0=time.perf_counter()
    if optimizer_type=="GaussNewton":
        params = gtsam.GaussNewtonParams(); params.setMaxIterations(max_iterations); params.setVerbosity("ERROR")
        opt = gtsam.GaussNewtonOptimizer(graph, initial, params); result = opt.optimize(); it = opt.iterations()
    elif optimizer_type=="LevenbergMarquardt":
        params = gtsam.LevenbergMarquardtParams(); params.setMaxIterations(max_iterations); params.setVerbosity("ERROR")
        opt = gtsam.LevenbergMarquardtOptimizer(graph, initial, params); result = opt.optimize(); it = opt.iterations()
    else:
        isam = gtsam.ISAM2(); isam.update(graph, initial); result=isam.calculateEstimate(); it=1
    return result, dict(opt_time=time.perf_counter()-t0, iterations=it, optimizer=optimizer_type)

# === gtsam_runner_noisy.py --------------------------------------------------
"""A thin runner that demonstrates *noisy‑initial* SLAM evaluation.
Run:
    python gtsam_noisy_init_demo.py  --robots 2 --steps 40 --landmarks 15  --noisy-init
Add --show-plot to visualise.
"""
import argparse, psutil, time, math, logging, sys, os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# project‑local imports ------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from sim.world import create_multi_robot_world
from sim.measurement import generate_multi_robot_measurements
from utils.cfg_loader import load_common

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("NoisySLAM")

# ------------------- helper -----------------------------------------------

def rmse(truth: np.ndarray, est: np.ndarray):
    diff = truth - est
    diff[:,:,2] = (diff[:,:,2]+math.pi)%(2*math.pi)-math.pi
    return np.sqrt(np.mean(diff**2, axis=(0,1)))

# ------------------- plotting ---------------------------------------------

def quick_plot(tra, est, lms_gt, lms_est=None, title="SLAM"):
    plt.figure(figsize=(8,7)); ax=plt.gca(); ax.set_aspect('equal')
    ax.scatter(lms_gt[:,0], lms_gt[:,1], marker='x', c='k', label='LM GT')
    if lms_est is not None:
        ax.scatter(lms_est[:,0], lms_est[:,1], marker='+', c='c', label='LM Est')
    cols=['r','b','g','m','y']
    for r in range(tra.shape[0]):
        c=cols[r%len(cols)]
        ax.plot(tra[r,:,0], tra[r,:,1], c, lw=2, label=f'R{r} GT')
        ax.plot(est[r,:,0], est[r,:,1], c, lw=1, ls='--', label=f'R{r} Est')
    ax.legend(); ax.set_title(title); plt.grid(True); plt.show()

# ------------------- main --------------------------------------------------

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--robots',type=int,default=2); p.add_argument('--steps',type=int,default=40)
    p.add_argument('--landmarks',type=int,default=15); p.add_argument('--noisy-init',action='store_true')
    p.add_argument('--plot',action='store_true'); return p.parse_args()

def main():
    args=parse_args(); np.random.seed(42)
    tra,lms = create_multi_robot_world(R=args.robots, T=args.steps, motion_type="figure8",
                                       num_landmarks=args.landmarks, world_size=20, noise_level=0.05)
    meas_cfg=dict(max_landmark_range=8, bearing_noise_std=0.05, range_noise_std=0.2,
                  enable_inter_robot_measurements=True, enable_loop_closure=False,
                  landmark_detection_prob=1.0, robot_detection_prob=1.0)
    meas, loops = generate_multi_robot_measurements(tra, lms, meas_cfg)

    graph,initial,key_map,stats = build_gtsam_graph(
        tra,lms, measurements=meas, loop_closures=loops, cfg=load_common(),
        use_truth_init= not args.noisy_init)  # invert flag name

    # ----- derive pre‑opt estimates ----------------------------------
    est0 = np.zeros_like(tra)
    for r in range(args.robots):
        for t in range(args.steps):
            k=key_map[f"x{r}_{t}"]; p=initial.atPose2(k)
            est0[r,t]=[p.x(),p.y(),p.theta()]
    pre_rmse = rmse(tra, est0)
    log.info("Initial pose RMSE = %.2f m, %.2f m, %.2f rad", *pre_rmse)

    # ----- optimise --------------------------------------------------
    result,opt_stats = run_gtsam_optimizer(graph, initial, max_iterations=100)

    # ----- post‑opt estimates ---------------------------------------
    est = np.zeros_like(tra)
    for r in range(args.robots):
        for t in range(args.steps):
            p=result.atPose2(key_map[f"x{r}_{t}"])
            est[r,t]=[p.x(),p.y(),p.theta()]
    post_rmse = rmse(tra, est)

    # landmarks
    est_lms = np.zeros_like(lms)
    for lid in range(lms.shape[0]):
        pt = result.atPoint2(key_map[f"l_{lid}"])
        est_lms[lid]=pt if isinstance(pt, np.ndarray) else np.array([pt.x(),pt.y()])
    lm_rmse = np.sqrt(np.mean((lms-est_lms)**2))

    print("\n=== Summary ===")
    print(f"Init mode        : {'truth' if not args.noisy_init else 'noisy'}")
    print(f"Initial pose RMSE: {pre_rmse[0]:.2f} {pre_rmse[1]:.2f} {pre_rmse[2]:.2f}")
    print(f"Final   pose RMSE: {post_rmse[0]:.2f} {post_rmse[1]:.2f} {post_rmse[2]:.2f}")
    print(f"Landmark RMSE    : {lm_rmse:.2f} m")
    print(f"Iterations       : {opt_stats['iterations']}")

    if args.plot:
        quick_plot(tra, est, lms, est_lms, title="Noisy‑init SLAM")

if __name__=='__main__':
    main()
