#!/usr/bin/env python
"""
Light-weight GBP-SLAM demo (debug-friendly) – fixed for _delta TypeError
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Single file, ≈370 LOC
* Requires fixed modules: algorithm.frontend.{graph_build v2.1, factor_ut v2.3}
* CLI: python demo_debug.py -h
"""
from __future__ import annotations
import sys, time, math, argparse, logging, traceback
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

import numpy as np

# ────────────────────────────────────────────────────────────
# 1. Dynamic imports (fixed paths expected)
# ────────────────────────────────────────────────────────────
def import_or_die():
    try:
        from sim.world import paths, landmarks              # type: ignore
        from algorithm.frontend.graph_build import GBPGraphBuilder  # type: ignore
        from algorithm.frontend.gbp import GBPGraph                  # type: ignore
        print("✓ Core modules loaded (fixed versions)")
    except ImportError as e:
        print(f"❌ Cannot import fixed SLAM modules: {e}")
        sys.exit(1)

    cache_ok = False
    try:
        from algorithm.frontend.factor_ut import get_global_cache_stats  # type: ignore
        cache_ok = True
        print("✓ Cache management available (fixed v2.3)")
    except ImportError:
        print("⚠ Cache management not available")

    return paths, landmarks, GBPGraphBuilder, GBPGraph, cache_ok

paths, landmarks, GBPGraphBuilder, GBPGraph, cache_available = import_or_die()

# ────────────────────────────────────────────────────────────
# 2. Logging helper
# ────────────────────────────────────────────────────────────
def setup_logging(debug: bool, log_file: Optional[str]):
    lvl = logging.DEBUG if debug else logging.INFO
    root = logging.getLogger()
    root.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(lvl)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)5s | %(name)s | %(message)s",
                                      "%H:%M:%S"))
    root.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)5s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"))
        root.addHandler(fh)
        print(f"📝 Logs → {log_file}")

    root.setLevel(lvl)
    return logging.getLogger("DEMO")

log = logging.getLogger("DEMO")          # placeholder, real init in main()

# ────────────────────────────────────────────────────────────
# 3. Utilities
# ────────────────────────────────────────────────────────────
def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2*math.pi) - math.pi

def validate_data(traj: np.ndarray, lms: np.ndarray,
                  meas: List[Dict[str, Any]], skip: bool = False) -> bool:
    if skip: return True
    bad: List[str] = []
    if not np.isfinite(traj).all(): bad.append("traj NaN/inf")
    if not np.isfinite(lms).all():  bad.append("lms NaN/inf")
    for i, m in enumerate(meas[:5]):            # 仅抽样前5条即可
        if "bearing_range" not in m: bad.append(f"meas{i} missing br")
    if bad:
        log.warning("Data issues: %s", ", ".join(bad))
        return False
    return True

def factor_health(factors, k: int = 5):
    stats = {}
    for i, f in enumerate(factors[:k]):
        try:
            if hasattr(f, "Rinv"):
                c = np.linalg.cond(f.Rinv)
                stats[f"f{i}"] = c
        except Exception: pass
    if stats:
        log.info("Factor health: cond min=%.2e max=%.2e",
                 min(stats.values()), max(stats.values()))
    return stats

def make_meas(traj, lms, rng, max_r=8.0, p=0.9,
              σr=0.02, σb=math.radians(0.3)):
    R,T,_ = traj.shape
    out: List[Dict[str,Any]] = []
    total = 0
    for r in range(R):
        for t in range(T):
            pos, th = traj[r,t,:2], traj[r,t,2]
            dxy = lms - pos
            d   = np.linalg.norm(dxy,axis=1)
            total += len(d)
            ok   = (d < max_r) & (rng.random(len(d)) < p)
            for i in np.where(ok)[0]:
                br = wrap_angle(math.atan2(dxy[i,1], dxy[i,0]) - th)
                out.append({
                    "type":"robot_lm","robot":r,"time":t,"id":i,
                    "bearing_range": np.array([wrap_angle(br+rng.normal(0,σb)),
                                               max(0.1,d[i]+rng.normal(0,σr))])
                })
    log.info("Generated %d measurements (%.1f%% visible)",
             len(out), 100*len(out)/total)
    return out

# ────────────────────────────────────────────────────────────
# 4. Pipeline helpers
# ────────────────────────────────────────────────────────────
def world(n_lm:int, T:int, seed:int):
    np.random.seed(seed)
    lms  = landmarks(num=n_lm, distribution="gaussian", sigma=[7,7])
    traj = paths(R=1,T=T,motion_type="figure8",center=(0,0),radius=10,
                 angular_velocity=0.1, linear_velocity=0.15,
                 noise_std=np.array([.05,.05,.01]))
    if traj.ndim==2: traj=traj[None,...]
    return traj,lms

def build_graph(traj,lms,meas,cfg, diag=True):
    if diag: cfg = {**cfg,"enable_performance_monitoring":True}
    b=GBPGraphBuilder(cfg)
    f,v=b.build(traj,lms,meas)
    log.info("Graph: %d vars, %d factors", len(v), len(f))
    if diag and hasattr(b,"get_performance_diagnostics"):
        try:
            s=b.get_performance_diagnostics()["build_stats"]
            log.info("Build time %.3fs, vectorization %.1fx",
                     s.get("build_time",0),s.get("vectorization_speedup",1))
        except Exception: pass
    return f,v

def priors(vars:Dict[str,np.ndarray]):
    out={}
    for k,x0 in vars.items():
        if k=="x0_0": out[k]=(x0, np.array([.1,.1,math.radians(5)]))
        elif k.startswith("x"): out[k]=(x0, np.array([.5,.5,math.radians(15)]))
        else: out[k]=(x0, np.array([1.,1.]))
    return out

def _delta(val):                       # ← 修复 TypeError
    if isinstance(val, dict):
        return float(val.get("max_delta", float("inf")))
    if isinstance(val, (list, tuple, np.ndarray)):
        return float(val[0]) if len(val) else float("inf")
    return float(val)

def optimise(factors, vars_, pri, max_iter, tol, damp):
    g = GBPGraph(factors=factors, variables=vars_, priors=pri,
                 bel_damp=damp[0], msg_damp=damp[1])
    t=time.perf_counter()
    raw_log=g.run(max_iter=max_iter, tol=tol)
    dur=time.perf_counter()-t
    energy=[_delta(e) for e in raw_log]  # 保证是 float 列表
    log.info("Opt done: %d iters, final Δ=%.2e, %.3fs",
             len(energy), energy[-1] if energy else float("nan"), dur)
    return energy,g

# ────────────────────────────────────────────────────────────
# 5. Plot (headless 可关闭)
# ────────────────────────────────────────────────────────────
def plot(traj,lms,means,energy,save=None,show=True):
    import matplotlib.pyplot as plt
    T=traj.shape[1]
    est=np.vstack([means.get(f"x0_{t}",traj[0,t])[:2] for t in range(T)])
    rmse=float(np.sqrt(((est-traj[0,:,:2])**2).sum(1).mean()))

    fig,ax=plt.subplots(1,3,figsize=(15,5))
    ax0,ax1,ax2=ax
    ax0.plot(traj[0,:,0],traj[0,:,1],"b-",lw=2,label="GT")
    ax0.plot(est[:,0],est[:,1],"r--",lw=2,label="Est")
    ax0.scatter(lms[:,0],lms[:,1],c="k",marker="*",s=80,label="LM")
    ax0.scatter([est[0,0]],[est[0,1]],c="r",marker="o")
    ax0.set_aspect("equal");ax0.grid();ax0.legend()
    ax0.set_title(f"Traj RMSE={rmse:.3f} m")

    err=np.linalg.norm(est-traj[0,:,:2],1)
    ax1.plot(err,"b-");ax1.grid();ax1.set_title("Pos error")

    if energy: ax2.semilogy(energy,"m-");ax2.grid()
    ax2.set_title("Convergence")
    plt.tight_layout()
    if save: plt.savefig(save,dpi=150); log.info("Plot → %s",save)
    if show: plt.show(); 
    else: plt.close(fig)

# ────────────────────────────────────────────────────────────
# 6. CLI / main
# ────────────────────────────────────────────────────────────
def cli(argv=None):
    p=argparse.ArgumentParser("demo_debug",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-T",type=int,default=40,help="trajectory length")
    p.add_argument("-n","--landmarks",type=int,default=8,help="#landmarks")
    p.add_argument("--max-iter",type=int,default=30,help="GBP max iterations")
    p.add_argument("--tol",type=float,default=1e-3,help="conv tolerance")
    p.add_argument("--damping",type=float,nargs=2,default=[0.8,0.7],
                   metavar=("BEL","MSG"),help="belief / message damping")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--debug",action="store_true")
    p.add_argument("--log-file")
    p.add_argument("--fast",action="store_true",help="skip heavy checks")
    p.add_argument("--headless",action="store_true")
    p.add_argument("--save-plot")
    return p.parse_args(argv)

def main(argv=None):
    a=cli(argv)
    global log
    log=setup_logging(a.debug, a.log_file or (f"debug_{int(time.time())}.log" if a.debug else None))
    log.info("Start demo: T=%d, LM=%d, max_iter=%d, tol=%1.1e",
             a.T, a.landmarks, a.max_iter, a.tol)

    rng=np.random.default_rng(a.seed)
    traj,lms=world(a.landmarks,a.T,a.seed)
    meas=make_meas(traj,lms,rng)
    validate_data(traj,lms,meas,skip=a.fast)

    cfg=dict(ut_mode="gbp", odom_sigma_xy=0.02, odom_sigma_theta=0.005,
             obs_sigma_bearing=math.radians(0.3), obs_sigma_range=0.02,
             enable_true_vectorization=True, enable_batch_processing=True,
             enable_cache_coordination=True)
    fac,vars_=build_graph(traj,lms,meas,cfg,diag=not a.fast)
    if not a.fast: factor_health(fac)

    energy,graph=optimise(fac,vars_,priors(vars_),a.max_iter,a.tol,a.damping)
    means=graph.get_means()

    if cache_available and not a.fast:
        try:
            from algorithm.frontend.factor_ut import get_global_cache_stats
            log.info("Cache stats: %s", get_global_cache_stats())
        except Exception: pass

    plot(traj,lms,means,energy,
         save=a.save_plot or ("result.png" if a.headless else None),
         show=not a.headless)

    rmse=np.sqrt(((np.vstack([means[f"x0_{t}"][:2] for t in range(a.T)])-
                   traj[0,:,:2])**2).sum(1).mean())
    conv=energy[-1] < a.tol if energy else False
    print(f"\n{'='*60}")
    print(f"RMSE {rmse:.3f} m | Converged: {conv}")
    print(f"{'='*60}")
    return 0 if conv and rmse<10 else 1

if __name__=="__main__":
    sys.exit(main())
