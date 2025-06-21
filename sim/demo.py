#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced GBP-SLAM Demo compatible with Production-Stable gbp.py
Enhanced with debugging, validation and terminal visualization
author: Hank – revised 2025-06-18
"""

import math, time, logging, collections, traceback, sys
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# ── Terminal Logo Display ──────────────────────────────────────────
def display_logo():
    """Display enhanced logo in terminal"""
    logo = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║     ███████╗██╗      █████╗ ███╗   ███╗    ██████╗ ███████╗     ║
    ║     ██╔════╝██║     ██╔══██╗████╗ ████║    ██╔══██╗██╔════╝     ║
    ║     ███████╗██║     ███████║██╔████╔██║    ██║  ██║█████╗       ║
    ║     ╚════██║██║     ██╔══██║██║╚██╔╝██║    ██║  ██║██╔══╝       ║
    ║     ███████║███████╗██║  ██║██║ ╚═╝ ██║    ██████╔╝███████╗     ║
    ║     ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝    ╚═════╝ ╚══════╝     ║
    ║                                                                  ║
    ║             🚀 Enhanced GBP-SLAM Demo System 🚀                  ║
    ║                                                                  ║
    ║        ✨ Production-Ready SLAM with Enhanced Debugging ✨       ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    
    print("\033[96m" + logo + "\033[0m")  # Cyan color
    print("\033[93m⚡ Initializing Enhanced GBP-SLAM Demo...\033[0m")  # Yellow
    time.sleep(0.5)

# Display logo first
display_logo()

# ── Enhanced imports with error handling ────────────────────────────
def safe_import_modules():
    """Import modules with proper error handling and correct path priority"""
    print("\033[94m🔄 Loading SLAM modules...\033[0m")
    
    try:
        # Try correct primary paths first
        from sim.world import paths, landmarks
        print("  \033[92m✓ World module (sim.world)\033[0m")
        world_module = "sim.world"
    except ImportError:
        try:
            # Fallback to algorithm.frontend
            from sim.world import paths, landmarks
            print("  \033[93m⚠ World module (algorithm.frontend.world - fallback)\033[0m")
            world_module = "algorithm.frontend.world"
        except ImportError as e:
            print(f"  \033[91m✗ World module: {e}\033[0m")
            return None
    
    try:
        # Try correct primary paths first
        from sim.graph_build import GBPGraphBuilder
        print("  \033[92m✓ Graph builder (sim.graph_build)\033[0m")
        graph_module = "sim.graph_build"
    except ImportError:
        try:
            # Fallback to algorithm.frontend
            from sim.graph_build import GBPGraphBuilder
            print("  \033[93m⚠ Graph builder (algorithm.frontend.graph_build - fallback)\033[0m")
            graph_module = "algorithm.frontend.graph_build"
        except ImportError as e:
            print(f"  \033[91m✗ Graph builder: {e}\033[0m")
            return None
    
    try:
        from algorithm.frontend.gbp import GBPGraph
        print("  \033[92m✓ GBP solver (algorithm.frontend.gbp)\033[0m")
    except ImportError as e:
        print(f"  \033[91m✗ GBP solver: {e}\033[0m")
        return None
    
    # Optional SPBP module
    try:
        from algorithm.frontend.spbp import SigmaPointGenerator
        print("  \033[92m✓ SPBP module (algorithm.frontend.spbp)\033[0m")
        spbp_available = True
    except ImportError:
        print("  \033[93m⚠ SPBP module not available (optional)\033[0m")
        spbp_available = False
    
    print("\033[92m🎉 All critical modules loaded successfully!\033[0m\n")
    return {
        'world_module': world_module,
        'graph_module': graph_module,
        'spbp_available': spbp_available,
        'paths': paths,
        'landmarks': landmarks,
        'GBPGraphBuilder': GBPGraphBuilder,
        'GBPGraph': GBPGraph
    }

# Import modules
modules = safe_import_modules()
if modules is None:
    print("\033[91m💥 Critical modules missing. Please check your installation.\033[0m")
    sys.exit(1)

# Extract imported objects
paths = modules['paths']
landmarks = modules['landmarks']
GBPGraphBuilder = modules['GBPGraphBuilder']
GBPGraph = modules['GBPGraph']

# ── Enhanced logging setup ─────────────────────────────────────────
class ColoredFormatter(logging.Formatter):
    """Enhanced formatter with colors and emojis"""
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m'  # Magenta
    }
    RESET = '\033[0m'
    
    EMOJIS = {
        'DEBUG': '🔍',
        'INFO': 'ℹ️ ',
        'WARNING': '⚠️ ',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        emoji = self.EMOJIS.get(record.levelname, '')
        reset = self.RESET
        
        log_message = super().format(record)
        return f"{color}{emoji} {log_message}{reset}"

# Setup enhanced logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[])

# Create console handler with colors
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Add file handler for detailed logs (using plain formatter to avoid ANSI codes in files)
log_filename = f'slam_demo_{int(time.time())}.log'
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
))

# Add handlers to root logger
root_logger = logging.getLogger()
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

log = logging.getLogger("SLAM-Demo")

# ── Enhanced utility functions ──────────────────────────────────────
def wrap_angle(th: float) -> float:
    """Wrap angle to [-π, π]"""
    return (th + math.pi) % (2.0 * math.pi) - math.pi

def validate_data_integrity(traj, lms, meas):
    """Validate data integrity and report issues with detailed diagnostics"""
    log.info("Validating data integrity...")
    issues = []
    
    # Check trajectory
    if not np.isfinite(traj).all():
        issues.append("Non-finite values in trajectory")
    
    # Check landmarks
    if not np.isfinite(lms).all():
        issues.append("Non-finite values in landmarks")
    
    # Detailed measurement analysis
    bearing_values = []
    range_values = []
    
    for i, m in enumerate(meas):
        if "bearing_range" not in m:
            issues.append(f"Measurement {i}: missing 'bearing_range'")
            continue
        
        br = m["bearing_range"]
        if len(br) != 2:
            issues.append(f"Measurement {i}: bearing_range has {len(br)} elements, expected 2")
            continue
        
        bearing, range_val = br[0], br[1]
        bearing_values.append(bearing)
        range_values.append(range_val)
        
        # Check bearing range
        if abs(bearing) > math.pi:
            issues.append(f"Measurement {i}: bearing {bearing:.3f} outside [-π, π]")
        if range_val <= 0:
            issues.append(f"Measurement {i}: invalid range {range_val:.3f}")
    
    # Statistical analysis of measurements
    if bearing_values and range_values:
        bearing_stats = {
            'min': np.min(bearing_values),
            'max': np.max(bearing_values), 
            'mean': np.mean(bearing_values),
            'std': np.std(bearing_values)
        }
        range_stats = {
            'min': np.min(range_values),
            'max': np.max(range_values),
            'mean': np.mean(range_values), 
            'std': np.std(range_values)
        }
        
        log.info("Measurement statistics:")
        log.info("  Bearings: min=%.3f, max=%.3f, mean=%.3f, std=%.3f", 
                bearing_stats['min'], bearing_stats['max'], 
                bearing_stats['mean'], bearing_stats['std'])
        log.info("  Ranges: min=%.3f, max=%.3f, mean=%.3f, std=%.3f",
                range_stats['min'], range_stats['max'],
                range_stats['mean'], range_stats['std'])
        
        # Check for suspicious values
        if range_stats['max'] > 50.0:
            issues.append(f"Very large range detected: {range_stats['max']:.3f} m")
        if range_stats['std'] > range_stats['mean']:
            issues.append(f"High range variance (std={range_stats['std']:.3f} > mean={range_stats['mean']:.3f})")
    
    if issues:
        log.warning("Data integrity issues found:")
        for issue in issues:
            log.warning("  %s", issue)
        return False
    else:
        log.info("✓ Data integrity validation passed")
        return True

def plot_cov_ellipse(ax, mean: np.ndarray, cov: np.ndarray,
                     n_std: float = 2.0, **kw):
    """Enhanced 2-D covariance ellipse plotting with error handling"""
    try:
        if cov.shape[0] < 2:
            return
        
        # Extract 2x2 submatrix and ensure positive definite
        cov_2d = cov[:2, :2]
        vals, vecs = np.linalg.eigh(cov_2d)
        vals = np.clip(vals, 1e-9, None)  # Ensure positive eigenvalues
        
        w, h = 2 * n_std * np.sqrt(vals)
        ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        
        # Fix matplotlib deprecation warning by using angle as keyword argument
        ellipse = Ellipse(mean[:2], w, h, angle=ang, fill=False, **kw)
        ax.add_patch(ellipse)
    except Exception as e:
        log.debug("Failed to plot covariance ellipse: %s", e)

def make_measurements(traj: np.ndarray, lms: np.ndarray,
                      sigma_r=0.01, sigma_b=np.deg2rad(0.1),  # Ultra-low noise for stability
                      max_range=8.0, p_obs=0.9, seed: int = 42  # Shorter range, higher detection
                      ) -> List[Dict[str, Any]]:
    """Enhanced measurement generation with ultra-conservative noise for stability"""
    log.info("Generating measurements with ultra-conservative parameters...")
    log.info("Noise: σ_r=%.3f m, σ_b=%.1f°, max_range=%.1f m, p_obs=%.1f", 
             sigma_r, np.degrees(sigma_b), max_range, p_obs)
    
    rng = np.random.default_rng(seed)
    R, T, _ = traj.shape
    meas = []
    
    total_possible = 0
    total_generated = 0
    
    for r in range(R):
        for t in range(T):
            pos_r = traj[r, t, :2]
            th_r = traj[r, t, 2]
            dxy = lms - pos_r
            dists = np.linalg.norm(dxy, axis=1)
            
            for i, d in enumerate(dists):
                total_possible += 1
                
                if d > max_range or rng.random() > p_obs:
                    continue
                
                true_bear = wrap_angle(math.atan2(dxy[i, 1], dxy[i, 0]) - th_r)
                noisy_bearing = wrap_angle(true_bear + rng.normal(0, sigma_b))
                noisy_range = d + rng.normal(0, sigma_r)
                
                # Very conservative validation and clamping
                if noisy_range <= 0.5:  # Higher minimum range threshold
                    log.debug("Skipping small range measurement: %.3f", noisy_range)
                    continue
                
                # Clamp to very reasonable values
                noisy_range = max(0.5, min(noisy_range, max_range))
                
                # Additional sanity check
                if abs(noisy_bearing) > math.pi:
                    log.debug("Skipping invalid bearing: %.3f", noisy_bearing)
                    continue
                
                meas.append({
                    "type": "robot_lm",
                    "robot": r,
                    "time": t,
                    "id": i,
                    "bearing_range": np.array([noisy_bearing, noisy_range])
                })
                total_generated += 1
    
    detection_rate = total_generated / total_possible if total_possible > 0 else 0
    log.info("Generated %d measurements from %d possible (%.1f%% detection rate)", 
             total_generated, total_possible, detection_rate * 100)
    
    return meas

def prepare_priors(vars_dict: Dict[str, np.ndarray]
                   ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Enhanced prior preparation with very strong priors for maximum numerical stability"""
    log.info("Preparing priors for %d variables...", len(vars_dict))
    
    pri = {}
    for k, init in vars_dict.items():
        # Validate initial values
        if not np.isfinite(init).all():
            log.warning("Non-finite initial value for variable %s: %s", k, init)
            init = np.nan_to_num(init)
        
        if k == "x0_0":
            # Extremely strong prior for first pose to firmly anchor the solution
            pri[k] = (init, np.array([0.001, 0.001, np.deg2rad(0.1)]))  # 1mm, 0.1°
        elif k.startswith("x"):
            # Strong priors for all poses to prevent drift
            pri[k] = (init, np.array([0.1, 0.1, np.deg2rad(2)]))  # 10cm, 2°
        else:
            # Strong landmark priors
            pri[k] = (init, np.array([0.5, 0.5]))  # 50cm
    
    log.info("Prepared priors: %d poses, %d landmarks", 
             sum(1 for k in pri if k.startswith('x')),
             sum(1 for k in pri if k.startswith('l')))
    
    # Log some sample priors for debugging
    for k in list(pri.keys())[:3]:
        mean, sigma = pri[k]
        log.debug("Prior %s: mean=%s, sigma=%s", k, mean, sigma)
    
    return pri

def analyze_convergence(energy_log):
    """Analyze convergence behavior"""
    if not energy_log:
        return {}
    
    analysis = {
        'converged': energy_log[-1] < 1e-4,
        'total_iterations': len(energy_log),
        'final_delta': energy_log[-1],
        'initial_delta': energy_log[0]
    }
    
    # Check for oscillations
    if len(energy_log) > 10:
        recent = energy_log[-10:]
        oscillation_score = np.std(recent) / np.mean(recent) if np.mean(recent) > 0 else 0
        analysis['oscillating'] = oscillation_score > 0.1
    
    return analysis

def enhanced_visualization(traj, est_traj, lms, est_lms, marginals, pos_err, ori_err, energy_log):
    """Enhanced visualization with more detailed analysis"""
    fig = plt.figure(figsize=(16, 10))
    
    # Main trajectory plot
    ax1 = plt.subplot(2, 3, (1, 4))
    ax1.plot(traj[0, :, 0], traj[0, :, 1], 'b-', lw=2, label="Ground Truth", alpha=0.8)
    ax1.plot(est_traj[:, 0], est_traj[:, 1], 'r--', lw=2, label="Estimate", alpha=0.8)
    
    # Mark start and end points
    ax1.scatter(traj[0, 0, 0], traj[0, 0, 1], c='blue', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(traj[0, -1, 0], traj[0, -1, 1], c='blue', s=100, marker='s', label='End', zorder=5)
    ax1.scatter(est_traj[0, 0], est_traj[0, 1], c='red', s=80, marker='o', zorder=5)
    ax1.scatter(est_traj[-1, 0], est_traj[-1, 1], c='red', s=80, marker='s', zorder=5)
    
    # Landmarks
    ax1.scatter(lms[:, 0], lms[:, 1], c='black', s=100, marker='*', label="GT landmarks", zorder=5)
    ax1.scatter(est_lms[:, 0], est_lms[:, 1], c='red', s=60, marker='x', label="Est landmarks", zorder=5)
    
    # Uncertainty ellipses
    for i in range(lms.shape[0]):
        cov_i = marginals.get(f"l_{i}", (None, None))[1]
        if cov_i is not None:
            plot_cov_ellipse(ax1, est_lms[i], cov_i, edgecolor='red', alpha=0.3)
    
    ax1.set_aspect('equal')
    ax1.legend()
    ax1.set_title("Trajectory and Landmarks", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Position error
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(pos_err, 'b-', lw=1.5)
    ax2.axhline(y=np.sqrt(np.mean(pos_err**2)), color='r', linestyle='--', 
                label=f'RMSE: {np.sqrt(np.mean(pos_err**2)):.3f}m')
    ax2.set_title("Position Error")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Error (m)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Orientation error
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(np.degrees(ori_err), 'g-', lw=1.5)
    ori_rmse = np.sqrt(np.mean(np.asarray(ori_err)**2))
    ax3.axhline(y=np.degrees(ori_rmse), color='r', linestyle='--',
                label=f'RMSE: {np.degrees(ori_rmse):.1f}°')
    ax3.set_title("Orientation Error")
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Error (°)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Convergence plot
    ax4 = plt.subplot(2, 3, 5)
    if energy_log:
        ax4.semilogy(energy_log, 'purple', lw=2)
        ax4.axhline(y=1e-4, color='r', linestyle='--', alpha=0.7, label='Target')
        ax4.legend()
    ax4.set_title("Convergence")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Max Delta (log)")
    ax4.grid(True, alpha=0.3)
    
    # Error statistics
    ax5 = plt.subplot(2, 3, 6)
    ax5.axis('off')
    
    # Compute comprehensive statistics
    pos_rmse = np.sqrt(np.mean(pos_err**2))
    pos_mae = np.mean(pos_err)
    ori_rmse = np.sqrt(np.mean(np.asarray(ori_err)**2))  # More efficient calculation
    ori_mae = np.mean(np.abs(ori_err))
    
    lm_errors = np.linalg.norm(lms - est_lms, axis=1)
    lm_rmse = np.sqrt(np.mean(lm_errors**2))
    
    # ----- helper strings for f-string below -----
    final_delta_str = f"{energy_log[-1]:.2e}" if energy_log else "N/A"
    converged_flag  = "✓" if energy_log and energy_log[-1] < 1e-4 else "✗"
    
    stats_text = f"""
    Performance Statistics
    ━━━━━━━━━━━━━━━━━━━━
    
    Position Errors:
      RMSE: {pos_rmse:.3f} m
      MAE:  {pos_mae:.3f} m
      Max:  {np.max(pos_err):.3f} m
    
    Orientation Errors:
      RMSE: {np.degrees(ori_rmse):.2f}°
      MAE:  {np.degrees(ori_mae):.2f}°
      Max:  {np.degrees(np.max(np.abs(ori_err))):.2f}°
    
    Landmark Errors:
      RMSE: {lm_rmse:.3f} m
      MAE:  {np.mean(lm_errors):.3f} m
    
    Convergence:
      Iterations: {len(energy_log) if energy_log else 0}
      Final Δ: {final_delta_str}
      Converged: {converged_flag}
    """
    
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle('Enhanced GBP-SLAM Results Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'slam_results_{int(time.time())}.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    log.info("Saved results plot to: %s", plot_filename)
    
    plt.show()

# ────────────────────────────────────────────────────────────
def main():
    print(f"\n\033[96m📝 Detailed logs will be saved to: {log_filename}\033[0m")
    print("\033[94m🚀 Starting SLAM computation...\033[0m\n")
    
    t0 = time.time()
    np.random.seed(42)

    # 1. Generate world
    log.info("Generating world...")
    print("\033[93m🌍 Generating world...\033[0m")
    
    lms = landmarks(num=5, distribution="gaussian", sigma=[7, 7])
    traj = paths(R=1, T=50, motion_type="figure8", center=(0, 0), radius=10,
                 angular_velocity=0.1, linear_velocity=0.15,
                 noise_std=np.array([0.05, 0.05, 0.01]))
    
    if traj.ndim == 2:  # Ensure shape = (1,T,3)
        traj = traj[None, ...]

    log.info("Generated trajectory: %s, landmarks: %s", traj.shape, lms.shape)
    print(f"  \033[92m✓ Trajectory: {traj.shape}, Landmarks: {lms.shape}\033[0m")

    # 2. Generate measurements
    print("\033[93m📡 Generating measurements...\033[0m")
    meas = make_measurements(traj, lms)
    print(f"  \033[92m✓ Generated {len(meas)} measurements\033[0m")

    # 3. Validate data
    print("\033[93m🔍 Validating data integrity...\033[0m")
    validation_passed = validate_data_integrity(traj, lms, meas)
    if validation_passed:
        print("  \033[92m✓ Data validation passed\033[0m")
    else:
        print("  \033[93m⚠ Data validation warnings (see logs)\033[0m")

    # 4. Build factor graph
    print("\033[93m🔧 Building factor graph...\033[0m")
    # Ultra-conservative configuration for numerical stability
    cfg = dict(
        ut_mode="gbp",  # Use standard mode for maximum stability
        ut_alpha=0.01,     # Very small alpha for maximum conservatism  
        ut_beta=2.0, 
        ut_kappa=0.0,
        odom_sigma_xy=0.01,  # Very small odometry noise
        odom_sigma_theta=0.002  # Very small angular noise
    )
    
    log.info("Using ultra-conservative configuration: %s", cfg)
    builder = GBPGraphBuilder(cfg)
    
    # Ensure configuration is properly applied
    log.debug("Builder configuration: %s", builder.cfg if hasattr(builder, 'cfg') else 'N/A')
    
    factors, variables = builder.build(traj, lms, meas)
    
    log.info("Built graph: %d variables, %d factors", len(variables), len(factors))
    
    # Diagnostic: Check factor health
    log.info("Performing factor health check...")
    try:
        for i, factor in enumerate(factors[:5]):  # Check first 5 factors
            if hasattr(factor, 'Rinv'):
                cond_num = np.linalg.cond(factor.Rinv)
                log.debug("Factor %d condition number: %.2e", i, cond_num)
                if cond_num > 1e12:
                    log.warning("Factor %d has ill-conditioned noise matrix: %.2e", i, cond_num)
    except Exception as e:
        log.debug("Could not check factor condition numbers: %s", e)
    
    log.info("Built graph: %d variables, %d factors", len(variables), len(factors))
    print(f"  \033[92m✓ Graph: {len(variables)} variables, {len(factors)} factors\033[0m")

    # 5. Prepare priors and run GBP
    print("\033[93m⚙️  Running GBP optimization...\033[0m")
    priors = prepare_priors(variables)
    
    # Ultra-conservative GBP parameters for maximum stability
    gbp_params = {
        'bel_damp': 0.9,      # Very high belief damping
        'msg_damp': 0.8       # Very high message damping  
    }
    
    log.info("Using ultra-conservative GBP parameters: %s", gbp_params)
    graph = GBPGraph(factors=factors,
                     variables=variables,
                     priors=priors,
                     **gbp_params)

    # Run optimization with very conservative parameters and more monitoring
    log.info("Starting GBP optimization with max_iter=50, tol=1e-3...")
    energy_log = graph.run(max_iter=50, tol=1e-3)  # Shorter run, relaxed tolerance
    
    # Enhanced convergence monitoring
    if energy_log:
        log.info("Energy progression (first 10): %s", 
                [f"{e:.2e}" for e in energy_log[:10]])
        if len(energy_log) > 10:
            log.info("Energy progression (last 10): %s",
                    [f"{e:.2e}" for e in energy_log[-10:]])
        
        # Check for explosive growth
        if len(energy_log) >= 2:
            growth_ratio = energy_log[-1] / energy_log[0] if energy_log[0] > 0 else float('inf')
            log.info("Energy growth ratio: %.2e", growth_ratio)
            if growth_ratio > 1e6:
                log.error("Explosive energy growth detected (ratio=%.2e)", growth_ratio)
    
    # Analyze convergence
    conv_analysis = analyze_convergence(energy_log)
    
    runtime = time.time() - t0
    log.info("Optimization finished in %.2f seconds", runtime)
    log.info("Iterations: %d, Final Δ: %.2e", 
             len(energy_log), energy_log[-1] if energy_log else float('nan'))
    
    if conv_analysis['converged']:
        print(f"  \033[92m✅ Converged in {len(energy_log)} iterations (Δ={energy_log[-1]:.2e})\033[0m")
    else:
        print(f"  \033[93m⚠ Did not converge in {len(energy_log)} iterations (Δ={energy_log[-1]:.2e})\033[0m")
    
    if conv_analysis.get('oscillating', False):
        print(f"  \033[93m⚠ Oscillation detected in convergence\033[0m")

    # 6. Extract results
    print("\033[93m📊 Extracting results...\033[0m")
    means = graph.get_means()
    marginals = graph.get_marginals()

    # Extract estimates
    T = traj.shape[1]
    est_traj = np.vstack([means.get(f"x0_{t}", traj[0, t]) for t in range(T)])
    est_lms = np.vstack([means.get(f"l_{i}", lms[i]) for i in range(lms.shape[0])])

    # 7. Compute comprehensive errors with validation
    pos_err = np.linalg.norm(traj[0, :, :2] - est_traj[:, :2], axis=1)
    ori_err = [wrap_angle(traj[0, t, 2] - est_traj[t, 2]) for t in range(T)]
    lm_err = np.linalg.norm(lms - est_lms, axis=1)
    
    # Compute and display metrics with sanity checks
    pos_rmse = np.sqrt((pos_err**2).mean())
    ori_mae = np.degrees(np.abs(ori_err)).mean()
    lm_rmse = np.sqrt((lm_err**2).mean())
    
    # Check for unrealistic results (indicating numerical failure)
    if pos_rmse > 1000 or not np.isfinite(pos_rmse):
        log.error("Unrealistic position RMSE detected: %.3f m - numerical instability likely", pos_rmse)
        print(f"  \033[91m❌ Position RMSE: {pos_rmse:.3f} m (UNREALISTIC - NUMERICAL FAILURE)\033[0m")
    else:
        print(f"  \033[92m✓ Position RMSE: {pos_rmse:.3f} m\033[0m")
    
    if ori_mae > 90 or not np.isfinite(ori_mae):
        log.error("Unrealistic orientation MAE detected: %.1f° - numerical instability likely", ori_mae)
        print(f"  \033[91m❌ Orientation MAE: {ori_mae:.1f}° (UNREALISTIC)\033[0m")
    else:
        print(f"  \033[92m✓ Orientation MAE: {ori_mae:.1f}°\033[0m")
    
    if lm_rmse > 100 or not np.isfinite(lm_rmse):
        log.error("Unrealistic landmark RMSE detected: %.3f m - numerical instability likely", lm_rmse)
        print(f"  \033[91m❌ Landmark RMSE: {lm_rmse:.3f} m (UNREALISTIC)\033[0m")
    else:
        print(f"  \033[92m✓ Landmark RMSE: {lm_rmse:.3f} m\033[0m")
    
    log.info("Position RMSE: %.3f m, Orientation MAE: %.1f°, Landmark RMSE: %.3f m",
             pos_rmse, ori_mae, lm_rmse)

    # 8. Check SPBP statistics if available
    if modules['spbp_available']:
        try:
            from algorithm.frontend.spbp import _default_generator
            if hasattr(_default_generator, 'get_cache_stats'):
                spbp_stats = _default_generator.get_cache_stats()
                log.info("SPBP cache stats: %s", spbp_stats)
                print(f"  \033[96m📈 SPBP cache stats: {spbp_stats}\033[0m")
            else:
                log.debug("SPBP _default_generator has no get_cache_stats method")
        except ImportError as e:
            log.debug("Could not import _default_generator from SPBP: %s", e)
        except AttributeError as e:
            log.debug("SPBP _default_generator not properly initialized: %s", e)
        except Exception as e:
            log.debug("Could not get SPBP stats: %s", e)

    # 9. Enhanced visualization
    print("\033[93m📊 Creating enhanced visualization...\033[0m")
    enhanced_visualization(traj, est_traj, lms, est_lms, marginals, 
                          pos_err, ori_err, energy_log)

    # Final summary with detailed diagnostic information
    print("\n" + "═" * 80)
    print("\033[96m🎉 SLAM Demo Completed!\033[0m")
    print("═" * 80)
    print(f"\033[92m✅ Total Runtime: {runtime:.2f} seconds\033[0m")
    
    # Provide comprehensive diagnostic information based on results
    if conv_analysis['converged'] and pos_rmse < 1.0:
        print(f"\033[92m🎯 EXCELLENT: Converged with good accuracy (RMSE: {pos_rmse:.3f}m)\033[0m")
    elif conv_analysis['converged'] and pos_rmse < 10.0:
        print(f"\033[93m🎯 GOOD: Converged with acceptable accuracy (RMSE: {pos_rmse:.3f}m)\033[0m")
    elif conv_analysis['converged']:
        print(f"\033[93m🎯 CONVERGED: But with poor accuracy (RMSE: {pos_rmse:.3f}m)\033[0m")
        print(f"\033[93m💡 Consider: reduce noise, add more measurements, or check sensor calibration\033[0m")
    elif pos_rmse > 1000:
        print(f"\033[91m⚠️  NUMERICAL FAILURE: Results are unrealistic\033[0m")
        print(f"\033[93m🔧 DEBUG STEPS:\033[0m")
        print(f"\033[93m   1. Check measurement statistics in logs\033[0m")
        print(f"\033[93m   2. Verify factor condition numbers\033[0m")
        print(f"\033[93m   3. Try even smaller noise parameters\033[0m")
        print(f"\033[93m   4. Check for measurement format issues\033[0m")
        print(f"\033[93m   5. Consider switching to 'standard' UT mode\033[0m")
        
        # Additional diagnostic info
        if energy_log:
            if len(energy_log) >= 2:
                growth = energy_log[-1] / energy_log[0] if energy_log[0] > 0 else float('inf')
                print(f"\033[93m   Energy growth ratio: {growth:.2e}\033[0m")
            print(f"\033[93m   Initial energy: {energy_log[0]:.2e}, Final: {energy_log[-1]:.2e}\033[0m")
    else:
        print(f"\033[93m⚠️  DID NOT CONVERGE: May need parameter tuning\033[0m")
        print(f"\033[93m   Performance: Position RMSE {pos_rmse:.3f}m, Orientation MAE {ori_mae:.1f}°\033[0m")
        if energy_log and len(energy_log) >= 2:
            improvement = (energy_log[0] - energy_log[-1]) / energy_log[0] * 100
            print(f"\033[93m   Energy improvement: {improvement:.1f}%\033[0m")
    
    print(f"\033[96m📝 Detailed logs saved to: {log_filename}\033[0m")
    print(f"\033[96m📊 Results plot saved\033[0m")
    print("═" * 80)

if __name__ == "__main__":
    try:
        main()
        print("\n\033[92m🎉 Thank you for using Enhanced GBP-SLAM!\033[0m")
    except Exception as e:
        print(f"\n\033[91m💥 Fatal error: {e}\033[0m")
        log.error("Fatal error: %s", e)
        log.error(traceback.format_exc())
        sys.exit(1)