#!/usr/bin/env python
# coding: utf-8
"""
Gaussian Belief Propagation SLAM Demo (Fixed Version)
author: Hank   2025-06-05
"""

import math
import time
import logging
import collections
import traceback
import sys
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# ======== Custom Modules =========
from sim.world import paths, landmarks
from sim.graph_build import GBPGraphBuilder
from algorithm.frontend.gbp import GBPGraph

# ---------------- Logging Configuration ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SLAM-Demo")

# ========= Core Utility Functions =========
def wrap_angle(theta: float) -> float:
    """Normalize angle to [-π, π] range"""
    return (theta + math.pi) % (2.0 * math.pi) - math.pi

def plot_cov_ellipse(ax, mean: np.ndarray, cov: np.ndarray, 
                     n_std: float = 2.0, **kwargs):
    """Plot covariance ellipse (robust version)"""
    if cov.size == 0 or cov.shape[0] < 2:
        return
        
    # Use only first two dimensions
    if mean.size > 2:
        mean = mean[:2]
        
    if cov.ndim != 2 or cov.shape[0] < 2 or cov.shape[1] < 2:
        return
        
    cov = cov[:2, :2]
    
    try:
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-6)  # Prevent negative values
        width, height = 2 * n_std * np.sqrt(vals)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        ellipse = Ellipse(xy=mean, width=width, height=height, 
                         angle=angle, fill=False, **kwargs)
        ax.add_patch(ellipse)
    except Exception as e:
        logger.warning(f"Failed to plot covariance ellipse: {str(e)}")
        
def make_measurements(paths: np.ndarray, lms: np.ndarray, 
                     sigma_r: float = 0.5, sigma_b: float = np.deg2rad(10),
                     max_range: float = 25.0, p_obs: float = 0.7, 
                     seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Generate robot-to-landmark measurements"""
    np.random.seed(seed)
    R, T, _ = paths.shape
    measurements = []
    
    for r in range(R):
        for t in range(T):
            p_r = paths[r, t, :2]
            theta_r = paths[r, t, 2]
            
            # Vectorized distance calculation
            delta = lms - p_r
            dists = np.sqrt(delta[:, 0]**2 + delta[:, 1]**2)
            visible_idx = np.where(dists < max_range)[0]
            
            for lm_id in visible_idx:
                if np.random.rand() > p_obs:  # Probability of detection
                    continue
                    
                # Calculate true distance and bearing
                dx = lms[lm_id, 0] - p_r[0]
                dy = lms[lm_id, 1] - p_r[1]
                true_range = math.sqrt(dx**2 + dy**2)
                true_bearing = wrap_angle(math.atan2(dy, dx) - theta_r)
                
                # Add noise
                meas_range = true_range + np.random.normal(0, sigma_r)
                meas_bearing = true_bearing + np.random.normal(0, sigma_b)
                
                measurements.append({
                    "type": "robot_lm",
                    "robot": r,
                    "time": t,
                    "id": int(lm_id),
                    "bearing_range": np.array([meas_bearing, meas_range])
                })
    
    return measurements

# ==================== Main Pipeline ====================
def main():
    start_time = time.time()
    np.random.seed(42)  # Fixed seed for reproducibility
    
    # ========= 1. Generate simulation data =========
    logger.info("Generating simulation data...")
    
    # Create landmarks (5 randomly distributed landmarks)
    landmarks_data = landmarks(
        num=10, 
        distribution="gaussian", 
        sigma=[3, 3]  # Standard deviation 7 meters
    )
    
    # Create trajectory (1 robot, 50 poses, figure-8 motion)
    trajectory = paths(
        R=1, 
        T=50,
        motion_type='figure8',
        center=(0, 0),
        radius=10,
        angular_velocity=0.1,
        linear_velocity=0.15,
        noise_std=np.array([0.05, 0.05, 0.01])  # Position noise 5cm, angle noise 0.01 rad
    )
    
    # Ensure trajectory is in 3D format (1, T, 3)
    if trajectory.ndim == 2:
        trajectory = trajectory[None, ...]
    
    # Generate measurements (robot to landmark)
    logger.info("Generating measurements...")
    measurements = make_measurements(
        trajectory, 
        landmarks_data,
        sigma_r=0.3,        # Range noise std 0.3m
        sigma_b=np.deg2rad(2),  # Bearing noise std 2 degrees
        max_range=25.0,      # Max observation range 25m
        p_obs=0.7           # 70% observation probability
    )
    
    logger.info(f"Generated measurements: {len(measurements)}")
    
    # ========= 2. Build Factor Graph =========
    logger.info("Building factor graph...")
    
    # Use dictionary configuration
    builder_config = {
        "ut_mode": "auto",
        "ut_alpha": 0.1,
        "ut_beta": 2.0,
        "ut_kappa": 0.0,
        "odom_sigma": [0.05, 0.05, 0.01]  # Match trajectory noise
    }
    
    builder = GBPGraphBuilder(builder_config)
    factors, variables = builder.build(trajectory, landmarks_data, measurements)
    logger.info(f"Graph built: {len(factors)} factors, {len(variables)} variables")
    
    # Show factor type distribution
    if factors:
        factor_types = collections.Counter(type(f).__name__ for f in factors)
        logger.info(f"Factor types: {factor_types}")
    else:
        logger.error("No factors generated! Check measurement processing.")
        return
    
    # ========= 3. Initialize GBP =========
    logger.info("Initializing GBP graph...")
    graph = GBPGraph(damping=0.3)
    
    # Add all variables with initial noise
    for key, init in variables.items():
        # Anchor first pose with strong prior
        if key == "x0_0":
            graph.add_variable(
                key, 
                dim=3, 
                prior=init,
                weight=np.array([1e6, 1e6, 1e5])  # High precision for anchor
            )
        # Pose variables
        elif key.startswith('x'):
            noise_pos = init[:2] + np.random.normal(0, 0.5, 2)
            noise_angle = wrap_angle(init[2] + np.random.normal(0, np.deg2rad(15)))
            graph.add_variable(
                key, 
                dim=3, 
                prior=np.array([*noise_pos, noise_angle]),
                weight=1e-4  # Medium precision
            )
        # Landmark variables
        else:
            noise_pos = init + np.random.normal(0, 0.5, 2)
            graph.add_variable(
                key, 
                dim=2, 
                prior=noise_pos,
                weight=1e-4  # Medium precision
            )
    
    # Add all factors to graph
    for factor in factors:
        graph.add_factor(factor)
    
    # ========= 4. Optimization =========
    logger.info("Starting optimization...")
    energy_log = []
    
    # Run in stages for better convergence
    stage_iterations = [10, 10, 10]
    for stage_iters in stage_iterations:
        stage_energy = graph.solve(max_iter=stage_iters, verbose=False)
        energy_log.extend(stage_energy)
        
        logger.info(f"Iteration {len(energy_log)} - Energy: {energy_log[-1]:.3e}")
        
        # Basic convergence check
        if len(energy_log) > 5 and (energy_log[-1] < 1e-3 or 
                                     abs(energy_log[-1] - energy_log[-5])/energy_log[-5] < 1e-4):
            logger.info("Convergence detected")
            break
    
    if energy_log:
        logger.info(f"Optimization completed: {len(energy_log)} iterations, Final energy: {energy_log[-1]:.3e}")
    else:
        logger.error("Optimization failed to produce energy logs")
    
    # ========= 5. Results Extraction =========
    T = trajectory.shape[1]
    
    # Estimated trajectory
    traj_est = np.zeros((T, 3))
    for t in range(T):
        key = f"x0_{t}"
        if key in graph.nodes:
            traj_est[t] = graph.get_mean(key)
        else:
            traj_est[t] = trajectory[0, t]  # Fallback to ground truth
    
    # Estimated landmarks
    lms_est = np.zeros_like(landmarks_data)
    for i in range(landmarks_data.shape[0]):
        key = f"l_{i}"
        if key in graph.nodes:
            lms_est[i] = graph.get_mean(key)
        else:
            lms_est[i] = landmarks_data[i]  # Fallback to ground truth
    
    # ========= 6. Performance Evaluation =========
    # Position error
    xy_err = np.linalg.norm(trajectory[0, :, :2] - traj_est[:, :2], axis=1)
    pos_rmse = np.sqrt(np.mean(xy_err**2))
    pos_mae = np.mean(np.abs(xy_err))
    
    # Angle error
    ang_err = wrap_angle(trajectory[0, :, 2] - traj_est[:, 2])
    ang_mae = np.mean(np.abs(ang_err))
    ang_rmse = np.sqrt(np.mean(ang_err**2))
    
    # Measurement statistics
    meas_dists = []
    for m in measurements:
        if m["type"] == "robot_lm":
            t = m["time"]
            lm_id = m["id"]
            robot_pos = trajectory[0, t, :2]
            lm_pos = landmarks_data[lm_id]
            meas_dists.append(np.linalg.norm(robot_pos - lm_pos))
    
    avg_meas_dist = np.mean(meas_dists) if meas_dists else 0
    
    # Time metrics
    total_time = time.time() - start_time
    iters_per_sec = len(energy_log) / total_time if total_time > 0 else 0
    
    # ========= 7. Visualization =========
    plt.figure(figsize=(16, 6))
    plt.suptitle(f"GBP SLAM Results | Time: {total_time:.2f}s | Iter/s: {iters_per_sec:.1f}", 
                 fontsize=14, fontfamily='DejaVu Sans')
    
    # Subplot 1: Trajectory and landmarks
    ax1 = plt.subplot(131, aspect='equal')
    ax1.set_title("Trajectory & Landmarks", fontfamily='DejaVu Sans')
    
    # Ground truth trajectory
    ax1.plot(trajectory[0, :, 0], trajectory[0, :, 1], 
             'b-', linewidth=2, alpha=0.7, label='GT Trajectory')
    
    # Estimated trajectory
    ax1.plot(traj_est[:, 0], traj_est[:, 1], 
             'r--', linewidth=2, alpha=0.8, label='Estimated Trajectory')
    
    # Landmarks
    ax1.scatter(landmarks_data[:, 0], landmarks_data[:, 1], 
                c='k', marker='*', s=150, label='GT Landmarks')
    ax1.scatter(lms_est[:, 0], lms_est[:, 1], 
                c='r', marker='x', s=100, label='Estimated Landmarks')
    
    # Add covariance ellipses
    for i in range(landmarks_data.shape[0]):
        key = f"l_{i}"
        if key in graph.nodes:
            try:
                cov = graph.get_cov(key)
                plot_cov_ellipse(ax1, lms_est[i], cov, n_std=2, 
                                edgecolor='red', alpha=0.3, linestyle='-')
            except Exception as e:
                logger.warning(f"Covariance for landmark {i} failed: {str(e)}")
    
    # Start point
    ax1.scatter(traj_est[0, 0], traj_est[0, 1], 
                s=120, c='g', marker='o', label='Start Point')
    
    ax1.set_xlabel("X (m)", fontfamily='DejaVu Sans')
    ax1.set_ylabel("Y (m)", fontfamily='DejaVu Sans')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='best', fontsize=9, prop={'family': 'DejaVu Sans'})
    
    # Subplot 2: Position error
    ax2 = plt.subplot(132)
    ax2.set_title("Position Error", fontfamily='DejaVu Sans')
    
    if xy_err.size > 0:
        # Position error plot
        ax2.plot(xy_err, 'b-', label='Position Error (m)')
        ax2.fill_between(range(len(xy_err)), xy_err, alpha=0.2, color='blue')
        
        # Angle error plot (right axis)
        ax2r = ax2.twinx()
        ax2r.plot(np.degrees(ang_err), 'r-', alpha=0.7, label='Angle Error (°)')
        
        # Annotate max error
        max_idx = np.argmax(xy_err)
        if max_idx < len(xy_err):
            ax2.annotate(f'Max: {xy_err[max_idx]:.2f}m', 
                         xy=(max_idx, xy_err[max_idx]),
                         xytext=(max_idx+2, xy_err[max_idx]*1.2),
                         arrowprops=dict(arrowstyle='->', color='blue'))
        
        ax2.set_xlabel("Pose Index", fontfamily='DejaVu Sans')
        ax2.set_ylabel("Position Error (m)", color='b', fontfamily='DejaVu Sans')
        ax2r.set_ylabel("Angle Error (°)", color='r', fontfamily='DejaVu Sans')
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        # Combine legends
        lines, labels = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2r.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right', prop={'family': 'DejaVu Sans'})
    
    # Subplot 3: Energy convergence
    ax3 = plt.subplot(133)
    ax3.set_title("Energy Convergence", fontfamily='DejaVu Sans')
    
    if energy_log:
        ax3.semilogy(energy_log, 'g-', linewidth=2, label='Total Energy')
        ax3.set_xlabel("Iteration", fontfamily='DejaVu Sans')
        ax3.set_ylabel("Energy (log scale)", fontfamily='DejaVu Sans')
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.legend(prop={'family': 'DejaVu Sans'})
    
    # Add performance summary
    summary_text = (
        f"Iterations: {len(energy_log)}\n"
        f"Final Energy: {energy_log[-1]:.2e}\n"
        f"Position RMSE: {pos_rmse:.3f} m\n"
        f"Position MAE: {pos_mae:.3f} m\n"
        f"Angle MAE: {np.degrees(ang_mae):.2f} °\n"
        f"Avg Meas Dist: {avg_meas_dist:.1f} m"
    )
    ax1.text(0.05, 0.95, summary_text, transform=ax1.transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             verticalalignment='top', fontsize=10, fontfamily='DejaVu Sans')
    
    # Save and show results
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("slam_results.png", dpi=150, bbox_inches='tight')
    logger.info("Results saved to slam_results.png")
    
    # ========= 8. Final Performance Report =========
    logger.info("=" * 60)
    logger.info("SLAM PERFORMANCE REPORT")
    logger.info(f"  - Total time: {total_time:.2f} seconds")
    logger.info(f"  - Iterations: {len(energy_log)}")
    logger.info(f"  - Iterations per second: {iters_per_sec:.1f}")
    logger.info(f"  - Final energy: {energy_log[-1]:.3e}")
    logger.info("  - Position accuracy:")
    logger.info(f"      RMSE: {pos_rmse:.4f} m")
    logger.info(f"      MAE: {pos_mae:.4f} m")
    logger.info("  - Orientation accuracy:")
    logger.info(f"      MAE: {np.degrees(ang_mae):.2f} °")
    logger.info(f"      RMSE: {np.degrees(ang_rmse):.2f} °")
    logger.info("  - Observation statistics:")
    logger.info(f"      Measurements: {len(measurements)}")
    logger.info(f"      Avg measurement distance: {avg_meas_dist:.1f} m")
    logger.info("=" * 60)
    
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)