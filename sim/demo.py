"""
One-click demo for 1-robot-2-landmark GBP + ADMM = identity (trivial).
"""
import matplotlib.pyplot as plt
from algorithm.frontend.gbp import GBPGraph
from algorithm.backend.admm import ADMMLocal
from sim import world, measurement, graph_builder
import numpy as np

# 1. world & measurements
lms_gt = world.landmarks()
paths_gt = world.paths(R=1)
meas = measurement.make_measurements(paths_gt, lms_gt)

# 2. initial guess
paths_init = paths_gt + np.random.randn(*paths_gt.shape)*0.5
lms_init   = lms_gt + np.random.randn(*lms_gt.shape)*1.0

# 3. build factor graph
R_noise = np.diag([measurement.SIG_R**2, measurement.SIG_B**2])
factors, initial = graph_builder.build_graph(0, paths_init, lms_init, meas, R_noise)

# 4. run GBP
g = GBPGraph()
for f in factors: g.add_factor(f)
for k, μ0 in initial.items(): g.add_variable(k, dim=2, prior=μ0, weight=1e-6)

for _ in range(10): g.iterate_once()

# 5. extract estimates
est_pose = np.vstack([g.get_mean(f"x0_{t}") for t in range(paths_gt.shape[1])])
est_lms  = np.vstack([g.get_mean(f"l{j}")  for j in range(lms_gt.shape[0])])

# 6. visualize result (quick)
plt.figure(figsize=(5,5)); ax=plt.gca(); ax.set_aspect('equal')
ax.plot(*paths_gt[0].T, 'k--', label='GT')
ax.plot(*paths_init[0].T, 'r:', label='Init')
ax.plot(*est_pose.T, 'g-', label='GBP')
ax.scatter(*lms_gt.T, c='k', marker='x')
ax.scatter(*est_lms.T, c='g')
ax.legend(); plt.show()
