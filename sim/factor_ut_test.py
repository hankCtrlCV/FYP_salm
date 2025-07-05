import numpy as np
from algorithm.frontend.factor_ut import PriorFactor, OdometryFactor

def test_prior_energy_zero():
    prior = np.array([1.,2.,.3])
    pf = PriorFactor("x0", prior, 1e-3)
    assert abs(pf.get_energy({"x0": prior.copy()})) < 1e-9

def test_odometry_energy_zero():
    p1 = np.array([0.,0.,0.])
    p2 = np.array([1.,0.,0.])
    delta = OdometryFactor._se2_relative_pose(p1, p2)
    of = OdometryFactor("x0","x1", delta, 1e-3)
    assert abs(of.get_energy({"x0":p1,"x1":p2})) < 1e-8
