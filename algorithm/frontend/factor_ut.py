"""
Bearing-Range Factor with Unscented linearization.
"""
import numpy as np
from algorithm.math.sigma import sigma_points, UTParam

class BearingRangeUTFactor:
    def __init__(self, pose_key, lm_key, z, R):
        self.pose_key, self.lm_key, self.z, self.R = pose_key, lm_key, z, R

    def linearize(self, mu_pose, P_pose, mu_lm, P_lm):
        # TODO: UT 计算 Λ, η
        raise NotImplementedError
