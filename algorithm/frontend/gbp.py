"""
Loopy Gaussian BP (synchronous) – minimal.
"""
import numpy as np
from collections import defaultdict

class GBPNode:
    def __init__(self, dim, prior=None, weight=0.0):
        self.dim = dim
        self.L   = np.eye(dim)*weight
        self.η   = np.zeros(dim)
        if prior is not None:
            self.η = self.L @ prior

    def reset(self):
        self.L[:] = 0; self.η[:] = 0

    def receive(self, Λ, η):
        self.L += Λ; self.η += η

    def mean_cov(self):
        P = np.linalg.inv(self.L + 1e-9*np.eye(self.dim))
        return P @ self.η, P

class GBPGraph:
    def __init__(self):
        self.nodes = defaultdict(lambda: None)
        self.factors = []

    # ---------- construction ----------
    def add_variable(self, key, dim=2, prior=None, weight=0.0):
        if self.nodes[key] is None:
            self.nodes[key] = GBPNode(dim, prior, weight)

    def add_factor(self, f): self.factors.append(f)

    # ---------- iteration ----------
    def iterate_once(self):
        mus, covs = {}, {}
        for k,n in self.nodes.items(): mus[k], covs[k] = n.mean_cov()
        for n in self.nodes.values(): n.reset()
        for f in self.factors:
            Λ, η = f.linearize(mus[f.pose_key], covs[f.pose_key],
                               mus[f.lm_key],   covs[f.lm_key])
            self.nodes[f.pose_key].receive(Λ, η)
            self.nodes[f.lm_key].receive(Λ, η)

    def get_mean(self, key):
        return self.nodes[key].mean_cov()[0]
