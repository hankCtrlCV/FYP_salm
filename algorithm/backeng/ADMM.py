"""
Local ADMM primal & dual update (single block).  Multi-robot demo 用得到。
"""
import numpy as np

class ADMMLocal:
    def __init__(self, dim=2, rho=1.0):
        self.dim, self.rho = dim, rho
        self.L = np.eye(dim) * 1e-3
        self.η = np.zeros(dim)
        self.x = np.zeros(dim)
        self.λ = np.zeros(dim)

    def set_local_info(self, Λ, η):
        self.L, self.η = Λ, η

    def step(self, z_consensus):
        A = self.L + self.rho*np.eye(self.dim)
        b = self.η + self.rho*(z_consensus - self.λ/self.rho)
        self.x = np.linalg.solve(A, b)
        self.λ += self.rho * (self.x - z_consensus)

    def pack(self):
        return self.x + self.λ / self.rho
