"""
Very simple neighbour-average consensus (placeholder).
"""
import numpy as np
class Consensus:
    def __init__(self): self.buff = []

    def add(self, vec): self.buff.append(vec)
    def compute(self):
        return np.mean(self.buff, axis=0) if self.buff else None
    def reset(self): self.buff.clear()
