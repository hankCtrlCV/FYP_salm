"""
Generate ground-truth world:
  • landmarks() → ndarray (L,2)
  • paths(R,T)  → ndarray (R,T,2)
"""
import numpy as np

def landmarks(n=2, lim=20.0) -> np.ndarray:
    """Uniformly sample landmark xy positions."""
    return np.random.uniform(-lim, lim, size=(n, 2))

def paths(R=1, T=50, radius=15.0, omega=0.15) -> np.ndarray:
    """Circular trajectories for each robot."""
    t = np.arange(T)
    xy = np.stack([radius * np.cos(omega * t),
                   radius * np.sin(omega * t)], axis=1)
    return np.repeat(xy[None, ...], R, axis=0)  # (R,T,2)
