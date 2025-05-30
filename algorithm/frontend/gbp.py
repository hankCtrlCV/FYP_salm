# --------------------------------------------------------
# Gaussian Belief Propagation – stable version
# --------------------------------------------------------
from __future__ import annotations
import numpy as np, logging, inspect
from typing import Dict, List, Any, Tuple, Optional
from algorithm.frontend.factor_ut import BearingRangeUTFactor   # 非线性 σ-点因子

_log  = logging.getLogger("GBP-Solver")
_EPS  = 1e-9     # 全局数值保护

# ========================================================
# 1. 内部节点类
# ========================================================
class _Node:
    def __init__(self, dim:int,
                 prior_mean:Optional[np.ndarray],
                 prior_sigma:Optional[float]):
        self.dim = dim
        self.L   = np.zeros((dim, dim))          # 信息矩阵 Λ
        self.eta = np.zeros(dim)                 # 信息向量 η

        if prior_mean is not None and prior_sigma is not None:
            Λ0 = np.eye(dim) / (prior_sigma**2)
            self.L   += Λ0
            self.eta += Λ0 @ prior_mean

        self._prior_L   = self.L.copy()
        self._prior_eta = self.eta.copy()

    # --- 基本操作 ----------------------------------------------------------
    def reset(self):
        self.L[:]   = self._prior_L
        self.eta[:] = self._prior_eta

    def add_info(self, Λ, η):
        self.L   += Λ
        self.eta += η

    def mean_cov(self) -> Tuple[np.ndarray, np.ndarray]:
        Σ = np.linalg.inv(self.L + _EPS*np.eye(self.dim))
        μ = Σ @ self.eta
        return μ, Σ

    # --- 先验能量（用于总能量计算） -----------------------------------------
    def get_prior_energy(self) -> float:
        if np.allclose(self._prior_L, 0):
            return 0.0
        μ, _ = self.mean_cov()
        μ0   = np.linalg.solve(self._prior_L, self._prior_eta)
        r    = μ - μ0
        return 0.5 * r @ self._prior_L @ r


# ========================================================
# 2. GBP 图类
# ========================================================
class GBPGraph:
    """
    *   add_variable(key, dim=2, prior=None, weight=σ⁻²)
    *   add_factor(factor)         # factor 必须实现 linearize()
    *   solve(n_iters)
    """

    # ---------- 初始化 ----------
    def __init__(self, damping: float = 0.25):
        self.nodes  : Dict[str, _Node] = {}
        self.factors: List[Any]        = []
        self.damping = damping

        # 运行时缓存
        self.current_mu : Dict[str, np.ndarray] = {}
        self.cov        : Dict[Any , np.ndarray] = {}
        self._init_done = False                 # 首轮初始化标记

    # ---------- 变量 / 因子 ----------
    def add_variable(self, key:str, *, dim:int = 2,
                     prior:Optional[np.ndarray] = None,
                     weight:float = 0.0):
        if key in self.nodes:
            return
        sigma = None if weight == 0 else 1/np.sqrt(weight)
        self.nodes[key] = _Node(dim, prior, sigma)

    def add_factor(self, fac:Any):
        self.factors.append(fac)

    # ====================================================
    # 3. 单次同步迭代
    # ====================================================
    def iterate_once(self):

        # ---------- (0) 第一次：用先验初始化 μ, Σ ----------
        if not self._init_done:
            self.current_mu = {k: n.mean_cov()[0] for k, n in self.nodes.items()}
            self.cov        = {k: n.mean_cov()[1] for k, n in self.nodes.items()}
            # 给所有跨变量配上一个极小 cross-cov，防止奇异
            tiny = 1e-9
            keys = list(self.nodes.keys())
            for i, ki in enumerate(keys):
                di = self.nodes[ki].dim
                for kj in keys[i+1:]:
                    dj = self.nodes[kj].dim
                    self.cov[(ki, kj)] = tiny*np.zeros((di, dj))
            self._init_done = True

        # ---------- (1) reset → 累加自环块 ----------
        for n in self.nodes.values():
            n.reset()

        cross_blocks : List[Tuple[str,str,np.ndarray]] = []   # (key_i, key_j, Λ_ij)

        for fac in self.factors:
            local = fac.linearize(self.current_mu, self.cov)  # {key:(Λ,η)} or {(ki,kj):Λ}
            for k, v in local.items():
                if isinstance(k, tuple):                      # 交叉项
                    Λij = 0.5*(v+v.T) + 1e-9*np.eye(v.shape[0])
                    cross_blocks.append((k[0], k[1], Λij))
                else:                                         # 自环项
                    Λ, η = v
                    Λ = 0.5*(Λ+Λ.T)
                    Λ = (1-self.damping)*Λ + self.damping*1e-6*np.eye(Λ.shape[0])
                    η = (1-self.damping)*η
                    self.nodes[k].add_info(Λ, η)

        # ---------- (2) 组装联合信息矩阵 ----------
        order  = list(self.nodes.keys())
        dims   = [self.nodes[k].dim for k in order]
        starts = np.cumsum([0] + dims[:-1])
        J      = sum(dims)

        Λ_joint = np.zeros((J, J))
        η_joint = np.zeros(J)

        for k, s, d in zip(order, starts, dims):
            Λ_joint[s:s+d, s:s+d] = self.nodes[k].L
            η_joint[s:s+d]        = self.nodes[k].eta

        for ki, kj, Λij in cross_blocks:
            if ki in order and kj in order:
                i, j   = order.index(ki), order.index(kj)
                si, di = starts[i], dims[i]
                sj, dj = starts[j], dims[j]
                Λ_joint[si:si+di, sj:sj+dj] += Λij
                Λ_joint[sj:sj+dj, si:si+di] += Λij.T

        # ---------- (3) 计算联合边缘 ----------
        Σ_joint = np.linalg.inv(Λ_joint + 1e-6*np.eye(J))
        μ_joint = Σ_joint @ η_joint

        # 可选：截断过大的方差
        Σ_diag = np.clip(np.diag(Σ_joint), None, 1e4)
        Σ_joint[np.diag_indices_from(Σ_joint)] = Σ_diag

        # ---------- (4) 拆回局部 μ, Σ ----------
        self.current_mu.clear()
        self.cov.clear()

        for k, s, d in zip(order, starts, dims):
            self.current_mu[k] = μ_joint[s:s+d]
            self.cov[k]        = Σ_joint[s:s+d, s:s+d]

        for i, (ki, si, di) in enumerate(zip(order, starts, dims)):
            for j in range(i+1, len(order)):
                kj, sj, dj = order[j], starts[j], dims[j]
                self.cov[(ki, kj)] = Σ_joint[si:si+di, sj:sj+dj]

    # ====================================================
    # 4. 能量与求解
    # ====================================================
    def energy(self) -> float:
        mu_dict = {k: n.mean_cov()[0] for k, n in self.nodes.items()}
        total   = 0.0
        for fac in self.factors:
            if not hasattr(fac, "get_energy"):   # 不是所有因子都实现
                continue
            narg = len(inspect.signature(fac.get_energy).parameters)
            total += fac.get_energy() if narg == 0 else fac.get_energy(mu_dict)
        for n in self.nodes.values():
            total += n.get_prior_energy()
        return total

    def solve(self, n_iters:int = 30, *, verbose:bool = False) -> List[float]:
        energy_log: List[float] = []
        diverged = 0
        for it in range(n_iters+1):
            if not self.current_mu:          # 首轮
                self.iterate_once()

            e = self.energy()
            energy_log.append(e)
            if verbose and (it == 0 or it % 5 == 0 or it == n_iters):
                _log.info(f"Iter {it:2d}  energy {e:.3e}")

            if it == n_iters:
                break

            if it > 2 and e > max(energy_log[-3:]):
                diverged += 1
            else:
                diverged = max(0, diverged-1)

            # 反馈给非线性因子
            for f in self.factors:
                if isinstance(f, BearingRangeUTFactor):
                    f._diverged_cnt = diverged

            self.iterate_once()

        return energy_log

    # ====================================================
    # 5. 查询辅助
    # ====================================================
    def get_mean(self, key:str) -> np.ndarray:
        return self.nodes[key].mean_cov()[0]

    def get_cov(self, key:str) -> np.ndarray:
        return self.nodes[key].mean_cov()[1]
