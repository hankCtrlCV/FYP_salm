# --------------------------------------------------------
# Gaussian Belief Propagation – stable version
# --------------------------------------------------------
from __future__ import annotations
import numpy as np, logging, inspect
from typing import Dict, List, Any, Tuple, Optional, Union
from algorithm.frontend.factor_ut import BearingRangeUTFactor   # 非线性 σ-点因子

_log  = logging.getLogger("GBP-Solver")
_EPS  = 1e-9     # 全局数值保护

# ========================================================
# 1. 内部节点类
# ========================================================
class _Node:
    def __init__(self, dim:int,
                 prior_mean:Optional[np.ndarray],
                 prior_sigma:Optional[Union[float, np.ndarray]]):  # 支持数组
        self.dim = dim
        self.L   = np.zeros((dim, dim))          # 信息矩阵 Λ
        self.eta = np.zeros(dim)                 # 信息向量 η

        if prior_mean is not None and prior_sigma is not None:
            # 处理数组或标量先验
            if np.isscalar(prior_sigma):
                Λ0 = np.eye(dim) / (prior_sigma**2 + _EPS)
            else:  # 数组或向量
                # 确保维度匹配
                if len(prior_sigma) != dim:
                    raise ValueError(f"先验σ维度({len(prior_sigma)})与变量维度({dim})不匹配")
                Λ0 = np.diag(1 / (np.square(prior_sigma) + _EPS))
                
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
        # 添加正则化确保数值稳定
        Λ = self.L + _EPS * np.eye(self.dim)
        try:
            Σ = np.linalg.inv(Λ)
        except np.linalg.LinAlgError:
            Σ = np.linalg.pinv(Λ)  # 如果奇异使用伪逆
            
        μ = Σ @ self.eta
        return μ, Σ

    # --- 先验能量（用于总能量计算） -----------------------------------------
    def get_prior_energy(self) -> float:
        if np.allclose(self._prior_L, 0, atol=1e-6):
            return 0.0
        μ, _ = self.mean_cov()
        μ0   = np.linalg.solve(self._prior_L + _EPS*np.eye(self.dim), self._prior_eta)
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
    def add_variable(self, name, dim, prior=None, weight=0):
        """添加变量到因子图 (修复版本)
        
        参数:
            name: 变量名称
            dim: 变量维度
            prior: 先验均值 (可选)
            weight: 权重值 (标量或数组) - 表示先验的精度 (σ⁻²)
        """
        # 确保名称唯一
        if name in self.nodes:
            raise ValueError(f"变量 '{name}' 已存在")
        
        # 处理权重值
        if not np.isscalar(weight):
            # 如果是数组，需要验证维度
            if isinstance(weight, (list, tuple, np.ndarray)):
                weight_arr = np.asarray(weight)
                if weight_arr.size != dim:
                    raise ValueError(f"权重大小({weight_arr.size})与维度({dim})不匹配")
                
                # 权重转换为精度 (σ⁻²)
                precision = weight_arr
            else:
                raise TypeError(f"权重类型不支持: {type(weight)}")
        else:
            # 标量权重
            precision = weight
            
        # 创建节点
        self.nodes[name] = _Node(dim, prior, precision)
        
        # 调试信息
        if not np.isscalar(precision) and np.any(precision > 1e3):
            max_prec = np.max(precision)
            _log.debug(f"添加变量 '{name}': 维度={dim}, 强权重={max_prec:.1e}")
        elif np.isscalar(precision) and precision > 1e3:
            _log.debug(f"添加变量 '{name}': 维度={dim}, 强权重={precision:.1e}")
    
    def add_factor(self, fac:Any):
        self.factors.append(fac)
        
    # ====================================================
    # 3. 单次同步迭代
    # ====================================================
    def iterate_once(self):
        """执行一次GBP迭代，修复 tuple 异常并统一矩阵格式"""
        # ---------- (0) 第一次：用先验初始化 μ, Σ ----------
        if not self._init_done:
            self.current_mu = {k: n.mean_cov()[0] for k, n in self.nodes.items()}
            self.cov = {}
            
            # 设置初始协方差（对角线块）
            for k, n in self.nodes.items():
                _, Σ = n.mean_cov()
                self.cov[k] = Σ
            
            # 初始化交叉协方差（微小值防奇异）
            keys = list(self.nodes.keys())
            for i, ki in enumerate(keys):
                di = self.nodes[ki].dim
                for kj in keys[i+1:]:
                    dj = self.nodes[kj].dim
                    self.cov[(ki, kj)] = 1e-9 * np.zeros((di, dj))
            
            self._init_done = True
        
        cross_blocks : List[Tuple[str,str,np.ndarray]] = []
        
        # 辅助函数：确保输入是有效矩阵
        # ==================== 交叉项处理函数优化 ====================
        def _ensure_matrix(val: Union[np.ndarray, tuple]) -> np.ndarray:
            """增强型矩阵格式处理"""
            if isinstance(val, tuple):
                # 智能识别矩阵位置：优先取第一个非None元素
                for item in val:
                    if item is not None and isinstance(item, (np.ndarray, list)):
                        return np.asarray(item, dtype=float)
                # 所有元素都是None的情况
                raise ValueError("元组中未找到有效矩阵")
            return np.asarray(val, dtype=float)
        
        def _stabilize_matrix(mat: np.ndarray) -> np.ndarray:
            """确保矩阵对称且数值稳定"""
            symmetric = 0.5 * (mat + mat.T)  # 对称化
            return symmetric + _EPS * np.eye(mat.shape[0])
        
        # ---------- (1) reset → 累加自环块 ----------
        for n in self.nodes.values():
            n.reset()
        
        # 处理每个因子
        for fac in self.factors:
            # 线性化因子，得到局部信息（可能包含自环项和交叉项）
            local = fac.linearize(self.current_mu, self.cov)
            
            for k, v in local.items():
                # ---------- 交叉项 (ki,kj) ----------
                if isinstance(k, tuple):  # 交叉项由元组键 (key_i, key_j) 标识
                    try:
                        matrix = _ensure_matrix(v)        # 统一格式处理（确保是矩阵）
                        Λij = _stabilize_matrix(matrix)   # 确保对称且正定
                        cross_blocks.append((k[0], k[1], Λij))  # 记录交叉项
                    except Exception as e:
                        _log.error(f"交叉项 {k} 处理失败: {e}\n"
                                   f"因子类型: {type(fac)}\n"
                                   f"返回值类型: {type(v)}\n"
                                   f"值: {v}")
                        continue
                
                # ---------- 自环项 (单变量) ----------
                else:
                    # 标准格式应为元组 (Λ, η)
                    if isinstance(v, tuple):
                        Λ, η = v
                    else:
                        # 兼容旧版（只返回Λ），则η设为0
                        _log.warning(f"因子 {fac} 的自环项未返回η，自动补零")
                        Λ = v
                        η = np.zeros(_ensure_matrix(v).shape[0])
                    
                    # 确保Λ是矩阵，并进行数值稳定处理
                    try:
                        Λ = _ensure_matrix(Λ)
                        Λ = _stabilize_matrix(Λ)
                    except Exception as e:
                        _log.error(f"自环项Λ处理失败: {e}\n"
                                   f"因子: {fac}, 变量: {k}")
                        continue
                    
                    # 阻尼处理 (使用实例的damping系数)
                    Λ = (1 - self.damping) * Λ + self.damping * 1e-6 * np.eye(Λ.shape[0])
                    η = (1 - self.damping) * η
                    
                    # 将信息累加到节点
                    self.nodes[k].add_info(Λ, η)
        
        # ---------- (2) 组装联合信息矩阵 ----------
        order  = list(self.nodes.keys())
        dims   = [self.nodes[k].dim for k in order]
        starts = np.cumsum([0] + dims[:-1])
        J      = sum(dims)

        Λ_joint = np.zeros((J, J))
        η_joint = np.zeros(J)

        # 填充自环块（对角线）
        for k, s, d in zip(order, starts, dims):
            Λ_joint[s:s+d, s:s+d] = self.nodes[k].L
            η_joint[s:s+d]        = self.nodes[k].eta

        # 填充交叉块（非对角线）
        for ki, kj, Λij in cross_blocks:
            if ki in order and kj in order:
                i, j   = order.index(ki), order.index(kj)
                si, di = starts[i], dims[i]
                sj, dj = starts[j], dims[j]
                Λ_joint[si:si+di, sj:sj+dj] += Λij
                Λ_joint[sj:sj+dj, si:si+di] += Λij.T  # 对称位置

        # ---------- (3) 计算联合边缘 ----------
        # 添加正则项防止奇异
        Λ_joint += 1e-6 * np.eye(J)
        try:
            Σ_joint = np.linalg.inv(Λ_joint)
        except np.linalg.LinAlgError:
            _log.error("联合信息矩阵奇异，使用伪逆")
            Σ_joint = np.linalg.pinv(Λ_joint)
        μ_joint = Σ_joint @ η_joint

        # 可选：截断过大的方差（防止数值爆炸）
        max_var = 1e4  # 最大允许方差
        diag_vars = np.diag(Σ_joint)
        if np.any(diag_vars > max_var):
            _log.warning("检测到过大方差，进行截断")
            # 只截断对角线元素
            np.fill_diagonal(Σ_joint, np.clip(diag_vars, None, max_var))

        # ---------- (4) 拆回局部 μ, Σ ----------
        self.current_mu.clear()
        self.cov.clear()

        # 更新每个变量的均值和协方差（对角线块）
        for k, s, d in zip(order, starts, dims):
            self.current_mu[k] = μ_joint[s:s+d]
            self.cov[k]        = Σ_joint[s:s+d, s:s+d]

        # 更新交叉协方差（非对角线块）
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

    # File: algorithm/frontend/gbp.py

    # ====================================================
    # 4. 能量与求解
    # ====================================================
    def solve(self, max_iter: int = 30, *, verbose: bool = False) -> List[float]:
        """优化因子图
        
        参数:
            max_iter: 最大迭代次数
            verbose: 是否打印详细日志
        """
        energy_log: List[float] = []
        diverged = 0
        
        # 优化循环
        for it in range(max_iter + 1):
            if not self.current_mu:  # 首次迭代
                self.iterate_once()
            
            # 计算当前能量
            e = self.energy()
            energy_log.append(e)
            
            # 打印日志
            if verbose and (it == 0 or it % 5 == 0 or it == max_iter):
                _log.info(f"Iter {it:2d}/{max_iter}  energy {e:.3e}")
            
            # 检测是否发散
            if it > 2 and e > max(energy_log[-3:]):
                diverged += 1
            else:
                diverged = max(0, diverged - 1)
            
            # 反馈给非线性因子（如果适用）
            for f in self.factors:
                if hasattr(f, "_diverged_cnt"):
                    f._diverged_cnt = diverged
            
            # 正常迭代
            if it < max_iter:
                self.iterate_once()
        
        return energy_log

    # ====================================================
    # 5. 查询辅助
    # ====================================================
    def get_mean(self, key:str) -> np.ndarray:
        return self.nodes[key].mean_cov()[0]

    def get_cov(self, key:str) -> np.ndarray:
        return self.nodes[key].mean_cov()[1]
