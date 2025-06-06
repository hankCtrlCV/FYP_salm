# --------------------------------------------------------
# Gaussian Belief Propagation – stable version
# --------------------------------------------------------
from __future__ import annotations
import numpy as np, logging, inspect
from typing import Dict, List, Any, Tuple, Optional, Union

_log  = logging.getLogger("GBP-Solver")
_EPS  = 1e-9     # 全局数值保护

# ========================================================
# 1. 内部节点类
# ========================================================
class _Node:
    def __init__(self, dim: int,
                 prior_mean: Optional[np.ndarray],
                 prior_sigma: Optional[Union[float, np.ndarray]]):
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
        self.cov        : Dict[Any, np.ndarray] = {}
        self._init_done = False                 # 首轮初始化标记
        self._variable_dims = {}                # 存储变量维度

    # ---------- 变量 / 因子 ----------
    def add_variable(self, name: str, dim: int, prior: Optional[np.ndarray] = None, weight: Union[float, np.ndarray] = 0):
        """添加变量到因子图
        
        参数:
            name: 变量名称
            dim: 变量维度
            prior: 先验均值 (可选)
            weight: 权重值 (标量或数组) - 表示先验的精度 (σ⁻²)
        """
        # 确保名称唯一
        if name in self.nodes:
            raise ValueError(f"变量 '{name}' 已存在")
        
        # 存储变量维度
        self._variable_dims[name] = dim
        
        # 创建节点
        self.nodes[name] = _Node(dim, prior, weight)
        
        # 调试信息
        if isinstance(weight, np.ndarray) and np.any(weight > 1e3):
            max_prec = np.max(weight)
            _log.debug(f"添加变量 '{name}': 维度={dim}, 强权重={max_prec:.1e}")
        elif np.isscalar(weight) and weight > 1e3:
            _log.debug(f"添加变量 '{name}': 维度={dim}, 强权重={weight:.1e}")

    def add_factor(self, fac: Any):
        self.factors.append(fac)
        
    # ====================================================
    # 3. 单次同步迭代
    # ====================================================
    def iterate_once(self):
        """执行一次GBP迭代，修复交叉项维度问题"""
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
        
        cross_blocks : List[Tuple[str, str, np.ndarray]] = []
        
        # ==================== 改进的交叉项处理 ====================
        def _validate_and_fix_cross_block(ki: str, kj: str, block_val: Any) -> Optional[np.ndarray]:
            """验证和修复交叉项维度"""
            try:
                # 获取期望的维度
                di = self._variable_dims.get(ki, self.nodes[ki].dim if ki in self.nodes else 0)
                dj = self._variable_dims.get(kj, self.nodes[kj].dim if kj in self.nodes else 0)
                
                # 处理各种可能的输入类型
                if isinstance(block_val, tuple):
                    # 优先使用第一个元素作为矩阵
                    matrix = np.asarray(block_val[0], dtype=float) if block_val[0] is not None else None
                    if matrix is None and len(block_val) > 1:
                        matrix = np.asarray(block_val[1], dtype=float) if block_val[1] is not None else None
                else:
                    matrix = np.asarray(block_val, dtype=float)
                
                if matrix is None:
                    raise ValueError("交叉项值为None")
                
                # 检查维度
                if matrix.shape == (di, dj):
                    # 维度正确
                    return matrix
                elif matrix.shape == (dj, di):
                    # 维度反转，自动转置
                    _log.debug(f"自动转置交叉项 ({ki}, {kj}): {matrix.shape} -> ({di}, {dj})")
                    return matrix.T
                else:
                    # 维度不匹配，尝试修复
                    if di > 0 and dj > 0 and matrix.size == di * dj:
                        # 尝试重塑
                        return matrix.reshape((di, dj))
                    else:
                        raise ValueError(f"形状不匹配: 期望({di},{dj}), 实际{matrix.shape}")
                
            except Exception as e:
                _log.error(f"交叉项 ({ki}, {kj}) 处理失败: {e}\n"
                           f"值类型: {type(block_val)}\n值内容: {block_val}")
                return None
        
        # ---------- (1) reset → 累加自环块 ----------
        for n in self.nodes.values():
            n.reset()
        
        # 处理每个因子
        for fac in self.factors:
            try:
                # 线性化因子
                local = fac.linearize(self.current_mu, self.cov)
                
                for k, v in local.items():
                    # ---------- 交叉项 (ki,kj) ----------
                    if isinstance(k, tuple) and len(k) == 2:
                        ki, kj = k
                        if ki not in self._variable_dims or kj not in self._variable_dims:
                            _log.warning(f"交叉项 {k} 包含未知变量, 跳过")
                            continue
                            
                        fixed_block = _validate_and_fix_cross_block(ki, kj, v)
                        if fixed_block is not None:
                            # 直接使用验证后的交叉项块
                            cross_blocks.append((ki, kj, fixed_block))
                    
                    # ---------- 自环项 (单变量) ----------
                    elif k in self.nodes:
                        if isinstance(v, tuple):
                            Λ, η = v
                        else:
                            # 兼容旧版（只返回Λ），则η设为0
                            Λ = v
                            η = np.zeros(self._variable_dims[k])
                        
                        # 确保有效矩阵
                        Λ = np.asarray(Λ, dtype=float)
                        if Λ.shape != (self._variable_dims[k], self._variable_dims[k]):
                            # 尝试重塑
                            if Λ.size == self._variable_dims[k] ** 2:
                                Λ = Λ.reshape((self._variable_dims[k], self._variable_dims[k]))
                            else:
                                _log.error(f"自环项Λ维度错误: {Λ.shape}, 期望({self._variable_dims[k]},{self._variable_dims[k]})")
                                continue
                        
                        # 数值稳定处理（仅方阵）
                        Λ = 0.5 * (Λ + Λ.T)  # 对称化
                        Λ += _EPS * np.eye(Λ.shape[0])  # 添加小对角项
                        
                        # 阻尼处理
                        damped_Λ = (1 - self.damping) * Λ + self.damping * _EPS * np.eye(Λ.shape[0])
                        damped_η = (1 - self.damping) * np.asarray(η, dtype=float)
                        
                        # 将信息累加到节点
                        self.nodes[k].add_info(damped_Λ, damped_η)
            except Exception as e:
                _log.error(f"因子处理失败: {type(fac).__name__}, 错误: {e}", exc_info=True)

        # ---------- (2) 组装联合信息矩阵 ----------
        order  = list(self.nodes.keys())
        dims   = [self.nodes[k].dim for k in order]
        starts = np.cumsum([0] + dims[:-1])
        J      = sum(dims)

        # 创建大矩阵
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
                
                # 只添加到下三角部分（避免重复）
                if i < j:
                    Λ_joint[si:si+di, sj:sj+dj] += Λij
                    Λ_joint[sj:sj+dj, si:si+di] += Λij.T

        # ---------- (3) 计算联合边缘 ----------
        # 添加正则项防止奇异
        Λ_joint += 1e-6 * np.eye(J)
        try:
            # 确保信息矩阵对称
            Λ_joint_sym = 0.5 * (Λ_joint + Λ_joint.T)
            Σ_joint = np.linalg.inv(Λ_joint_sym)
        except np.linalg.LinAlgError:
            _log.error("联合信息矩阵奇异，使用伪逆")
            Σ_joint = np.linalg.pinv(Λ_joint)
        μ_joint = Σ_joint @ η_joint

        # 截断过大的方差（防止数值爆炸）
        diag_vars = np.diag(Σ_joint)
        if np.any(diag_vars > 1e6):
            _log.warning("检测到过大方差，进行截断")
            max_var = np.median(diag_vars) * 100  # 中位数的100倍
            np.fill_diagonal(Σ_joint, np.clip(diag_vars, None, max_var))

        # ---------- (4) 拆回局部 μ, Σ ----------
        self.current_mu.clear()
        self.cov.clear()

        # 更新每个变量的均值和协方差（对角线块）
        for k, s, d in zip(order, starts, dims):
            self.current_mu[k] = μ_joint[s:s+d].copy()
            self.cov[k]        = Σ_joint[s:s+d, s:s+d].copy()

        # 更新交叉协方差（非对角线块）
        for i, (ki, si, di) in enumerate(zip(order, starts, dims)):
            for j in range(i+1, len(order)):
                kj, sj, dj = order[j], starts[j], dims[j]
                self.cov[(ki, kj)] = Σ_joint[si:si+di, sj:sj+dj].copy()

    # ====================================================
    # 4. 能量与求解
    # ====================================================
    def energy(self) -> float:
        mu_dict = {k: n.mean_cov()[0] for k, n in self.nodes.items()}
        total   = 0.0
        for fac in self.factors:
            try:
                if hasattr(fac, "get_energy"):
                    sig = inspect.signature(fac.get_energy)
                    if 'mu' in sig.parameters:
                        total += fac.get_energy(mu_dict)
                    else:
                        total += fac.get_energy()
            except Exception as e:
                _log.warning(f"因子 {type(fac).__name__} 能量计算失败: {e}")
                
        for n in self.nodes.values():
            total += n.get_prior_energy()
        return total

    # ====================================================
    # 5. 求解主循环
    # ====================================================
    def solve(self, max_iter: int = 30, *, verbose: bool = False) -> List[float]:
        """优化因子图
        
        参数:
            max_iter: 最大迭代次数
            verbose: 是否打印详细日志
        """
        energy_log: List[float] = []
        diverged_cnt = 0
        prev_e = float('inf')
        
        # 初始迭代（确保状态初始化）
        if not self._init_done or not self.current_mu:
            self.iterate_once()
        
        # 优化循环
        for it in range(max_iter + 1):
            # 执行迭代（除了首次迭代已执行）
            if it > 0:
                self.iterate_once()
            
            # 计算当前能量
            try:
                e = self.energy()
            except Exception as exc:
                _log.error(f"能量计算失败: {exc}")
                # 如果第一次迭代失败，尝试重新初始化
                if it == 0:
                    self.iterate_once()
                    e = self.energy()
                else:
                    e = float('inf')
                    
            energy_log.append(e)
            
            # 计算能量变化
            if it > 0:
                delta = e - prev_e
                sign = '+' if delta >= 0 else ''
            else:
                delta = 0
                sign = ''
            
            # 打印日志
            if verbose:
                log_entry = f"Iter {it:2d}/{max_iter}  energy {e:.3e}  Δ: {sign}{delta:.2e}"
                if it == 0 or it % 5 == 0 or it == max_iter:
                    _log.info(log_entry)
                elif delta > 0:  # 仅对能量增加显示警告
                    _log.debug(f"{log_entry} (警告: 能量增加)")
            
            # 检测是否发散 (连续3次能量增加)
            if it >= 2 and e > max(energy_log[-3:-1]):
                diverged_cnt += 1
                if verbose:
                    _log.warning(f"可能发散: 连续 {diverged_cnt} 次能量上升")
                if diverged_cnt >= 3:
                    _log.warning("发散次数过多，终止优化")
                    break
            else:
                diverged_cnt = max(0, diverged_cnt - 1)
            
            # 更新前次能量
            prev_e = e
            
            # 反馈发散计数给非线性因子
            for f in self.factors:
                if hasattr(f, "_diverged_cnt"):
                    f._diverged_cnt = diverged_cnt
            
        return energy_log

    # ====================================================
    # 6. 查询辅助
    # ====================================================
    def get_mean(self, key: str) -> Optional[np.ndarray]:
        """获取变量均值"""
        if key in self.current_mu:
            return self.current_mu[key].copy()
        elif key in self.nodes:
            return self.nodes[key].mean_cov()[0]
        _log.warning(f"未知变量 {key}")
        return None

    def get_cov(self, key: str) -> Optional[np.ndarray]:
        """获取变量协方差"""
        if key in self.cov:
            return self.cov[key].copy()
        elif key in self.nodes:
            return self.nodes[key].mean_cov()[1]
        _log.warning(f"未知变量 {key}")
        return None