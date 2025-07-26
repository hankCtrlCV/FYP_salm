"""
Enhanced GBP Factor Graph Builder for Multi-Robot SLAM
Date: 2025-07-07

"""

from __future__ import annotations
import math
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict
import warnings
import time
from dataclasses import dataclass
import pathlib
import yaml
from dataclasses import asdict  
from algorithm.frontend.gbp import make_pd, _MIN_PREC

# ✅ 修复1: Import fallback处理
try:
    from utils.cfg_loader import load_gbp
except ImportError:
    def load_gbp():
        """Fallback implementation when utils.cfg_loader is not available"""
        return {}


from algorithm.frontend.factor_ut import (
    # Core factor classes
    PriorFactor, OdometryFactor, BearingRangeUTFactor, LoopClosureFactor,
    PoseToPoseUTFactor, performance_config, ThreadSafeCache,
    
    # Utility functions  
    wrap_angle, ensure_positive_definite,
    validate_matrix_properties,
    
    # SPBP utilities
    validate_spbp_dimensions, safe_spbp_information_conversion,
    
    # Global cache and configuration system
    numerical_config, configure_numerical_parameters,
    _factor_cache_manager, get_global_cache_stats, clear_all_caches,
    
    # Multi-robot utilities
    create_multi_robot_odometry_chain, create_landmark_observation_factors,
    create_inter_robot_factors, create_pose_to_pose_ut_factors
)

logger = logging.getLogger("GBPBuilder")
_log = logger  


#------------------------------------------------------
# ✅ 修复2: _flatten_yaml 增加覆盖提示
def _flatten_yaml(d: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    把多层 YAML dict 打平成一层 **并丢掉中间层名字**，
    {'noise': {'obs_sigma_range':0.1}} → {'obs_sigma_range':0.1}
    若同名键冲突，后出现者覆盖先出现者（常见用法）。
    """
    flat = {}
    for k, v in d.items():
        if isinstance(v, dict):
            nested_flat = _flatten_yaml(v, verbose)
            for nested_key, nested_val in nested_flat.items():
                if verbose and nested_key in flat:
                    logger.info(f"YAML config override: {nested_key} = {nested_val} (was {flat[nested_key]})")
                flat[nested_key] = nested_val
        else:
            if verbose and k in flat:
                logger.info(f"YAML config override: {k} = {v} (was {flat[k]})")
            flat[k] = v
    return flat

#------------------------------------------------------
class GBPFactorAdapter:
    """
    Adapter for factor_ut.* factors to GBPGraph interface.
    Ensures unified variable order, dimension, and linearization format.
    """

    def __init__(self, raw_factor):
        self._f = raw_factor
        # *核心：变量顺序确定且唯一*
        self._var_keys = self._extract_variable_keys()
        _log.debug(f"Init GBPFactorAdapter: {self._f}, extracted keys: {self._var_keys}")
        print(f"[DEBUG] Init factor {type(self._f)} with var_keys={self._var_keys}")

    def _extract_variable_keys(self):
        """提取变量键，确保顺序一致"""
        # 方法1：直接包含var_keys
        if hasattr(self._f, 'var_keys') and self._f.var_keys:
            return list(self._f.var_keys)
            
        # 方法2：通过常见命名模式提取
        for attr_set in [('var',), ('pose_key', 'lm_key'), ('pose1_key', 'pose2_key'), ('v1', 'v2')]:
            # 检查所有属性是否存在
            if all(hasattr(self._f, attr) for attr in attr_set):
                # 只有当所有值都有效时才返回
                values = [getattr(self._f, attr) for attr in attr_set]
                if all(isinstance(v, str) for v in values):
                    return values
        
        # 方法3：OdometryFactor特殊处理
        if isinstance(self._f, OdometryFactor) and hasattr(self._f, 'v1') and hasattr(self._f, 'v2'):
            return [self._f.v1, self._f.v2]
            
        # 方法4：BearingRangeUTFactor特殊处理
        if isinstance(self._f, BearingRangeUTFactor) and hasattr(self._f, 'pose_key') and hasattr(self._f, 'lm_key'):
            return [self._f.pose_key, self._f.lm_key]
        
        # 方法5：从其他常见属性猜测
        keys = []
        for attr in ['pose_var', 'lm_var', 'var1', 'var2']:
            val = getattr(self._f, attr, None)
            if isinstance(val, str):
                keys.append(val)
        
        return keys

    def _get_dim(self, var_key):
        # 保证所有变量都能查到正确维度（adapter最常出问题的点）
        for name in ("_get_dim", "get_dim", "variable_dim"):
            if hasattr(self._f, name):
                return getattr(self._f, name)(var_key)
        if var_key.startswith("x"): return 3
        if var_key.startswith("l"): return 2
        _log.warning(f"Cannot infer dim for {var_key}, default to 1")
        return 1

    def _convert_blocks_to_joint(self, blocks, mu_dict):
        """将块字典转换为联合信息矩阵格式"""
        var_keys = self._var_keys if self._var_keys else sorted(blocks.keys())
        dims, var_indices, total_dim = {}, {}, 0
        for v in var_keys:
            if v in mu_dict:
                dims[v] = len(mu_dict[v])
                var_indices[v] = (total_dim, total_dim + dims[v])
                total_dim += dims[v]
        if total_dim == 0:
            raise ValueError(f"No valid variables found for factor {type(self._f).__name__}")
        Lambda_joint = np.zeros((total_dim, total_dim))
        eta_joint = np.zeros(total_dim)
        # Diagonal blocks
        for v in var_keys:
            if v in blocks and v in var_indices:
                i, j = var_indices[v]
                if isinstance(blocks[v], tuple) and len(blocks[v]) == 2:
                    Lambda_diag, eta_diag = blocks[v]
                    Lambda_joint[i:j, i:j] += Lambda_diag
                    eta_joint[i:j] += eta_diag
                else:
                    Lambda_joint[i:j, i:j] += blocks[v]
        # Cross terms
        for k, val in blocks.items():
            if isinstance(k, tuple) and len(k)==2 and all(v in var_indices for v in k):
                i1, j1 = var_indices[k[0]]
                i2, j2 = var_indices[k[1]]
                Lambda_joint[i1:j1, i2:j2] += val
                if k[0]!=k[1]:  # enforce symmetry
                    Lambda_joint[i2:j2, i1:j1] += val.T
        Lambda_joint = ensure_positive_definite(Lambda_joint, regularization=1e-6)
        return Lambda_joint, eta_joint

    def _joint_to_blocks(self, Lambda, eta, mu_dict):
        """
        将联合矩阵(Lambda, eta)转换为块字典格式
        """
        blocks = {}
        var_keys = [k for k in self._var_keys if k in mu_dict]
        
        # 如果没有提取到变量键，则尝试通过尺寸匹配
        if not var_keys:
            # 紧急回退策略：按照已知变量推断
            _log.warning(f"No variable keys for factor {type(self._f).__name__}, inferring from mu_dict")
            total_size = Lambda.shape[0]
            remaining = total_size
            pos = 0
            
            # 按字母顺序尝试变量
            for k in sorted(mu_dict.keys()):
                dim = len(mu_dict[k])
                if remaining >= dim:
                    var_keys.append(k)
                    remaining -= dim
                    if remaining == 0:
                        break
        
        # 计算各变量在联合矩阵中的位置索引
        var_dims = {v: len(mu_dict[v]) for v in var_keys}
        var_indices = {}
        start_idx = 0
        for v in var_keys:
            dim = var_dims[v]
            var_indices[v] = (start_idx, start_idx + dim)
            start_idx += dim
        
        # 确保矩阵尺寸和变量尺寸匹配
        if start_idx != Lambda.shape[0]:
            _log.warning(f"Matrix size {Lambda.shape[0]} doesn't match total variable dims {start_idx}")
            # 添加最小化的块，确保solver至少有东西可用
            for v in var_keys:
                dim = var_dims[v]
                blocks[v] = (np.eye(dim) * 1e-8, np.zeros(dim))
            return blocks
        
        # 为每个变量创建对角块 (Lambda_ii, eta_i)
        for v in var_keys:
            i, j = var_indices[v]
            L_ii = Lambda[i:j, i:j]
            e_i = eta[i:j] if eta is not None else np.zeros(j-i)
            # 确保对角块是正定的
            L_ii_pd = ensure_positive_definite(L_ii, 1e-8)
            blocks[v] = (L_ii_pd, e_i)
        
        # 创建交叉块 Lambda_ij
        for idx1, v_i in enumerate(var_keys):
            for v_j in var_keys[idx1+1:]:
                i1, j1 = var_indices[v_i]
                i2, j2 = var_indices[v_j]
                L_ij = Lambda[i1:j1, i2:j2]
                
                # 只添加非零交叉块
                if not np.allclose(L_ij, 0):
                    blocks[(v_i, v_j)] = L_ij
        
        return blocks

    def _generate_weak_constraint_dict(self, mu_dict):
        """生成弱约束的字典格式"""
        blocks = {}
        rel_vars = [v for v in self._var_keys if v in mu_dict]
        
        # 如果没有提取到变量，使用所有可能的变量
        if not rel_vars:
            rel_vars = list(mu_dict.keys())
        
        for v in rel_vars:
            dim = len(mu_dict[v])
            blocks[v] = (np.eye(dim) * 1e-8, np.zeros(dim))
        
        return blocks

    def linearize(self, mu_dict, Sigma_dict):
        try:
            raw = self._f.linearize(mu_dict, Sigma_dict)
            if isinstance(raw, tuple) and len(raw)==2 and len(self._var_keys)==1:
                raw = {self._var_keys[0]: raw}
            if not isinstance(raw, dict):
                _log.warning(f"Unexpected factor linearize return type: {type(raw)}")
                return self._generate_weak_constraint_dict(mu_dict)

            sanitized = {}
            for key, blk in raw.items():
                # 1) 交叉块：key 本身就是 tuple，直接透传
                if isinstance(key, tuple):
                    sanitized[key] = blk
                    continue

                # 2) 单变量块：(L, η)
                if isinstance(blk, tuple) and len(blk)==2:
                    L_blk, eta_blk = blk
                    L_blk = make_pd(np.asarray(L_blk), _MIN_PREC)
                    sanitized[key] = (L_blk, eta_blk)

                # 3) 纯信息矩阵：只对方阵做 PD 投影
                elif isinstance(blk, np.ndarray):
                    if blk.ndim == 2 and blk.shape[0] == blk.shape[1]:
                        sanitized[key] = make_pd(blk, _MIN_PREC)
                    else:
                        sanitized[key] = blk

                # 4) 其它奇怪类型还是兜底
                else:
                    _log.warning(f"Skip unexpected block type {type(blk)} for key {key}")
                    sanitized[key] = blk

            return sanitized

        except Exception as e:
            _log.warning("Skip factor %s (linearize failed: %s)", type(self._f).__name__, e)
            return self._generate_weak_constraint_dict(mu_dict)


    def __getattr__(self, item):
        return getattr(self._f, item)
    
    def __repr__(self):
        return f"GBPFactorAdapter({self._f})"
    
    
@dataclass
class GraphBuildStats:
    """图构建统计信息"""
    total_factors: int = 0
    prior_factors: int = 0
    odometry_factors: int = 0
    observation_factors: int = 0
    loop_closure_factors: int = 0
    inter_robot_factors: int = 0  # 
    
    connected_variables: int = 0
    disconnected_variables: int = 0
    
    build_time: float = 0.0
    cache_hit_rate: float = 0.0
    
    # 性能指标
    vectorization_speedup: float = 1.0
    batch_processing_efficiency: float = 1.0

class GBPGraphBuilder:
    """
    Enhanced GBP Factor Graph Builder for Multi-Robot SLAM
    Fully integrated with factor_ut.py v2.3 for optimal performance
    """
    
    _FALLBACK_CONFIG: Dict[str, Any] = {
        # -------- 行为开关 --------
        "anchor_strategy": "poses",  # "mixed" / "first_robot"
        "add_anchor": True,
        "enable_loop_closure": True,
        "enable_true_vectorization": False,
        "enable_batch_processing":   False,
        "enable_factor_reordering":  False,
        "enable_cache_coordination": False,
        "enable_performance_monitoring": False,
        "enable_input_validation": True,
        "enable_adaptive_noise": False,
        "enable_visibility_check": False,
        "enable_strict_validation": False,
        "cache_warmup": False,
        "enable_inter_robot_observations": False,   # ← ★ 新增
        "clear_caches_on_init": False,              # ← ★ 新增
        
        # -------- 观测 / 噪声 --------
        "min_obs_range": 0.5,
        "max_obs_range": 30.0,
        "max_obs_angle": 2 * math.pi,
        "obs_sigma_bearing": 0.01,
        "obs_sigma_range": 0.10,
        # -------- 先验 / 里程计 --------
        "prior_sigma_xy": 0.05,
        "prior_sigma_theta": 0.05,
        "odom_sigma_xy": 0.02,
        "odom_sigma_theta": math.radians(1),
        # -------- 机器人间观测 --------
        "inter_robot_obs_sigma_bearing": 0.015,
        "inter_robot_obs_sigma_range": 0.20,
        # -------- UT / SPBP --------
        "ut_mode": "gbp",
        "inter_robot_obs_mode": "gbp",  # "gbp" / "spbp"
        "ut_alpha": 1e-3,
        "ut_beta": 2.0,
        "ut_kappa": 0.0,
        "ut_linear_threshold": 1e-5,
        "ut_residual_threshold": 3.0,
        "p2p_distance_threshold": 2.0,
        "p2p_residual_threshold": 3.0,
        # -------- 其他 --------
        
     }
    
    @staticmethod
    def _read_yaml(path: Union[str, pathlib.Path]) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    
    def __init__(self,
                 cfg: Optional[Dict[str, Any]] = None,
                 extra_yaml: Optional[Union[str, pathlib.Path]] = None):
        # 0) 兜底
        base_cfg = self._FALLBACK_CONFIG

        # 1) 项目默认 YAML（已展平）
        yaml_cfg = load_gbp()

        # 2) 额外 YAML
        if extra_yaml:
            yaml_cfg.update(_flatten_yaml(self._read_yaml(extra_yaml)))

        # 3) CLI/demo 即时覆盖
        user_cfg = cfg or {}

        # 4) 合并，右侧覆盖左侧
        self.cfg = {**base_cfg, **yaml_cfg, **user_cfg}
        # ➀ 写回 performance_config
        for k in vars(performance_config):            # 全字段扫描
            if k in self.cfg:
                try:
                    # 允许 "32"/"32.0"，并最低限定为 1
                    setattr(performance_config, k,
                            max(1, int(float(self.cfg[k]))))
                except Exception as ex:
                    logger.warning("Bad performance value %s=%s (%s)", k,
                                   self.cfg[k], ex)        
        logger.info("Performance override: %s", asdict(performance_config))

        # ➁ 立即重建全局缓存
        ThreadSafeCache.rebuild_caches()

        self._local_cache = {"noise_matrices": {}, "sigma_arrays": {}}
        self._validate_config()
        self._configure_numerical_parameters()
        self._setup_cache_coordination()

        # State and monitoring
        self.factors, self.variables = [], {}
        self._robot_count = self._landmark_count = 0
        self.stats = GraphBuildStats()
        self._performance_monitor = PerformanceMonitor() \
            if self.cfg["enable_performance_monitoring"] else None
        
        self._landmarks = None
        self.robot_paths = None
        self._build_start_time = None
        self._scene_id = None

        # -------- legacy 字段向后兼容 --------
        legacy_xy = self.cfg.pop("inter_robot_obs_sigma_xy", None)
        legacy_th = self.cfg.pop("inter_robot_obs_sigma_theta", None)
        if legacy_xy is not None and "inter_robot_obs_sigma_range" not in self.cfg:
            self.cfg["inter_robot_obs_sigma_range"] = legacy_xy
        if legacy_th is not None and "inter_robot_obs_sigma_bearing" not in self.cfg:
            self.cfg["inter_robot_obs_sigma_bearing"] = legacy_th

    def build(self, robot_paths: Union[np.ndarray, List[np.ndarray]], 
              landmark_pos: np.ndarray, 
              measurements: List[Dict[str, Any]], 
              loop_closures: Optional[List[Dict[str, Any]]] = None,

              inter_robot_obs: Optional[List[Dict[str, Any]]] = None,              
              scene_id: Optional[str] = None) -> Tuple[List, Dict]:
        """
        构建多机器人SLAM因子图
        
        Args:
            robot_paths: 机器人轨迹 (R, T, 3) 或 List[(T_i, 3)]
            landmark_pos: 地标位置 (L, 2)
            measurements: 观测测量列表
            loop_closures: 可选的环路闭合约束
            inter_robot_obs: ✅ 新增：机器人间观测
            
        Returns:
            (factors, variables): 因子列表和变量字典
        """
        self._scene_id = scene_id
        self._build_start_time = time.perf_counter()
        logger.info("Building enhanced GBP factor graph...")
        
        # 输入验证和标准化
        robot_paths = self._normalize_robot_paths(robot_paths)
        if self.cfg["enable_input_validation"]:
            self._validate_inputs_enhanced(robot_paths, landmark_pos, measurements)
        
        # 重置状态
        self._reset_state()
        
        # 存储引用
        self._landmarks = landmark_pos.copy()
        self.robot_paths = robot_paths
        self._robot_count = len(robot_paths)
        self._landmark_count = len(landmark_pos)
        
        logger.info("Input: %d robots, %d landmarks, %d measurements", 
                   self._robot_count, self._landmark_count, len(measurements))
        
        # 构建图组件
        with self._monitor_section("landmark_variables"):
            self._add_landmark_variables()
        
        with self._monitor_section("robot_trajectories"):
            self._add_robot_trajectories()
        
        with self._monitor_section("observation_factors"):
            self._add_observation_factors(measurements)
        
        # ✅ 新增：机器人间观测因子
        if inter_robot_obs and self.cfg["enable_inter_robot_observations"]:
            with self._monitor_section("inter_robot_factors"):
                self._add_inter_robot_observation_factors(inter_robot_obs)
        
        if loop_closures and self.cfg["enable_loop_closure"]:
            with self._monitor_section("loop_closure_factors"):
                self._add_loop_closure_factors(loop_closures)
        
        # 全局约束
        if self.cfg["add_anchor"]:
            with self._monitor_section("anchor_constraints"):
                self._add_global_anchor_constraints()
        
        # 图优化和验证
        with self._monitor_section("graph_optimization"):
            self._optimize_and_validate_graph()
        
        # 最终统计和日志
        self._finalize_build()
        print("\n[DEBUG] 变量约束诊断：")
        for var in self.variables:
            involved = []
            for i, factor in enumerate(self.factors):
                # 兼容所有因子实现
                if hasattr(factor, "_get_dim"):
                    if factor._get_dim(var) > 0:
                        involved.append((i, type(factor).__name__))
                elif hasattr(factor, "variable_keys"):
                    if var in getattr(factor, "variable_keys", []):
                        involved.append((i, type(factor).__name__))
            print(f"  {var}: 被以下{len(involved)}个因子约束 -> {involved if involved else '[未被任何因子约束]'}")
        print("[DEBUG] 约束诊断结束\n")
        print("\n[DIAG] ====== Global Information Matrix ======")
        Lambda = self.build_information_matrix(return_sparse=False)
        eigvals = np.linalg.eigvalsh(Lambda)
        print(f"[DIAG] InfoMatrix eigenvalues: {eigvals}")
        print(f"[DIAG] Condition number: {np.linalg.cond(Lambda):.3g}")
        print(f"[DIAG] Min eigval: {eigvals.min():.3g}, Max eigval: {eigvals.max():.3g}\n")

        return self.factors, self.variables
    def _prefixed(self, key: str) -> str:
        """如果指定了 scene_id，就自动加前缀"""
        if self._scene_id:
            return f"{self._scene_id}_{key}"
        return key
    
    def build_single_robot(self,
                           trajectory: np.ndarray,
                           landmarks: np.ndarray,
                           measurements: List[Dict],
                           scene_id: Optional[str] = None) -> Tuple[List, Dict]:
        """单机器人SLAM的便利方法"""
        robot_paths = [trajectory]
        return self.build(robot_paths, landmarks, measurements, scene_id=scene_id)

    # -------------------------------------------------------------------------
    # ✅ 新增：机器人间观测因子
    # -------------------------------------------------------------------------
    
    def _add_inter_robot_observation_factors(self, inter_robot_obs: List[Dict[str, Any]]):
        """
        添加机器人间位姿观测因子
        
        Args:
            inter_robot_obs: 机器人间观测列表
        """
        logger.info("Adding %d inter-robot observation factors...", len(inter_robot_obs))
        
        valid_count = 0
        for obs in inter_robot_obs:
            try:
                if self._add_single_inter_robot_observation(obs):
                    valid_count += 1
            except Exception as e:
                logger.warning("Failed to add inter-robot observation: %s", str(e))
        
        logger.info("Added %d valid inter-robot observation factors", valid_count)
        self.stats.inter_robot_factors = valid_count
        

    def _add_single_inter_robot_observation(self, obs: Dict[str, Any]) -> bool:
        """✅ 修复7: 添加单个机器人间观测因子，使用安全None检查"""
        # 提取参数
        observer_robot = obs.get("observer_robot", obs.get("robot1"))
        observer_time = obs.get("observer_time", obs.get("time1"))
        observed_robot = obs.get("observed_robot", obs.get("robot2"))
        observed_time = obs.get("observed_time", obs.get("time2"))
        # ---- measurement vector: [bearing, range] ----
        meas_vec = None
        if "bearing_range" in obs and obs["bearing_range"] is not None:
            meas_vec = np.asarray(obs["bearing_range"], dtype=float)
        elif "measurement" in obs and obs["measurement"] is not None:   # 兼容别名
            meas_vec = np.asarray(obs["measurement"], dtype=float)
        else:
            return False          # 没找到合法 2-DoF 观测

        # 保证正好 2 维
        if meas_vec.size != 2:
            return False
                
        # ✅ 修复7: 使用安全的None检查，避免numpy数组问题
        if self._safe_none_check(observer_robot, observer_time, observed_robot, observed_time):
            return False
        
        # 验证范围
        if not (0 <= observer_robot < self._robot_count and 0 <= observed_robot < self._robot_count):
            return False
        
        if observer_robot == observed_robot:  # 同一机器人
            return False
        
        # 创建变量键
        observer_key = self._prefixed(f"x{observer_robot}_{observer_time}")
        observed_key = self._prefixed(f"x{observed_robot}_{observed_time}")
        
        # 验证变量存在
        if observer_key not in self.variables or observed_key not in self.variables:
            return False
        
        # 噪声协方差
        noise_cov = obs.get("noise_covariance")
        if noise_cov is None:
            noise_cov = self._get_inter_robot_noise_matrix()
        else:
            noise_cov = np.array(noise_cov, dtype=np.float64)
            if self.cfg["enable_strict_validation"]:
                try:
                    validate_matrix_properties(noise_cov, "inter_robot_noise_covariance")
                except ValueError as e:
                    logger.warning("Invalid inter-robot noise covariance: %s", str(e))
                    noise_cov = ensure_positive_definite(noise_cov, numerical_config.regularization)
        
        # ✅ 创建PoseToPoseUTFactor
        factor = PoseToPoseUTFactor(
            observer_key,
            observed_key,
            meas_vec.astype(np.float64),
            noise_cov,
            mode=self.cfg["inter_robot_obs_mode"],
            alpha=self.cfg["ut_alpha"],
            beta=self.cfg["ut_beta"],
            kappa=self.cfg["ut_kappa"],
            distance_threshold=self.cfg["p2p_distance_threshold"],
            residual_sigma_thresh=self.cfg["p2p_residual_threshold"]
        )
        
        self.factors.append(factor)
        print(f"[DEBUG] Add factor: {factor}")
        return True

    # -------------------------------------------------------------------------
    # ✅ 重构的核心图构建方法 - 消除代码重复
    # -------------------------------------------------------------------------
    
    def _add_robot_trajectories(self):
        """添加机器人姿态变量和里程计因子"""
        # 检查是否由_add_global_anchor_constraints负责添加先验
        add_trajectory_priors = not self.cfg.get("add_anchor", True)
        
        if self.cfg["enable_true_vectorization"]:
            self._add_all_robot_trajectories_vectorized(add_trajectory_priors)
        else:
            for rid, poses in enumerate(self.robot_paths):
                self._add_single_robot_trajectory(rid, poses, add_trajectory_priors)
        
        logger.info("Added %d robot trajectories", self._robot_count)

    def _add_all_robot_trajectories_vectorized(self, add_trajectory_priors: bool = True):
        """✅ 修复1: 真正向量化的机器人轨迹处理，准确计算时间"""
        local_start = time.perf_counter()  # ✅ 修复1: 本地计时开始
        
        prior_sigma = self._get_prior_sigma()
        odom_sigma = self._get_odometry_sigma()
        
        # 批量添加所有姿态变量
        for rid, poses in enumerate(self.robot_paths):
            T = poses.shape[0]
            # 批量添加姿态变量
            for t in range(T):
                key = self._prefixed(f"x{rid}_{t}")
                self.variables[key] = poses[t].astype(np.float64).copy()
            
            # ✅ 修复: 根据参数决定是否添加先验因子
            if add_trajectory_priors:
                first_pose_key = self._prefixed(f"x{rid}_0")
                self.factors.append(PriorFactor(first_pose_key, poses[0].copy(), prior_sigma))
                self.stats.prior_factors += 1
                logger.debug(f"Added trajectory prior to {first_pose_key}")
        
        # ✅ 向量化里程计因子创建
        total_odom_factors = 0
        for rid, poses in enumerate(self.robot_paths):
            if poses.shape[0] > 1:
                odom_factors = self._create_vectorized_odometry_factors(rid, poses, odom_sigma)
                self.factors.extend(odom_factors)
                total_odom_factors += len(odom_factors)
        
        self.stats.odometry_factors = total_odom_factors
        
        # ✅ 修复1: 使用本地时间计算向量化加速比
        if self._performance_monitor:
            vectorized_time = time.perf_counter() - local_start  # 只计算本函数耗时
            naive_time_estimate = sum(len(path) for path in self.robot_paths) * 1e-4  # 估计
            self.stats.vectorization_speedup = max(1.0, naive_time_estimate / max(vectorized_time, 1e-6))

    def _create_vectorized_odometry_factors(self, robot_id: int, poses: np.ndarray, 
                                          odom_sigma: np.ndarray) -> List[OdometryFactor]:
        """✅ 修复5: 更高效的向量化里程计因子创建"""
        T = poses.shape[0]
        if T <= 1:
            return []
        
        # ✅ 修复5: 真正的向量化计算所有相对位姿
        # 批量计算所有delta (避免循环)
        from_poses = poses[:-1]  # T-1 个起始位姿
        to_poses = poses[1:]     # T-1 个目标位姿
        
        factors = []
        
        # 虽然SE(2)相对位姿计算仍需逐个处理，但可以预计算键名
        keys_from = [self._prefixed(f"x{robot_id}_{t}")   for t in range(T-1)]
        keys_to   = [self._prefixed(f"x{robot_id}_{t+1}") for t in range(T-1)]
        
        # 批量创建因子 (减少函数调用开销)
        for i, (from_key, to_key) in enumerate(zip(keys_from, keys_to)):
            # ✅ 复用factor_ut中的实现，消除代码重复
            delta = OdometryFactor._se2_relative_pose(from_poses[i], to_poses[i])
            factor = OdometryFactor(from_key, to_key, delta, odom_sigma)
            factors.append(factor)
        
        return factors

    def _add_single_robot_trajectory(self, robot_id: int, poses: np.ndarray, add_trajectory_priors: bool = True):
        """单个机器人轨迹处理（回退方法）"""
        T = poses.shape[0]
        
        # 添加姿态变量
        for t in range(T):
            key = self._prefixed(f"x{robot_id}_{t}")
            self.variables[key] = poses[t].astype(np.float64).copy()
        
        # ✅ 修复: 根据参数决定是否添加先验因子
        if add_trajectory_priors:
            first_pose_key = self._prefixed(f"x{robot_id}_0")
            prior_sigma = self._get_prior_sigma()
            
            self.factors.append(PriorFactor(first_pose_key, poses[0].copy(), prior_sigma))
            self.stats.prior_factors += 1
            logger.debug(f"Added trajectory prior to {first_pose_key}")
        
        # 添加里程计因子
        if T > 1:
            odom_sigma = self._get_odometry_sigma()
            for t in range(T - 1):
                from_key = self._prefixed(f"x{robot_id}_{t}")
                to_key   = self._prefixed(f"x{robot_id}_{t+1}")
                
                # ✅ 复用factor_ut实现
                delta = OdometryFactor._se2_relative_pose(poses[t], poses[t+1])
                
                factor = OdometryFactor(from_key, to_key, delta, odom_sigma)
                self.factors.append(factor)
                print(f"[DEBUG] Add factor: {factor}")
                self.stats.odometry_factors += 1

    def _add_observation_factors(self, measurements: List[Dict[str, Any]]):
        """✅ 增强的观测因子处理，支持真正的批处理"""
        print(f"[DEBUG] _add_observation_factors called with {len(measurements)} measurements")  # ← 新增
        print(f"[DEBUG] First measurement sample: {measurements[0] if measurements else 'NO MEASUREMENTS'}")  # ← 新增
        logger.info("Processing %d measurements...", len(measurements))
        
        if self.cfg["enable_batch_processing"]:
            print(f"[DEBUG] Using batch processing") 
            valid_count = self._add_batch_observation_factors(measurements)
        else:
            print(f"[DEBUG] Using sequential processing")
            valid_count = self._add_sequential_observation_factors(measurements)
            
        print(f"[DEBUG] _add_observation_factors completed, valid_count={valid_count}")
        self.stats.observation_factors = valid_count
        logger.info("Added %d observation factors", valid_count)

    def _add_batch_observation_factors(self, measurements: List[Dict[str, Any]]) -> int:
        """✅ 真正的批处理观测因子创建"""
        start_time = time.perf_counter()
        
        # 按机器人分组
        robot_measurements = defaultdict(list)
        valid_measurements = []
        
        # 批量验证
        for measurement in measurements:
            robot_id = measurement.get("robot")
            time_step = measurement.get("time")
            landmark_id = measurement.get("id")
            
            if self._validate_measurement_params(robot_id, time_step, landmark_id):
                obs_data = self._extract_observation_data(measurement)
                if obs_data is not None:
                    robot_measurements[robot_id].append((measurement, obs_data))
                    valid_measurements.append((measurement, obs_data))
        
        # 预计算噪声矩阵（如果不是自适应的）
        base_noise_matrix = None
        if not self.cfg["enable_adaptive_noise"]:
            base_noise_matrix = self._get_observation_noise_matrix(10.0)  # 使用中等距离
        
        # 批量创建因子
        total_factors = 0
        for robot_id, robot_obs in robot_measurements.items():
            factors_created = self._process_robot_observations_batch(
                robot_id, robot_obs, base_noise_matrix
            )
            total_factors += factors_created
        
        # 计算批处理效率
        batch_time = time.perf_counter() - start_time
        if self._performance_monitor:
            estimated_sequential_time = len(valid_measurements) * 1e-4  # 估计
            self.stats.batch_processing_efficiency = max(1.0, 
                estimated_sequential_time / max(batch_time, 1e-6))
        
        return total_factors

    def _process_robot_observations_batch(self, robot_id: int, 
                                        robot_obs: List[Tuple[Dict, Tuple[float, float]]],
                                        base_noise_matrix: Optional[np.ndarray]) -> int:
        """批量处理单个机器人的观测"""
        factors_created = 0
        
        for measurement, (bearing, range_val) in robot_obs:
            try:
                pose_key     = self._prefixed(f"x{robot_id}_{measurement['time']}")
                landmark_key = self._prefixed(f"l_{measurement['id']}")
                
                
                # 可见性检查
                if self.cfg["enable_visibility_check"]:
                    if not self._check_visibility(pose_key, landmark_key, range_val):
                        continue
                
                # 使用预计算的噪声矩阵或计算自适应噪声
                if base_noise_matrix is not None:
                    noise_cov = base_noise_matrix
                else:
                    noise_cov = self._get_observation_noise_matrix(range_val)
                
                measurement_vector = np.array([bearing, range_val], dtype=np.float64)
                
                factor = BearingRangeUTFactor(
                    pose_key, landmark_key, measurement_vector, noise_cov,
                    mode=self.cfg["ut_mode"],
                    alpha=self.cfg["ut_alpha"],
                    beta=self.cfg["ut_beta"], 
                    kappa=self.cfg["ut_kappa"],
                    linear_threshold=self.cfg["ut_linear_threshold"],
                    residual_sigma_thresh=self.cfg["ut_residual_threshold"]
                )
                
                self.factors.append(factor)
                print(f"[DEBUG] Add factor: {factor}")
                factors_created += 1
                
            except Exception as e:
                logger.warning("Batch processing error for robot %d: %s", robot_id, str(e))
        self.factors.append(factor)

        return factors_created

    def _add_sequential_observation_factors(self, measurements: List[Dict[str, Any]]) -> int:
        """传统的序列化观测因子处理"""
        print(f"[DEBUG] _add_sequential_observation_factors called with {len(measurements)} measurements")  # ← 新增
        
        valid_count = 0
        for i, measurement in enumerate(measurements):
            print(f"[DEBUG] Processing measurement {i}: {measurement}")  # ← 新增
            try:
                result = self._process_single_measurement(measurement)
                print(f"[DEBUG] Measurement {i} result: {result}")  # ← 新增
                if result:
                    valid_count += 1
            except Exception as e:
                print(f"[DEBUG] Exception in measurement {i}: {str(e)}")  # ← 新增
                logger.warning("Error processing measurement: %s", str(e))
        
        print(f"[DEBUG] _add_sequential_observation_factors returning valid_count={valid_count}")  # ← 新增
        return valid_count

    # -------------------------------------------------------------------------
    # ✅ 集成数值配置和缓存协调
    # -------------------------------------------------------------------------
    
    def _configure_numerical_parameters(self):
        """✅ 配置全局数值参数，并清理相关缓存"""
        numerical_params = self.cfg.get("numerical_config", {})
        
        if numerical_params:
            # 记录旧的regularization值
            old_regularization = numerical_config.regularization
            
            configure_numerical_parameters(**numerical_params)
            
            # ✅ 如果regularization改变，清理依赖的本地缓存
            if numerical_config.regularization != old_regularization:
                self._local_cache["noise_matrices"].clear()
                logger.debug("Cleared local noise matrix cache due to regularization change")
            
            logger.info("Configured numerical parameters: %s", numerical_params)

    def _setup_cache_coordination(self):
        """✅ 设置缓存协调"""
        if self.cfg["clear_caches_on_init"]:
            clear_all_caches()
            logger.info("Cleared global factor caches")
        
        # 记录初始缓存状态
        self._initial_cache_stats = get_global_cache_stats()
        logger.debug("Initial cache stats: %s", self._initial_cache_stats)

    def _get_observation_noise_matrix(self, range_val: float) -> np.ndarray:
        """✅ 修复2: 修复的观测噪声矩阵，正确处理自适应噪声缓存"""
        
        # 如果启用自适应噪声，直接计算，不使用缓存
        if self.cfg["enable_adaptive_noise"]:
            # ✅ 修复2: 在函数开始就复制标量值，避免累积修改
            sigma_bearing = float(self.cfg["obs_sigma_bearing"])  # 显式复制
            sigma_range = float(self.cfg["obs_sigma_range"])      # 显式复制
            
            range_factor = 1.0 + 0.1 * (range_val / 10.0)
            sigma_range *= range_factor
            sigma_bearing *= math.sqrt(range_factor)
            
            R = np.diag([sigma_bearing**2, sigma_range**2])
            return ensure_positive_definite(R, numerical_config.regularization)
        
        # 非自适应噪声使用缓存
        config_hash = hash((
            self.cfg["obs_sigma_bearing"], 
            self.cfg["obs_sigma_range"],
            numerical_config.regularization
        ))
        cache_key = f"obs_noise_{config_hash}"
        
        if cache_key in self._local_cache["noise_matrices"]:
            return self._local_cache["noise_matrices"][cache_key].copy()
        
        # 计算并缓存
        sigma_bearing = self.cfg["obs_sigma_bearing"]
        sigma_range = self.cfg["obs_sigma_range"]
        R = np.diag([sigma_bearing**2, sigma_range**2])
        R = ensure_positive_definite(R, numerical_config.regularization)
        
        self._local_cache["noise_matrices"][cache_key] = R.copy()
        return R

    def _get_inter_robot_noise_matrix(self) -> np.ndarray:
        """✅ 修复3: 解决缓存一致性问题"""
        # ✅ 修复3: 加入regularization到缓存键中，确保一致性
        regularization = numerical_config.regularization
        cache_key = f"inter_robot_noise_2dof_{regularization}"
        
        if cache_key not in self._local_cache["noise_matrices"]:
            sigma_b = self.cfg["inter_robot_obs_sigma_bearing"]
            sigma_r = self.cfg["inter_robot_obs_sigma_range"]

            cov = np.diag([sigma_b ** 2, sigma_r ** 2])
            cov = ensure_positive_definite(cov, regularization)
            self._local_cache["noise_matrices"][cache_key] = cov
        
        # ✅ 修复3: 总是返回拷贝，避免意外修改缓存
        return self._local_cache["noise_matrices"][cache_key].copy()

    def _get_prior_sigma(self) -> np.ndarray:
        """获取先验约束标准差"""
        cache_key = "prior_sigma"
        
        if cache_key not in self._local_cache["sigma_arrays"]:
            sigma = np.array([
                self.cfg["prior_sigma_xy"],
                self.cfg["prior_sigma_xy"], 
                self.cfg["prior_sigma_theta"]
            ], dtype=np.float64)
            self._local_cache["sigma_arrays"][cache_key] = sigma
        
        return self._local_cache["sigma_arrays"][cache_key].copy()

    def _get_odometry_sigma(self) -> np.ndarray:
        """获取里程计噪声标准差"""
        cache_key = "odom_sigma"
        
        if cache_key not in self._local_cache["sigma_arrays"]:
            sigma = np.array([
                self.cfg["odom_sigma_xy"],
                self.cfg["odom_sigma_xy"],
                self.cfg["odom_sigma_theta"]
            ], dtype=np.float64)
            self._local_cache["sigma_arrays"][cache_key] = sigma
        
        return self._local_cache["sigma_arrays"][cache_key].copy()

    # -------------------------------------------------------------------------
    # ✅ 增强的验证和错误处理
    # -------------------------------------------------------------------------
    
    def _validate_config(self):
        """✅ 增强的配置验证"""
        # 基本键检查
        required_keys = [
            "ut_mode", "max_obs_range", "obs_sigma_bearing", "obs_sigma_range",
            "prior_sigma_xy", "prior_sigma_theta", "odom_sigma_xy", "odom_sigma_theta"
        ]
        
        for key in required_keys:
            if key not in self.cfg:
                raise ValueError(f"Required config key missing: {key}")
        
        # ✅ UT模式验证
        valid_ut_modes = ["auto", "gbp", "spbp"]
        if self.cfg["ut_mode"] not in valid_ut_modes:
            raise ValueError(f"Invalid ut_mode: {self.cfg['ut_mode']}")
        
        if self.cfg["inter_robot_obs_mode"] not in valid_ut_modes:
            raise ValueError(f"Invalid inter_robot_obs_mode: {self.cfg['inter_robot_obs_mode']}")
        
        # ✅ 数值范围验证
        self._validate_numerical_ranges()
        
        # ✅ UT参数验证
        self._validate_ut_parameters()
        
        logger.info("Enhanced configuration validation passed")

    def _validate_numerical_ranges(self):
        """✅ 修复4: 验证数值参数范围，使用安全的键检查"""
        positive_params = [
            "max_obs_range", "min_obs_range",
            "obs_sigma_bearing", "obs_sigma_range",
            "prior_sigma_xy", "prior_sigma_theta",
            "odom_sigma_xy", "odom_sigma_theta",
            "inter_robot_obs_sigma_bearing",   # ✅ 新字段
            "inter_robot_obs_sigma_range"      # ✅ 新字段
        ]
        
        # ✅ 修复4: 只检查实际存在的参数
        existing_params = set(positive_params) & set(self.cfg.keys())
        
        for param in existing_params:
            if self.cfg[param] <= 0:
                raise ValueError(f"Parameter {param} must be positive")
        
        if self.cfg.get("min_obs_range", 0) >= self.cfg.get("max_obs_range", float('inf')):
            raise ValueError("min_obs_range must be less than max_obs_range")

    def _validate_ut_parameters(self):
        """验证UT参数"""
        # Alpha检查
        for alpha_key in ["ut_alpha"]:
            alpha = self.cfg.get(alpha_key, 1e-3)
            if not (1e-4 <= alpha <= 1.0):
                logger.warning("%s should be in [1e-4, 1.0], got %f", alpha_key, alpha)
                self.cfg[alpha_key] = np.clip(alpha, 1e-4, 1.0)
        
        # Beta检查
        for beta_key in ["ut_beta"]:
            beta = self.cfg.get(beta_key, 2.0)
            if beta < 0:
                logger.warning("%s should be non-negative, got %f", beta_key, beta)
                self.cfg[beta_key] = max(0, beta)

    def _validate_inputs_enhanced(self, robot_paths: List[np.ndarray], 
                                landmarks: np.ndarray, measurements: List[Dict]):
        """✅ 修复8: 增强的输入验证，避免就地修改输入数据"""
        # 验证机器人轨迹
        for i, path in enumerate(robot_paths):
            if path.ndim != 2 or path.shape[1] != 3:
                raise ValueError(f"Robot {i} path must be (T, 3)")
            if path.shape[0] < 1:
                raise ValueError(f"Robot {i} path must have at least 1 pose")
            if not np.isfinite(path).all():
                raise ValueError(f"Robot {i} path contains non-finite values")
            
            # ✅ 修复8: 角度标准化不修改原数组
            angles = path[:, 2]
            normalized_angles = np.array([wrap_angle(a) for a in angles])
            if not np.allclose(angles, normalized_angles, atol=1e-6):
                logger.info("Normalizing angles for robot %d", i)
                # 创建拷贝，不修改原数组
                path_copy = path.copy()
                path_copy[:, 2] = normalized_angles
                robot_paths[i] = path_copy  # 替换引用，不影响调用方原数据
        
        # 验证地标
        if landmarks.ndim != 2 or landmarks.shape[1] != 2:
            raise ValueError(f"Landmarks must be (L, 2)")
        if not np.isfinite(landmarks).all():
            raise ValueError("Landmarks contain non-finite values")
        
        # 验证测量一致性
        if self.cfg["enable_strict_validation"]:
            self._validate_measurements_consistency(measurements, len(robot_paths), len(landmarks))

    def _validate_measurements_consistency(self, measurements: List[Dict], 
                                         num_robots: int, num_landmarks: int):
        """验证测量一致性"""
        invalid_count = 0
        robot_ids, landmark_ids = set(), set()
        
        for i, measurement in enumerate(measurements):
            robot_id = measurement.get("robot")
            landmark_id = measurement.get("id")
            time_step = measurement.get("time")
            
            if None in [robot_id, landmark_id, time_step]:
                invalid_count += 1
                continue
            
            if not (0 <= robot_id < num_robots and 0 <= landmark_id < num_landmarks):
                invalid_count += 1
                continue
            
            robot_ids.add(robot_id)
            landmark_ids.add(landmark_id)
            
            obs_data = self._extract_observation_data(measurement)
            if obs_data is None:
                invalid_count += 1
        
        logger.info("Validation: %d robots, %d landmarks observed, %d invalid measurements", 
                   len(robot_ids), len(landmark_ids), invalid_count)

    # -------------------------------------------------------------------------
    # ✅ 性能优化和监控
    # -------------------------------------------------------------------------
    
    def _optimize_and_validate_graph(self):
        """图优化和验证"""
        if self.cfg["enable_factor_reordering"]:
            self._optimize_factor_ordering_advanced()
        
        self._validate_graph_connectivity()
        
        if self.cfg["cache_warmup"]:
            self._warmup_factor_caches()

    def _optimize_factor_ordering_advanced(self):
        """✅ 高级因子排序，提高缓存局部性"""
        def factor_sort_key(factor):
            type_priority = {
                PriorFactor: 0,
                OdometryFactor: 1, 
                BearingRangeUTFactor: 2,
                PoseToPoseUTFactor: 3,
                LoopClosureFactor: 4
            }
            
            # 主要变量引用用于局部性
            if isinstance(factor, PriorFactor):
                var_ref = factor.var
                var_ref2 = ""
            elif isinstance(factor, OdometryFactor):
                var_ref = factor.v1
                var_ref2 = factor.v2  # ✅ 加入第二个变量
            elif isinstance(factor, BearingRangeUTFactor):
                var_ref = factor.pose_key
                var_ref2 = factor.lm_key
            elif isinstance(factor, PoseToPoseUTFactor):
                var_ref = factor.pose1_key
                var_ref2 = factor.pose2_key
            elif isinstance(factor, LoopClosureFactor):
                var_ref = factor.pose1_key
                var_ref2 = factor.pose2_key
            else:
                var_ref = ""
                var_ref2 = ""
            
            # ✅ 改进的排序键：类型 + 主变量 + 次变量
            combined_ref = f"{var_ref}:{var_ref2}"
            return (type_priority.get(type(factor), 999), combined_ref)
        
        self.factors.sort(key=factor_sort_key)
        logger.debug("Optimized factor ordering for improved cache locality")

    def _warmup_factor_caches(self):
        """✅ 修复6: 改进的因子缓存预热，确保覆盖所有因子类型"""
        if len(self.factors) == 0 or len(self.variables) == 0:
            return
        
        try:
            # 按因子类型分组采样
            factor_samples = {
                PriorFactor: [],
                OdometryFactor: [],
                BearingRangeUTFactor: [],
                PoseToPoseUTFactor: [],
                LoopClosureFactor: []
            }
            
            # 每种类型采样1-2个因子
            for factor in self.factors:
                factor_type = type(factor)
                if factor_type in factor_samples and len(factor_samples[factor_type]) < 2:
                    factor_samples[factor_type].append(factor)
            
            # 创建覆盖所有变量类型的虚拟状态
            pose_vars = {k: v for k, v in self.variables.items() if k.startswith('x')}
            landmark_vars = {k: v for k, v in self.variables.items() if k.startswith('l')}
            
            # 选择代表性变量（每种类型至少1个）
            dummy_mu = {}
            dummy_cov = {}
            
            # 添加姿态变量样本
            pose_items = list(pose_vars.items())[:5]  # 最多5个姿态
            for var, val in pose_items:
                dummy_mu[var] = val.copy()
                dummy_cov[var] = np.eye(len(val)) * 0.01
            
            # 添加地标变量样本  
            landmark_items = list(landmark_vars.items())[:3]  # 最多3个地标
            for var, val in landmark_items:
                dummy_mu[var] = val.copy()
                dummy_cov[var] = np.eye(len(val)) * 0.01
            
            # ✅ 修复6: 为PoseToPoseUTFactor添加交叉协方差
            pose_keys = list(pose_vars.keys())[:5]
            for i in range(len(pose_keys)):
                for j in range(i+1, len(pose_keys)):
                    # 添加交叉协方差项用于SPBP预热
                    cross_key = (pose_keys[i], pose_keys[j])
                    if cross_key not in dummy_cov:
                        dummy_cov[cross_key] = np.zeros((3, 3))  # 零矩阵作为交叉协方差
                    
                    # 反向交叉协方差也添加
                    cross_key_rev = (pose_keys[j], pose_keys[i])
                    if cross_key_rev not in dummy_cov:
                        dummy_cov[cross_key_rev] = np.zeros((3, 3))
            
            # 对每种因子类型进行预热
            total_warmed = 0
            for factor_type, samples in factor_samples.items():
                for factor in samples:
                    try:
                        # ✅ 修复6: 针对PoseToPoseUTFactor特殊处理
                        if isinstance(factor, PoseToPoseUTFactor):
                            # 确保有交叉协方差，否则跳过SPBP预热
                            cross_key = (factor.pose1_key, factor.pose2_key)
                            if cross_key not in dummy_cov:
                                # 跳过SPBP预热，只做GBP预热
                                continue
                        
                        factor.linearize(dummy_mu, dummy_cov)
                        factor.get_energy(dummy_mu)
                        total_warmed += 1
                    except:
                        pass  # 忽略预热失败
            
            logger.debug(f"Cache warmup completed: {total_warmed} factors warmed")
            
        except Exception as e:
            logger.debug(f"Cache warmup failed: {str(e)}")

    def _finalize_build(self):
        """完成构建，更新统计信息"""
        build_time = time.perf_counter() - self._build_start_time
        self.stats.build_time = build_time
        self.stats.total_factors = len(self.factors)
        
        # 计算缓存命中率
        current_cache_stats = get_global_cache_stats()
        if "factor_caches" in current_cache_stats:
            hit_rates = []
            for cache_stats in current_cache_stats["factor_caches"].values():
                if "hit_rate" in cache_stats:
                    hit_rates.append(cache_stats["hit_rate"])
            if hit_rates:
                self.stats.cache_hit_rate = sum(hit_rates) / len(hit_rates)
        
        self._log_build_summary()

    def _log_build_summary(self):
        """✅ 增强的构建摘要日志"""
        logger.info("=== Enhanced GBP Graph Build Summary ===")
        logger.info("Build time: %.3f seconds", self.stats.build_time)
        
        logger.info("Variables: %d total", len(self.variables))
        logger.info("  - Robot poses: %d", sum(1 for k in self.variables if k.startswith('x')))
        logger.info("  - Landmarks: %d", sum(1 for k in self.variables if k.startswith('l')))
        
        logger.info("Factors: %d total", self.stats.total_factors)
        logger.info("  - Prior: %d", self.stats.prior_factors)
        logger.info("  - Odometry: %d", self.stats.odometry_factors)
        logger.info("  - Observations: %d", self.stats.observation_factors)
        logger.info("  - Inter-robot: %d", self.stats.inter_robot_factors)  # ✅ 新增
        logger.info("  - Loop closures: %d", self.stats.loop_closure_factors)
        
        logger.info("Performance metrics:")
        logger.info("  - Cache hit rate: %.1f%%", self.stats.cache_hit_rate * 100)
        logger.info("  - Vectorization speedup: %.1fx", self.stats.vectorization_speedup)
        logger.info("  - Batch processing efficiency: %.1fx", self.stats.batch_processing_efficiency)
        
        logger.info("Connectivity: %d connected, %d disconnected", 
                   self.stats.connected_variables, self.stats.disconnected_variables)

    # -------------------------------------------------------------------------
    # 其余辅助方法（保持不变或轻微改进）
    # -------------------------------------------------------------------------
    
    def _add_landmark_variables(self):
        """添加地标变量"""
        for lid, pos in enumerate(self._landmarks):
            key = self._prefixed(f"l_{lid}")
            self.variables[key] = pos.astype(np.float64).copy()
        
        logger.debug("Added %d landmark variables", len(self._landmarks))

    def _normalize_robot_paths(self, robot_paths) -> List[np.ndarray]:
        """标准化机器人路径格式"""
        if isinstance(robot_paths, np.ndarray):
            if robot_paths.ndim == 3:
                normalized_paths = [robot_paths[i] for i in range(robot_paths.shape[0])]
            elif robot_paths.ndim == 2:
                normalized_paths = [robot_paths]
            else:
                raise ValueError(f"Invalid robot_paths dimensions: {robot_paths.shape}")
        elif isinstance(robot_paths, list):
            normalized_paths = [np.asarray(path, dtype=np.float64) for path in robot_paths]
        else:
            raise ValueError(f"Invalid robot_paths type: {type(robot_paths)}")
        
        # ✅ 无条件角度归一化，确保数值稳定性
        for i, path in enumerate(normalized_paths):
            if path.shape[1] >= 3:  # 确保有角度列
                path[:, 2] = np.array([wrap_angle(angle) for angle in path[:, 2]])
                logger.debug(f"Normalized angles for robot {i}")
        
        return normalized_paths

    def _validate_measurement_params(self, robot_id, time_step, landmark_id) -> bool:
        """验证测量参数"""
        print(f"[DEBUG] _validate_measurement_params: robot_id={robot_id}, time_step={time_step}, landmark_id={landmark_id}")  # ← 新增
        
        if robot_id is None or time_step is None or landmark_id is None:
            print(f"[DEBUG] _validate_measurement_params FAIL: None values")  # ← 新增
            return False
        
        if robot_id < 0 or robot_id >= self._robot_count:
            print(f"[DEBUG] _validate_measurement_params FAIL: robot_id {robot_id} out of range [0, {self._robot_count})")  # ← 新增
            return False
        
        if time_step < 0 or time_step >= self.robot_paths[robot_id].shape[0]:
            print(f"[DEBUG] _validate_measurement_params FAIL: time_step {time_step} out of range [0, {self.robot_paths[robot_id].shape[0]})")  # ← 新增
            return False
        
        if landmark_id < 0 or landmark_id >= self._landmark_count:
            print(f"[DEBUG] _validate_measurement_params FAIL: landmark_id {landmark_id} out of range [0, {self._landmark_count})")  # ← 新增
            return False
        
        print(f"[DEBUG] _validate_measurement_params SUCCESS")  # ← 新增
        return True

    def _extract_observation_data(self, measurement: Dict) -> Optional[Tuple[float, float]]:
        """提取观测数据"""
        print(f"[DEBUG] _extract_observation_data called with: {measurement}")  # ← 新增
        
        if "bearing_range" in measurement:
            br = measurement["bearing_range"]
            print(f"[DEBUG] Found bearing_range: {br}")  # ← 新增
            if len(br) == 2:
                bearing, range_val = float(br[0]), float(br[1])
                print(f"[DEBUG] Extracted from bearing_range: bearing={bearing}, range={range_val}")  # ← 新增
            else:
                print(f"[DEBUG] FAIL: bearing_range length != 2")  # ← 新增
                return None
        elif "bearing" in measurement and "range" in measurement:
            bearing = float(measurement["bearing"])
            range_val = float(measurement["range"])
            print(f"[DEBUG] Extracted from separate fields: bearing={bearing}, range={range_val}")  # ← 新增
        else:
            print(f"[DEBUG] FAIL: No bearing/range data found in measurement")  # ← 新增
            return None
        
        bearing = wrap_angle(bearing)
        print(f"[DEBUG] After wrap_angle: bearing={bearing}")  # ← 新增
        
        if range_val <= 0:
            print(f"[DEBUG] FAIL: range_val <= 0: {range_val}")  # ← 新增
            return None
        
        print(f"[DEBUG] _extract_observation_data SUCCESS: ({bearing}, {range_val})")  # ← 新增
        return bearing, range_val

    def _check_visibility(self, pose_key: str, landmark_key: str, range_val: float) -> bool:
        """检查可见性"""
        if range_val < self.cfg["min_obs_range"] or range_val > self.cfg["max_obs_range"]:
            return False
        
        if self.cfg["max_obs_angle"] < 2 * math.pi:
            pose = self.variables[pose_key]
            landmark = self.variables[landmark_key]
            
            dx = landmark[0] - pose[0]
            dy = landmark[1] - pose[1]
            bearing_to_landmark = wrap_angle(math.atan2(dy, dx) - pose[2])
            
            half_fov = self.cfg["max_obs_angle"] / 2
            if abs(bearing_to_landmark) > half_fov:
                return False
        
        return True

    def _process_single_measurement(self, measurement: Dict[str, Any]) -> bool:
        print(f"[DEBUG] _process_single_measurement called with: {measurement}")  # ← 新增
        
        m_type = measurement.get("type", "robot_lm")
        print(f"[DEBUG] m_type={m_type}")  # ← 新增
        
        if m_type not in ["robot_lm", "br_lm", "bearing_range"]:
            print(f"[DEBUG] FAIL: unsupported type {m_type}")
            return False

        robot_id = measurement.get("robot")
        time_step = measurement.get("time")
        landmark_id = measurement.get("id")
        print(f"[DEBUG] Extracted params: robot_id={robot_id}, time_step={time_step}, landmark_id={landmark_id}")  # ← 新增

        if not self._validate_measurement_params(robot_id, time_step, landmark_id):
            print(f"[DEBUG] FAIL: invalid measurement params: robot={robot_id}, time={time_step}, id={landmark_id}")
            print(f"[DEBUG] robot_count={self._robot_count}, landmark_count={self._landmark_count}")  # ← 新增
            if robot_id is not None and 0 <= robot_id < self._robot_count:
                print(f"[DEBUG] Robot path length for robot {robot_id}: {self.robot_paths[robot_id].shape[0]}")  # ← 新增
            return False

        obs_data = self._extract_observation_data(measurement)
        print(f"[DEBUG] obs_data={obs_data}")  # ← 新增
        
        if obs_data is None:
            print(f"[DEBUG] FAIL: obs_data is None in measurement: {measurement}")
            return False

        bearing, range_val = obs_data
        print(f"[DEBUG] bearing={bearing}, range_val={range_val}")  # ← 新增

        pose_key = self._prefixed(f"x{robot_id}_{time_step}")
        landmark_key = self._prefixed(f"l_{landmark_id}")
        print(f"[DEBUG] Generated keys: pose_key={pose_key}, landmark_key={landmark_key}")  # ← 新增

        if pose_key not in self.variables or landmark_key not in self.variables:
            print(f"[DEBUG] FAIL: variable not found: {pose_key} or {landmark_key}")
            print(f"[DEBUG] pose_key in variables: {pose_key in self.variables}")
            print(f"[DEBUG] landmark_key in variables: {landmark_key in self.variables}")
            print(f"[DEBUG] All variable keys: {list(self.variables.keys())}")
            return False

        if self.cfg["enable_visibility_check"]:
            visibility_result = self._check_visibility(pose_key, landmark_key, range_val)
            print(f"[DEBUG] visibility_check enabled, result={visibility_result}")  # ← 新增
            if not visibility_result:
                print(f"[DEBUG] FAIL: not visible: pose_key={pose_key}, landmark_key={landmark_key}, range={range_val}")
                return False

        # 到这一步如果没return就会加因子
        print(f"[DEBUG] All checks passed, creating factor...")  # ← 新增
        
        noise_cov = self._get_observation_noise_matrix(range_val)
        measurement_vector = np.array([bearing, range_val], dtype=np.float64)

        factor = BearingRangeUTFactor(
            pose_key, landmark_key, measurement_vector, noise_cov,
            mode=self.cfg["ut_mode"],
            alpha=self.cfg["ut_alpha"],
            beta=self.cfg["ut_beta"], 
            kappa=self.cfg["ut_kappa"],
            linear_threshold=self.cfg["ut_linear_threshold"],
            residual_sigma_thresh=self.cfg["ut_residual_threshold"]
        )

        self.factors.append(factor)
        print(f"[DEBUG] Add factor: {factor}")
        print(f"[DEBUG] SUCCESS: Added observation factor ({pose_key}, {landmark_key})")
        # ====== 观测Jacobian与噪声打印 ======
        try:
            # mu_dict用于线性化，实际运行传入当前self.variables即可
            H_pose, H_lm = factor.compute_jacobian(self.variables[pose_key], self.variables[landmark_key])
            print(f"[DIAG] Jacobian H_pose (观测):\n{H_pose}")
            print(f"[DIAG] Jacobian H_lm (地标):\n{H_lm}")
            print(f"[DIAG] H_pose范数={np.linalg.norm(H_pose):.3g}，最小奇异值={np.linalg.svd(H_pose, compute_uv=False).min():.3g}")
            print(f"[DIAG] H_lm范数={np.linalg.norm(H_lm):.3g}，最小奇异值={np.linalg.svd(H_lm, compute_uv=False).min():.3g}")
        except Exception as e:
            print(f"[DIAG] Jacobian computation failed: {e}")

        print(f"[DIAG] Obs noise covariance:\n{noise_cov}")
        print(f"[DIAG] Obs info eigenvalues: {np.linalg.eigvalsh(np.linalg.inv(noise_cov))}")
        # ======================================
        return True


    def _safe_none_check(self, *values) -> bool:
        """✅ 安全的None检查，避免numpy数组的truth value问题"""
        for val in values:
            try:
                if val is None:
                    return True
            except ValueError:
                # numpy数组可能抛出"truth value is ambiguous"
                continue
        return False

    def _add_loop_closure_factors(self, loop_closures: List[Dict[str, Any]]):
        """添加环路闭合因子"""
        logger.info("Adding %d loop closure factors...", len(loop_closures))
        
        for closure in loop_closures:
            try:
                robot1_id = closure.get("robot1", closure.get("from_robot"))
                time1 = closure.get("time1", closure.get("from_time"))
                robot2_id = closure.get("robot2", closure.get("to_robot"))
                time2 = closure.get("time2", closure.get("to_time"))
                
                relative_pose = closure.get("relative_pose")
                information = closure.get("information_matrix")
                
                # ✅ 使用安全的None检查
                if self._safe_none_check(robot1_id, time1, robot2_id, time2, relative_pose):
                    continue
                
                pose1_key = self._prefixed(f"x{robot1_id}_{time1}")
                pose2_key = self._prefixed(f"x{robot2_id}_{time2}")
                
                if pose1_key not in self.variables or pose2_key not in self.variables:
                    continue
                
                if information is None:
                    sigma_xy = self.cfg["odom_sigma_xy"] * 2
                    sigma_theta = self.cfg["odom_sigma_theta"] * 2
                    information = np.diag([1/sigma_xy**2, 1/sigma_xy**2, 1/sigma_theta**2])
                
                information = np.array(information, dtype=np.float64)
                
                # ✅ 使用factor_ut的数值验证
                if self.cfg["enable_strict_validation"]:
                    try:
                        validate_matrix_properties(information, "loop_closure_information")
                    except ValueError as e:
                        logger.warning("Invalid loop closure information matrix: %s", str(e))
                        information = ensure_positive_definite(information, numerical_config.regularization)
                
                factor = LoopClosureFactor(
                    pose1_key, pose2_key, 
                    np.array(relative_pose, dtype=np.float64),
                    information
                )
                
                self.factors.append(factor)
                print(f"[DEBUG] Add factor: {factor}")
                self.stats.loop_closure_factors += 1
                
            except Exception as e:
                logger.warning("Failed to add loop closure: %s", str(e))

    def _add_global_anchor_constraints(self):
        """添加全局锚点约束"""
        strategy = self.cfg["anchor_strategy"]
        
        if strategy == "landmarks":
            self._add_landmark_anchors()
        elif strategy == "poses":
            self._add_pose_anchors()
        elif strategy == "mixed":
            self._add_mixed_anchors()

    def _add_landmark_anchors(self):
        """添加地标锚点"""
        if self._landmark_count == 0:
            logger.warning("No landmarks available, falling back to pose anchors")
            self._add_pose_anchors()
            return
        if self._landmark_count < 2:
            self._add_pose_anchors()
            return
        
        max_distance = 0
        best_pair = (0, 1)
        
        for i in range(self._landmark_count):
            for j in range(i + 1, self._landmark_count):
                distance = np.linalg.norm(self._landmarks[i] - self._landmarks[j])
                if distance > max_distance:
                    max_distance = distance
                    best_pair = (i, j)
        
        anchor_sigma = np.array([2.0, 2.0])
        
        for lid in best_pair:
            key = self._prefixed(f"l_{lid}")
            if key in self.variables:
                factor = PriorFactor(key, self._landmarks[lid].copy(), anchor_sigma)
                self.factors.append(factor)
                print(f"[DEBUG] Add factor: {factor}")
                self.stats.prior_factors += 1
        
        logger.info("Added landmark anchors: l_%d, l_%d", best_pair[0], best_pair[1])

    def _add_pose_anchors(self):
        """
        强制每个机器人添加刚性pose anchor（极小sigma），用于debug信息矩阵正定性问题。
        建议debug阶段仅用pose anchor（禁用landmark anchor）。
        """
        anchors_added = 0
        for rid in range(self._robot_count):
            pose_key = self._prefixed(f"x{rid}_0")
            if pose_key in self.variables:
                # 检查是否已经有对这个变量的先验，避免重复
                has_prior = any(isinstance(f, PriorFactor) and f.var == pose_key for f in self.factors)
                if has_prior:
                    logger.info(f"已跳过{pose_key}的锚点(已有先验)")
                    continue
                # 强约束：刚性anchor（调小sigma，数值要正，不能为0，推荐0.05或更小）
                anchor_sigma = np.array([0.05, 0.05, 0.05], dtype=np.float64)
                factor = PriorFactor(pose_key, self.variables[pose_key].copy(), anchor_sigma)
                self.factors.append(factor)
                print(f"[DEBUG] Add factor: {factor}")
                self.stats.prior_factors += 1
                anchors_added += 1
                logger.info(f"Added strong pose anchor: {pose_key} (sigma={anchor_sigma.tolist()})")
            else:
                logger.warning(f"变量{pose_key}不存在，无法添加pose anchor！")
        if anchors_added == 0:
            logger.error("未添加任何pose anchor，可能会导致信息矩阵非正定。")

    
    def _add_mixed_anchors(self):
        """添加混合锚点"""
        if self._landmark_count >= 1:
            landmark_key = self._prefixed("l_0")
            if landmark_key in self.variables:
                anchor_sigma = np.array([1.0, 1.0])
                factor = PriorFactor(landmark_key, self._landmarks[0].copy(), anchor_sigma)
                self.factors.append(factor)
                print(f"[DEBUG] Add factor: {factor}")
                self.stats.prior_factors += 1
        
        first_pose = self._prefixed("x0_0")
        if first_pose in self.variables:
            pose = self.variables[first_pose].copy()
            pose[:2] = 0
            anchor_sigma = np.array([10.0, 10.0, math.radians(1.0)])
            factor = PriorFactor(first_pose, pose, anchor_sigma)
            self.factors.append(factor)
            print(f"[DEBUG] Add factor: {factor}")
            self.stats.prior_factors += 1
        
        logger.info("Added mixed anchors")

    def _validate_graph_connectivity(self):
        """验证图连接性"""
        connected_vars = set()
        
        for factor in self.factors:
            if isinstance(factor, PriorFactor):
                connected_vars.add(factor.var)
            elif isinstance(factor, OdometryFactor):
                connected_vars.add(factor.v1)
                connected_vars.add(factor.v2)
            elif isinstance(factor, BearingRangeUTFactor):
                connected_vars.add(factor.pose_key)
                connected_vars.add(factor.lm_key)
            elif isinstance(factor, PoseToPoseUTFactor):  # ✅ 新增
                connected_vars.add(factor.pose1_key)
                connected_vars.add(factor.pose2_key)
            elif isinstance(factor, LoopClosureFactor):
                connected_vars.add(factor.pose1_key)
                connected_vars.add(factor.pose2_key)
        
        all_vars = set(self.variables.keys())
        disconnected = all_vars - connected_vars
        
        self.stats.connected_variables = len(connected_vars)
        self.stats.disconnected_variables = len(disconnected)
        
        if disconnected:
            logger.warning("Found %d disconnected variables", len(disconnected))

    def _reset_state(self):
        """重置状态"""
        self.factors.clear()
        self.variables.clear()
        self._local_cache["noise_matrices"].clear()
        self._local_cache["sigma_arrays"].clear()
        self._landmarks = None
        self.robot_paths = None
        self._robot_count = 0
        self._landmark_count = 0
        self.stats = GraphBuildStats()

    def _monitor_section(self, section_name: str):
        """性能监控上下文管理器"""
        return self._performance_monitor.section(section_name) if self._performance_monitor else DummyContext()

    # -------------------------------------------------------------------------
    # ✅ 新增：诊断和监控方法
    # -------------------------------------------------------------------------
    
    def get_performance_diagnostics(self) -> Dict[str, Any]:
        """获取性能诊断信息"""
        cache_stats = get_global_cache_stats()
        
        return {
            "build_stats": {
                "build_time": self.stats.build_time,
                "total_factors": self.stats.total_factors,
                "vectorization_speedup": self.stats.vectorization_speedup,
                "batch_processing_efficiency": self.stats.batch_processing_efficiency,
                "cache_hit_rate": self.stats.cache_hit_rate
            },
            "factor_breakdown": {
                "prior_factors": self.stats.prior_factors,
                "odometry_factors": self.stats.odometry_factors,
                "observation_factors": self.stats.observation_factors,
                "inter_robot_factors": self.stats.inter_robot_factors,
                "loop_closure_factors": self.stats.loop_closure_factors
            },
            "cache_performance": cache_stats,
            "memory_usage": {
                "local_cache_size": {
                    "noise_matrices": len(self._local_cache["noise_matrices"]),
                    "sigma_arrays": len(self._local_cache["sigma_arrays"])
                }
            },
            "configuration": {
                "vectorization_enabled": self.cfg["enable_true_vectorization"],
                "batch_processing_enabled": self.cfg["enable_batch_processing"],
                "cache_coordination_enabled": self.cfg["enable_cache_coordination"],
                "ut_mode": self.cfg["ut_mode"],
                "inter_robot_obs_mode": self.cfg["inter_robot_obs_mode"]
            }
        }

    def get_variable_types(self) -> Dict[str, str]:
        """获取变量类型映射"""
        var_types = {}
        for var_name in self.variables:
            if var_name.startswith('x'):
                var_types[var_name] = 'pose'
            elif var_name.startswith('l'):
                var_types[var_name] = 'landmark'
            else:
                var_types[var_name] = 'unknown'
        return var_types

    def save_diagnostics(self, filepath: str) -> bool:
        """保存诊断信息到文件"""
        try:
            import json
            diagnostics = self.get_performance_diagnostics()
            with open(filepath, 'w') as f:
                json.dump(diagnostics, f, indent=2, default=str)
            logger.info("Saved diagnostics to %s", filepath)
            return True
        except Exception as e:
            logger.error("Failed to save diagnostics: %s", str(e))
            return False

    # -------------------------------------------------------------------------
    # ✅ 新增：信息矩阵构建方法（实现补丁2）
    # -------------------------------------------------------------------------
    
    def build_information_matrix(self, mu_dict: Optional[Dict[str, np.ndarray]] = None,
                            Sigma_dict: Optional[Dict[Any, np.ndarray]] = None,
                            return_sparse: bool = True) -> Union[np.ndarray, Any]:
        """
        构建全局信息矩阵 Λ，实现自动镜像补全和正定化
        
        Args:
            mu_dict: 当前均值估计（如果为None，使用self.variables）
            Sigma_dict: 当前协方差估计（如果为None，使用单位阵）
            return_sparse: 是否返回稀疏矩阵
            
        Returns:
            信息矩阵（稀疏或稠密格式）
        """
        import scipy.sparse
        from collections import defaultdict
        
        logger.info("Building global information matrix...")
        
        # 使用当前变量作为默认值
        if mu_dict is None:
            mu_dict = self.variables.copy()
        
        # 构建默认协方差
        if Sigma_dict is None:
            Sigma_dict = {}
            for var_key, var_val in mu_dict.items():
                dim = len(var_val)
                Sigma_dict[var_key] = np.eye(dim) * 0.1  # 默认不确定性
        
        # 收集所有变量和维度
        var_order = sorted(mu_dict.keys())
        var_dims = {v: len(mu_dict[v]) for v in var_order}
        var_indices = {}
        current_idx = 0
        
        for var in var_order:
            var_indices[var] = (current_idx, current_idx + var_dims[var])
            current_idx += var_dims[var]
        
        total_dim = current_idx
        
        # ✅ 使用字典累加信息块
        info_blocks = defaultdict(lambda: 0)  # 默认值为0
        
        # 处理每个因子
        for factor_idx, factor in enumerate(self.factors):
            try:
                # 获取因子的线性化信息块
                blocks = factor.linearize(mu_dict, Sigma_dict)
                assert isinstance(blocks, dict), \
                    f"{type(factor).__name__}.linearize must return dict, got {type(blocks)}"
                
                # 累加块，实现自动镜像补全
                for key, block in blocks.items():
                    if isinstance(key, tuple) and len(key) == 2:
                        # 标准二元组键 (vi, vj)
                        vi, vj = key
                        
                        # 累加正向块
                        info_blocks[(vi, vj)] += block
                        
                        # 如果不是对角块，且镜像尚未出现，则补transpose
                        if vi != vj and (vj, vi) not in blocks:
                            info_blocks[(vj, vi)] += block.T
                    
                    elif isinstance(key, str):
                        # 单变量键，处理为对角块 (Λ, η)
                        vi = key
                        if isinstance(block, tuple) and len(block) == 2:
                            Lambda, eta = block
                            prev = info_blocks[(vi, vi)]
                            if isinstance(prev, int) and prev == 0:
                                info_blocks[(vi, vi)] = (Lambda, eta)
                            else:
                                if isinstance(prev, tuple) and len(prev) == 2:
                                    prev_Lambda, prev_eta = prev
                                    info_blocks[(vi, vi)] = (prev_Lambda + Lambda, prev_eta + eta)
                                else:
                                    # 处理不一致的情况
                                    logger.warning(f"不一致的对角块格式: {vi}, 预期元组, 实际: {type(prev)}")
                                    if isinstance(prev, np.ndarray):
                                        # 假设之前是矩阵，现在是元组
                                        info_blocks[(vi, vi)] = (prev + Lambda, eta)
                                    else:
                                        # 安全回退
                                        info_blocks[(vi, vi)] = (Lambda, eta)
                        else:
                            # 单个矩阵，当作Λ处理
                            prev = info_blocks[(vi, vi)]
                            if isinstance(prev, int) and prev == 0:
                                info_blocks[(vi, vi)] = block
                            else:
                                if isinstance(prev, tuple) and len(prev) == 2:
                                    # 之前是(Λ,η)元组，现在是矩阵
                                    prev_Lambda, prev_eta = prev
                                    info_blocks[(vi, vi)] = (prev_Lambda + block, prev_eta)
                                else:
                                    # 之前也是矩阵
                                    info_blocks[(vi, vi)] += block
                    
                    else:
                        logger.warning(f"未知块键格式: {key}, 类型: {type(key)}")
                        continue
                        
            except Exception as e:
                logger.warning("Factor %d linearization failed: %s", factor_idx, str(e))
                continue
        
        # 构建稀疏矩阵
        row_indices = []
        col_indices = []
        data_values = []
        
        for (vi, vj), block in info_blocks.items():
            if vi not in var_indices or vj not in var_indices:
                continue
                
            i_start, i_end = var_indices[vi]
            j_start, j_end = var_indices[vj]
            
            # 处理对角块 (Λ, η)，只取Λ部分
            if vi == vj and isinstance(block, tuple) and len(block) == 2:
                block = block[0]  # 只取Λ
            
            # 确保是numpy数组
            block = np.asarray(block)
            
            # 添加块到稀疏矩阵数据
            for local_i in range(block.shape[0]):
                for local_j in range(block.shape[1]):
                    if abs(block[local_i, local_j]) > 1e-12:  # 跳过零元素
                        row_indices.append(i_start + local_i)
                        col_indices.append(j_start + local_j)
                        data_values.append(block[local_i, local_j])
        
        # 创建稀疏矩阵
        Lambda_sparse = scipy.sparse.coo_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(total_dim, total_dim)
        ).tocsr()
        
        # 对称化和正定化
        Lambda_dense = Lambda_sparse.toarray()
        Lambda_dense = 0.5 * (Lambda_dense + Lambda_dense.T)  # 强制对称
        Lambda_dense = ensure_positive_definite(Lambda_dense, numerical_config.min_eigenvalue)  # 确保正定
        
        # 验证结果
        min_eig = np.linalg.eigvalsh(Lambda_dense).min()
        logger.info("Information matrix built: dim=%d, min_eigenvalue=%.6e", total_dim, min_eig)
        
        # 返回适当格式
        if return_sparse:
            return scipy.sparse.csr_matrix(Lambda_dense)
        else:
            return Lambda_dense
    
    
    def build_information_matrix_dict(self,
            mu_dict: Optional[Dict[str, np.ndarray]] = None,
            Sigma_dict: Optional[Dict[Any, np.ndarray]] = None
        ) -> Dict[Tuple[str, str], np.ndarray]:
        """
        通过先构建完整稠密信息矩阵 Λ，再切片得到每个变量及变量对的块字典表示。
        """
        # 1) 得到完整的稠密信息矩阵
        Lambda_dense = self.build_information_matrix(mu_dict, Sigma_dict, return_sparse=False)

        # 2) 计算每个变量在大矩阵中的索引区间
        var_order = sorted(self.variables.keys())
        var_dims = {v: len(self.variables[v]) for v in var_order}
        var_indices: Dict[str, Tuple[int,int]] = {}
        idx = 0
        for v in var_order:
            var_indices[v] = (idx, idx + var_dims[v])
            idx += var_dims[v]

        # 3) 按照索引区间切片，填充块字典
        blocks: Dict[Tuple[str,str], np.ndarray] = {}
        for vi in var_order:
            i0, i1 = var_indices[vi]
            # 对角块一定要有
            blocks[(vi, vi)] = Lambda_dense[i0:i1, i0:i1].copy()
            # 处理上三角的交叉块，并同时添加对称的下三角
            for vj in var_order:
                if vj <= vi:
                    continue
                j0, j1 = var_indices[vj]
                sub = Lambda_dense[i0:i1, j0:j1]
                if not np.allclose(sub, 0):
                    blocks[(vi, vj)] = sub.copy()
                    blocks[(vj, vi)] = sub.T.copy()
        return blocks

    
    def verify_information_matrix(self, info_matrix: Union[np.ndarray, Dict]) -> Dict[str, Any]:
        """
        验证信息矩阵的性质
        
        Args:
            info_matrix: 信息矩阵（稠密、稀疏或块字典格式）
            
        Returns:
            验证结果字典
        """
        import scipy.sparse
        
        # 转换为稠密格式进行分析
        if isinstance(info_matrix, dict):
            # 从块字典重建矩阵
            var_order = sorted(self.variables.keys())
            var_dims = {v: len(self.variables[v]) for v in var_order}
            var_indices = {}
            current_idx = 0
            
            for var in var_order:
                var_indices[var] = (current_idx, current_idx + var_dims[var])
                current_idx += var_dims[var]
            
            total_dim = current_idx
            dense_matrix = np.zeros((total_dim, total_dim))
            
            for (vi, vj), block in info_matrix.items():
                if vi in var_indices and vj in var_indices:
                    i_start, i_end = var_indices[vi]
                    j_start, j_end = var_indices[vj]
                    dense_matrix[i_start:i_end, j_start:j_end] = block
        elif scipy.sparse.issparse(info_matrix):
            dense_matrix = info_matrix.toarray()
        else:
            dense_matrix = np.asarray(info_matrix)
        
        # 计算验证指标
        results = {}
        
        # 1. 对称性
        symmetry_error = np.max(np.abs(dense_matrix - dense_matrix.T))
        results["is_symmetric"] = symmetry_error < 1e-10
        results["symmetry_error"] = float(symmetry_error)
        
        # 2. 特征值分析
        eigenvalues = np.linalg.eigvalsh(dense_matrix)
        results["min_eigenvalue"] = float(eigenvalues.min())
        results["max_eigenvalue"] = float(eigenvalues.max())
        results["condition_number"] = float(eigenvalues.max() / eigenvalues.min())
        results["is_positive_definite"] = eigenvalues.min() > 1e-12
        
        # 3. 稀疏性
        total_elements = dense_matrix.size
        nonzero_elements = np.count_nonzero(np.abs(dense_matrix) > 1e-12)
        results["sparsity"] = 1.0 - nonzero_elements / total_elements
        results["nonzero_elements"] = int(nonzero_elements)
        
        # 4. 块结构分析
        if isinstance(info_matrix, dict):
            results["num_blocks"] = len(info_matrix)
            diagonal_blocks = sum(1 for (vi, vj) in info_matrix if vi == vj)
            results["diagonal_blocks"] = diagonal_blocks
            results["off_diagonal_blocks"] = len(info_matrix) - diagonal_blocks
        
        # 5. 矩阵范数
        results["frobenius_norm"] = float(np.linalg.norm(dense_matrix, 'fro'))
        results["spectral_norm"] = float(np.linalg.norm(dense_matrix, 2))
        
        return results
# =====================================================================
# ✅ 性能监控辅助类
# =====================================================================

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.sections = {}
    
    def section(self, name: str):
        """监控段落上下文管理器"""
        return PerformanceSection(self, name)

class PerformanceSection:
    """性能监控段落"""
    
    def __init__(self, monitor: PerformanceMonitor, name: str):
        self.monitor = monitor
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time
        self.monitor.sections[self.name] = duration

class DummyContext:
    """虚拟上下文管理器"""
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# =====================================================================
# ✅ 增强的工厂函数
# =====================================================================

def create_single_robot_graph(trajectory: np.ndarray, landmarks: np.ndarray,
                             measurements: List[Dict], 
                             config: Optional[Dict] = None) -> Tuple[List, Dict]:
    """
    单机器人SLAM工厂函数
    
    Args:
        trajectory: 机器人轨迹 (T, 3)
        landmarks: 地标位置 (L, 2)
        measurements: 观测测量
        config: 可选配置
        
    Returns:
        (factors, variables): 因子图组件
    """
    builder = GBPGraphBuilder(config)
    return builder.build_single_robot(trajectory, landmarks, measurements)

def create_multi_robot_graph(robot_trajectories: List[np.ndarray], 
                            landmarks: np.ndarray,
                            measurements: List[Dict],
                            loop_closures: Optional[List[Dict]] = None,
                            inter_robot_obs: Optional[List[Dict]] = None,  # ✅ 新增
                            config: Optional[Dict] = None) -> Tuple[List, Dict]:
    """
    多机器人SLAM工厂函数
    
    Args:
        robot_trajectories: 机器人轨迹列表
        landmarks: 共享地标位置  
        measurements: 所有测量
        loop_closures: 可选的环路闭合约束
        inter_robot_obs: ✅ 新增：机器人间观测
        config: 可选配置
        
    Returns:
        (factors, variables): 因子图组件
    """
    builder = GBPGraphBuilder(config)
    return builder.build(robot_trajectories, landmarks, measurements, 
                        loop_closures, inter_robot_obs)

def create_high_performance_graph(*args, **kwargs) -> Tuple[List, Dict]:
    """
    ✅ 新增：高性能因子图创建
    
    启用所有性能优化选项
    """
    perf_flags = {
        "enable_true_vectorization": True,
        "enable_batch_processing": True,
        "enable_cache_coordination": True,
        "enable_factor_reordering": True,
        "cache_warmup": True,
        "enable_performance_monitoring": True
    }
    
    # 合并用户配置
    user_config = kwargs.pop("config", {})
    config = {**perf_flags, **user_config}
    
    return create_multi_robot_graph(*args, config=config, **kwargs)

# ------------------------------------------------------------------
# Helper: export builder 结果到 GBPGraph
# ------------------------------------------------------------------
def export_to_gbp(builder: GBPGraphBuilder, build_info_matrix: bool = False):
    """将 GBPGraphBuilder 转换成 GBPGraph 所需格式"""
    if not isinstance(builder, GBPGraphBuilder):
        raise TypeError("export_to_gbp expects a GBPGraphBuilder instance")

    # ---- 1. variables & angle_vars --------------------------------
    variables = builder.variables.copy()
    
    # 根据实际维度确定角度索引
    angle_vars = {}
    for k, v in variables.items():
        if k.startswith("x"):  # 姿态变量
            if v.size == 3:      # SE(2): [x, y, theta]
                angle_vars[k] = [2]
            elif v.size == 6:    # SE(3): [x, y, z, roll, pitch, yaw]  
                angle_vars[k] = [3, 4, 5]
            elif v.size == 7:    # SE(3) with quaternion: [x, y, z, qx, qy, qz, qw]
                angle_vars[k] = [6]
            else:
                logger.warning(f"Unknown pose dimension for {k}: {v.size}, skipping angle vars")
        # 地标变量通常没有角度

    # ---- 2. factor 适配 ------------------------------------------
    factors = [GBPFactorAdapter(f) for f in builder.factors]

    # ---- 3. Priors : 检查是否已有足够的 priors ----
    # ---- 3. Priors : export every real PriorFactor into VarNode priors ----
    priors = {}
    for f in builder.factors:
        if isinstance(f, PriorFactor):
            var = f.var
            mu0 = builder.variables[var]
            # re‑linearize the PriorFactor to get its (L0, η0)
            adapter = GBPFactorAdapter(f)
            blocks  = adapter.linearize(builder.variables, {})
            if var not in blocks:
                continue
            L0, η0 = blocks[var]
            # turn info form back into covariance & σ
            cov0   = np.linalg.inv(L0)
            σ0     = np.sqrt(np.diag(cov0))
            μ0_new = cov0.dot(η0)
            priors[var] = (μ0_new, σ0)
    
        # ---- Debug: 导出因子数和类型 ----
    print(f"[DEBUG] Exporting {len(factors)} factors to GBPGraph")
    if len(factors) > 0:
        print(f"[DEBUG] Factor types: {[type(f) for f in factors[:10]]} ...")
    else:
        print("[ERROR] No factors to export! All factors filtered out!")

    # ---- 4. 可选：构建信息矩阵字典 ----
    if build_info_matrix:
        info_matrix_dict = builder.build_information_matrix_dict()
        return variables, factors, priors, angle_vars, info_matrix_dict
    else:
        return variables, factors, priors, angle_vars


# 导出符号
__all__ = [
    'GBPGraphBuilder', 'GraphBuildStats', 'GBPFactorAdapter', 
    'create_single_robot_graph', 'create_multi_robot_graph', 'create_high_performance_graph',
    'export_to_gbp',  
    'PerformanceMonitor'
]