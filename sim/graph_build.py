"""
GBP-Factor-Graph Builder
———————————
将 world.py 生成的真值轨迹 & measurement.py 观测
转换为 GBP / SPBP 因子图（factor list + 初值 dict）
"""

from __future__ import annotations
import numpy as np, logging
from typing import Dict, List, Tuple, Any, Optional

from algorithm.frontend.factor_ut import (
    BearingRangeUTFactor, OdometryFactor, PriorFactor
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GBPBuilder")


class GBPGraphBuilder:
    # ---------- 默认配置 ----------
    DEFAULT_CONFIG = {
        # 噪声
        "prior_sigma":         0.10,
        "odom_sigma":          0.10,
        "obs_sigma_range":     0.30,
        "obs_sigma_bearing":   np.deg2rad(2.0),
        # UT / SPBP
        "ut_mode":   "auto",   
        "ut_alpha":  0.30,
        "ut_kappa":  2.0,
        # 其它
        "max_obs_range":       25.0,
    }

    # ---------- 初始化 ----------
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.config     = self._validate_cfg(cfg or {})
        self.factors    : List[object]             = []
        self.variables  : Dict[str, np.ndarray]    = {}
        self._R_cache   : Optional[np.ndarray]     = None
        self._landmarks : Optional[np.ndarray]     = None   # 真值坐标，用于可见性检测

    # =================================================================
    # 公有  build()
    # =================================================================
    def build(self,
              robot_paths: np.ndarray,                  # (R,T,2)
              landmark_positions: np.ndarray,           # (L,2)
              measurements: List[List[List[Tuple[int, np.ndarray]]]]
              ) -> Tuple[List[object], Dict[str, np.ndarray]]:

        self._validate_inputs(robot_paths, landmark_positions, measurements)

        # Reset internal containers
        self.factors.clear()
        self.variables.clear()
        self._R_cache   = None
        self._landmarks = landmark_positions.copy()

        # 1. 地标变量
        self._add_landmarks(landmark_positions)

        # 2. 各机器人子图
        for rid in range(robot_paths.shape[0]):
            self._add_robot(rid,
                            robot_paths[rid],
                            measurements[rid])

        logger.info(
            f"✔ build() finished — factors: {len(self.factors)} , "
            f"vars: {len(self.variables)}")
        return self.factors, self.variables

    # =================================================================
    # 内部 – 每个机器人
    # =================================================================
    def _add_robot(self, rid: int,
                   poses: np.ndarray,                              # (T,2)
                   obs_data: List[List[Tuple[int, np.ndarray]]]) -> None:

        T = poses.shape[0]

        # -------- pose 变量节点 --------
        for t in range(T):
            self.variables[f"x{rid}_{t}"] = poses[t].copy()

        # -------- 先验因子 --------
        self.factors.append(
            PriorFactor(f"x{rid}_0", poses[0], self.config["prior_sigma"])
        )

        # -------- 里程计因子 --------
        for t in range(T - 1):
            delta = poses[t + 1] - poses[t]
            self.factors.append(
                OdometryFactor(f"x{rid}_{t}", f"x{rid}_{t + 1}",
                               delta, self.config["odom_sigma"])
            )

        # -------- 观测因子 --------
        Rmat = self._noise_R()
        for t, obs_list in enumerate(obs_data):
            pose_key = f"x{rid}_{t}"
            for lm_id, meas in obs_list:
                # 可见性二次确认（根据真值+阈值）
                if np.linalg.norm(self._landmarks[lm_id] - poses[t]) > self.config["max_obs_range"]:
                    continue

                self.factors.append(
                    BearingRangeUTFactor(
                        pose_var=pose_key,
                        lm_var=f"l_{lm_id}",
                        measurement=meas.astype(float),
                        R=Rmat,
                        mode=self.config["ut_mode"],
                        alpha=self.config["ut_alpha"],
                        kappa=self.config["ut_kappa"]
                    )
                )

    # =================================================================
    # 内部 – 地标节点
    # =================================================================
    def _add_landmarks(self, lms: np.ndarray) -> None:
        for lid, pos in enumerate(lms):
            self.variables[f"l_{lid}"] = pos.copy()

    # =================================================================
    # 内部 – 噪声协方差
    # =================================================================
    def _noise_R(self) -> np.ndarray:
        if self._R_cache is None:
            self._R_cache = np.diag([
                self.config["obs_sigma_range"] ** 2,
                self.config["obs_sigma_bearing"] ** 2
            ])
        return self._R_cache.copy()

    # =================================================================
    # 校验
    # =================================================================
    def _validate_cfg(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        C = self.DEFAULT_CONFIG.copy(); C.update(cfg)
        assert 0 < C["ut_alpha"] <= 1
        assert C["ut_kappa"] >= 0
        assert C["ut_mode"] in ("gbp", "spbp", "auto"), "ut_mode must be 'gbp' , 'spbp' or 'auto'"
        return C

    def _validate_inputs(self,
                         paths: np.ndarray,
                         lms: np.ndarray,
                         meas: List[List]) -> None:
        if paths.ndim != 3 or paths.shape[2] != 2:
            raise ValueError("paths shape must be (R,T,2)")
        if lms.ndim != 2 or lms.shape[1] != 2:
            raise ValueError("lms shape must be (L,2)")
        if len(meas) != paths.shape[0]:
            raise ValueError("measurements robot dimension mismatch")
        for rid, data in enumerate(meas):
            if len(data) != paths.shape[1]:
                raise ValueError(f"robot {rid} measurement T mismatch")

# =====================================================================
# quick test
# =====================================================================
if __name__ == "__main__":
    from sim.world import paths, landmarks
    from sim.measurement import make_measurements

    np.random.seed(1)
    lms   = landmarks(5, distribution="gaussian", sigma=[7, 7])
    pths  = paths(3, 30, motion_type="circular", random_radius=True)
    meas  = make_measurements(pths, lms, max_range=20.0, seed=1)

    # ① 自动模式（推荐体验）
    builder = GBPGraphBuilder({"ut_mode": "auto", "ut_alpha": 0.25})
    fac, var = builder.build(pths, lms, meas)

    # 统计各模式数量
    gbp_cnt  = sum(isinstance(f, BearingRangeUTFactor) and f.mode == "gbp"  for f in fac)
    spbp_cnt = sum(isinstance(f, BearingRangeUTFactor) and f.mode == "spbp" for f in fac)
    print(f"demo ok: {len(fac)} factors, {len(var)} vars "
          f"(GBP={gbp_cnt}, SPBP={spbp_cnt})")
