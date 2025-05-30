import numpy as np
from typing import List, Tuple

def range_bearing(robot_xy: np.ndarray, lm_xy: np.ndarray) -> Tuple[float, float]:
    """理想距离 / 方位角（无噪声）"""
    delta   = lm_xy - robot_xy
    rng     = np.linalg.norm(delta)
    bearing = np.arctan2(delta[1], delta[0])
    return rng, bearing


def make_measurements(
    paths: np.ndarray,
    lms:   np.ndarray,
    *,
    sigma_r:   float = 0.30,
    sigma_b:   float = np.deg2rad(2),
    max_range: float = 25.0,
    p_obs:     float = 1.0,            # ← 新增：可见后被采样的概率
    noise:     bool  = True,
    seed:      int   = None
) -> List[List[List[Tuple[int, np.ndarray]]]]:
    """
    返回:
        measurements[robot_idx][time_step] = [
            (lm_id, np.array([range, bearing])), ...
        ]
    """
    if paths.ndim != 3 or paths.shape[2] != 2:
        raise ValueError("paths 应为 (R,T,2)")
    if lms.ndim != 2 or lms.shape[1] != 2:
        raise ValueError("lms 应为 (L,2)")
    if not (0.0 < p_obs <= 1.0):
        raise ValueError("p_obs 必须在 (0,1]")

    rng = np.random.default_rng(seed)
    R, T, _ = paths.shape
    L       = lms.shape[0]

    measurements: List[List[List[Tuple[int, np.ndarray]]]] = []

    for r in range(R):
        robot_meas: List[List[Tuple[int, np.ndarray]]] = []
        for t in range(T):
            robot_pos   = paths[r, t]                 # (2,)
            deltas      = lms - robot_pos             # (L,2)
            dists       = np.linalg.norm(deltas, axis=1)

            # 1) 视距条件
            visible_idx = np.where(dists < max_range)[0]
            if visible_idx.size == 0:
                robot_meas.append([])
                continue

            # 2) 抽样概率  p_obs
            keep_mask   = rng.random(visible_idx.size) < p_obs
            kept_idx    = visible_idx[keep_mask]
            if kept_idx.size == 0:
                robot_meas.append([])
                continue

            # 3) 真值
            true_rng    = dists[kept_idx]
            true_brg    = np.arctan2(deltas[kept_idx,1], deltas[kept_idx,0])

            # 4) 噪声
            if noise:
                meas_rng = true_rng + rng.normal(0, sigma_r,  kept_idx.size)
                meas_brg = true_brg + rng.normal(0, sigma_b, kept_idx.size)
            else:
                meas_rng, meas_brg = true_rng, true_brg

            # 5) 打包
            obs = [
                (int(lm_id), np.array([d, b], dtype=float))
                for lm_id, d, b in zip(kept_idx, meas_rng, meas_brg)
            ]
            robot_meas.append(obs)
        measurements.append(robot_meas)

    return measurements
