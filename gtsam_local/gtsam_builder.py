import gtsam, numpy as np, math
from typing import Dict, List

# ---------- 1. 键生成 ----------
def x_key(robot: int, t: int):
    return gtsam.symbol('x', robot*10000 + t)

def l_key(lm_id: int):
    return gtsam.symbol('l', lm_id)

# ---------- 2. 图 & 初值 ----------
def build_graph(dataset: Dict,
                use_isam: bool=False,
                robust: bool=False):
    graph   = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # ----- 噪声模型 -----
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.05,0.05,math.radians(2)]))
    odo_noise   = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.08,0.08,math.radians(1)]))
    br_noise    = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([math.radians(3),0.12]))
    inter_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.15,0.15,math.radians(2)]))

    if robust:
        def huber(s):
            return gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber(k=1.345), s)
        odo_noise  = huber(odo_noise)
        br_noise   = huber(br_noise)
        inter_noise= huber(inter_noise)

    # ---------- 3. 变量 & 先验 ----------
    poses_gt = dataset["poses"]       # shape (R,T,3)
    R,T = poses_gt.shape[:2]
    for r in range(R):
        key0 = x_key(r,0)
        graph.add(gtsam.PriorFactorPose2(
            key0,
            gtsam.Pose2(*poses_gt[r,0]),
            prior_noise))
        initial.insert(key0, gtsam.Pose2(*poses_gt[r,0]))

    # ---------- 4. 里程计 ----------
    for odo in dataset["odom"]:
        r = odo["robot"]; t1=odo["t_from"]; t2=odo["t_to"]
        key1, key2 = x_key(r,t1), x_key(r,t2)
        delta = gtsam.Pose2(*odo["delta"])
        graph.add(gtsam.BetweenFactorPose2(key1,key2,delta,odo_noise))
        if not initial.exists(key2):
            prev = initial.atPose2(key1)
            initial.insert(key2, prev.compose(delta))

    # ---------- 5. bearing-range ----------
    for m in dataset["obs"]:
        r,t,lm = m["robot"], m["time"], m["lm_id"]
        pkey = x_key(r,t); lkey = l_key(lm)
        bearing = gtsam.Rot2.fromAngle(m["bearing"])
        graph.add(gtsam.BearingRangeFactor2D(
            pkey,lkey,bearing,m["range"],br_noise))
        if not initial.exists(lkey):
            pose_est = initial.atPose2(pkey)
            initial.insert(lkey, pose_est.transformFrom(
                gtsam.Point2(m["range"]*math.cos(m["bearing"]),
                             m["range"]*math.sin(m["bearing"]))))

    # ---------- 6. inter-robot ----------
    for inter in dataset.get("inter", []):
        k1 = x_key(inter["robot1"], inter["t1"])
        k2 = x_key(inter["robot2"], inter["t2"])
        graph.add(gtsam.BetweenFactorPose2(
            k1,k2,gtsam.Pose2(*inter["delta"]), inter_noise))

    # ---------- 7. 求解 ----------
    if use_isam:
        isam = gtsam.ISAM2()
        isam.update(graph, initial)
        result = isam.calculateEstimate()
    else:
        params = gtsam.LevenbergMarquardtParams()
        params.setLinearSolverType("MULTIFRONTAL_QR")
        result = gtsam.LevenbergMarquardtOptimizer(
            graph, initial, params).optimize()

    return result
