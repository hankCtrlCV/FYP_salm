import gtsam, numpy as np, pandas as pd
from gtsam.symbol_shorthand import X, L     # X(i) pose_i,  L(j) landmark_j

def run_gtsam(ds_dir: str):
    # 1. 读取 CSV
    odom = pd.read_csv(f"{ds_dir}/odom.csv")
    obs  = pd.read_csv(f"{ds_dir}/obs.csv")
    lms  = pd.read_csv(f"{ds_dir}/landmarks_gt.csv")

    graph = gtsam.NonlinearFactorGraph()
    init  = gtsam.Values()

    # 2. 先验：固定 x0
    init.insert(X(0), gtsam.Pose2(0,0,0))
    graph.add( gtsam.PriorFactorPose2(X(0), gtsam.Pose2(0,0,0), prior_noise) )

    # 3. 里程计 BetweenFactorPose2
    for _, row in odom.iterrows():
        i, j = int(row['from']), int(row['to'])
        dpose = gtsam.Pose2(row.dx, row.dy, row.dtheta)
        graph.add( gtsam.BetweenFactorPose2(X(i), X(j), dpose, odom_noise) )
        # 初值：用 dead-reckoning 递推
        if not init.exists(X(j)):
            init.insert(X(j), init.atPose2(X(i)).compose(dpose))

    # 4. 地标初值（可随机或量测反解）
    for _, lm in lms.iterrows():
        init.insert(L(int(lm.id)), gtsam.Point2(lm.x, lm.y))

    # 5. BearingRangeFactor2D
    for _, row in obs.iterrows():
        pose_id, lm_id = int(row.pose_id), int(row.lm_id)
        graph.add( gtsam.BearingRangeFactor2D(
            X(pose_id), L(lm_id),
            gtsam.Rot2(row.bearing), row['range'],
            br_noise() ) )

    # 6. LM optimize
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("ERROR")
    result = gtsam.LevenbergMarquardtOptimizer(graph, init, params).optimize()

    # 7. 提取结果为 numpy 方便对比
    N = odom[['from','to']].values.max()+1
    traj_hat = np.vstack([ result.atPose2(X(i)).vector() for i in range(N) ])
    lms_hat  = np.vstack([ result.atPoint2(L(j)).vector() for j in lms.id ])

    extra = {"iters": result.iterations(), "error": graph.error(result)}
    return traj_hat, lms_hat, extra
