import numpy as np, gtsam, math, itertools

def pose_rmse(result: gtsam.Values, gt_poses, robot=0):
    errs=[]
    for t in range(gt_poses.shape[1]):
        key = gtsam.symbol('x', robot*10000+t)
        if not result.exists(key): continue
        est = result.atPose2(key)
        gt  = gt_poses[robot,t]
        dx,dy = est.x()-gt[0], est.y()-gt[1]
        dth   = math.atan2(math.sin(est.theta()-gt[2]),
                           math.cos(est.theta()-gt[2]))
        errs.append(dx*dx+dy*dy+dth*dth)
    return math.sqrt(np.mean(errs))

def landmark_rmse(result, gt_lms):
    errs=[]
    for i, gt in enumerate(gt_lms):
        key=gtsam.symbol('l',i)
        if not result.exists(key): continue
        p=result.atPoint2(key)
        errs.append((p.x()-gt[0])**2+(p.y()-gt[1])**2)
    return math.sqrt(np.mean(errs))
