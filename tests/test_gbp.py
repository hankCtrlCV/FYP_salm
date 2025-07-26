# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 替换成你自己实现的那个 GBPGraph
from algorithm.frontend.gbp import GBPGraph

# --- 更真实的 SLAM 演示: 单个机器人多个位姿与路标 ---
np.random.seed(0)

# 参数
num_poses = 6
num_landmarks = 10

# 生成地面真值机器人轨迹（圆形）
angles = np.linspace(0, 2*np.pi, num_poses, endpoint=False)
poses_gt = [np.array([5*np.cos(a), 5*np.sin(a), a]) for a in angles]

# 生成随机路标
landmarks_gt = [np.random.uniform(-10, 10, size=2) for _ in range(num_landmarks)]

# 构造带噪声的初始猜测
variables = {}
for i, p in enumerate(poses_gt):
    variables[f'x{i}'] = p + np.random.randn(3)*0.1
for j, lm in enumerate(landmarks_gt):
    variables[f'l{j}'] = lm + np.random.randn(2)*0.5

# --- 因子定义 ---
class PosePriorFactor:
    def __init__(self, var, mean, sigma):
        self.var = var
        self.mean = mean
        self.sigma2 = sigma**2
        self._f = self
    def _get_dim(self, key): return 3
    def linearize(self, mu, cov):
        L = np.diag(1.0/self.sigma2)
        eta = L @ self.mean
        return {self.var: (L, eta)}

class LandmarkPriorFactor:
    def __init__(self, var, mean, sigma):
        self.var = var
        self.mean = mean
        self.sigma2 = sigma**2
        self._f = self
    def _get_dim(self, key): return 2
    def linearize(self, mu, cov):
        L = np.diag(1.0/self.sigma2)
        eta = L @ self.mean
        return {self.var: (L, eta)}

class OdometryFactor:
    def __init__(self, i, j, delta, sigma):
        self.i, self.j = i, j
        self.delta = delta
        self.sigma2 = sigma**2
        self._f = self
    def _get_dim(self, key): return 3
    def linearize(self, mu, cov):
        xi, xj = mu[self.i], mu[self.j]
        dx = xj - xi - self.delta
        dx[2] = np.arctan2(np.sin(dx[2]), np.cos(dx[2]))
        H_i, H_j = -np.eye(3), np.eye(3)
        Rinv = np.diag(1.0/self.sigma2)
        Lii = H_i.T @ Rinv @ H_i
        Ljj = H_j.T @ Rinv @ H_j
        Lij = H_i.T @ Rinv @ H_j
        etai = H_i.T @ Rinv @ dx
        etaj = H_j.T @ Rinv @ dx
        return {
            self.i: (Lii, etai),
            self.j: (Ljj, etaj),
            (self.i, self.j): Lij,
            (self.j, self.i): Lij.T
        }

class BearingRangeFactor:
    def __init__(self, i, j, meas, sigma):
        self.i, self.j = i, j
        self.z = meas
        self.sigma2 = sigma**2
        self._f = self
    def _get_dim(self, key):
        return 2 if key.startswith('l') else 3
    def linearize(self, mu, cov):
        xi, lj = mu[self.i], mu[self.j]
        dx = lj[:2] - xi[:2]
        q = dx.dot(dx); r = np.sqrt(q)
        bearing = np.arctan2(dx[1], dx[0]) - xi[2]
        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
        h = np.array([bearing, r])
        # 雅可比
        H_i = np.zeros((2,3)); H_j = np.zeros((2,2))
        H_i[0,0], H_i[0,1], H_i[0,2] =  dx[1]/q, -dx[0]/q, -1
        H_i[1,0], H_i[1,1]           = -dx[0]/r, -dx[1]/r
        H_j[0,0], H_j[0,1]           = -H_i[0,0], -H_i[0,1]
        H_j[1,0], H_j[1,1]           =  dx[0]/r,  dx[1]/r
        Rinv = np.diag(1.0/self.sigma2)
        Lii = H_i.T @ Rinv @ H_i
        Ljj = H_j.T @ Rinv @ H_j
        Lij = H_i.T @ Rinv @ H_j
        e = self.z - h
        etai = H_i.T @ Rinv @ e
        etaj = H_j.T @ Rinv @ e
        return {
            self.i: (Lii, etai),
            self.j: (Ljj, etaj),
            (self.i, self.j): Lij,
            (self.j, self.i): Lij.T
        }

# --- 构造因子列表 ---
factors = []
# 位姿先验 (收紧一些数值更稳定)
factors.append(PosePriorFactor('x0', poses_gt[0], np.array([0.5,0.5,0.2])))
# 路标弱先验
for j, lm in enumerate(landmarks_gt):
    factors.append(LandmarkPriorFactor(f'l{j}', lm, np.array([1.0,1.0])))
# 里程计因子
odom_sigma = np.array([0.1,0.1,0.05])
for i in range(num_poses-1):
    delta = poses_gt[i+1] - poses_gt[i]
    factors.append(OdometryFactor(f'x{i}', f'x{i+1}', delta, odom_sigma))
# 方位-距离因子
br_sigma = np.array([0.05,0.3])
for i, p in enumerate(poses_gt):
    for j, lm in enumerate(landmarks_gt):
        dx = lm - p[:2]
        meas = np.array([np.arctan2(dx[1],dx[0]) - p[2], np.linalg.norm(dx)])
        factors.append(BearingRangeFactor(f'x{i}', f'l{j}', meas, br_sigma))

# --- 运行 GBP ---
graph = GBPGraph(factors, variables)
history, stats = graph.run(max_iter=200, tol=1e-5, verbose=True)

print(f"\n收敛：{stats['converged']}，迭代次数：{stats['iterations']}，最终Δ={stats['final_delta']:.2e}")

# --- 可视化结果 ---
est = graph.get_means()
# 真值 vs 估计路标
xs_gt = [lm[0] for lm in landmarks_gt]; ys_gt = [lm[1] for lm in landmarks_gt]
xs_e  = [est[f'l{j}'][0] for j in range(num_landmarks)]
ys_e  = [est[f'l{j}'][1] for j in range(num_landmarks)]
plt.figure(figsize=(6,6))
plt.scatter(xs_gt, ys_gt, marker='x', color='k', label='LM GT')
plt.scatter(xs_e , ys_e , marker='+', color='c', label='LM Est')
# 轨迹
px_gt = [p[0] for p in poses_gt]; py_gt = [p[1] for p in poses_gt]
px_e  = [est[f'x{i}'][0] for i in range(num_poses)]
py_e  = [est[f'x{i}'][1] for i in range(num_poses)]
plt.plot(px_gt, py_gt, 'r-', label='GT traj')
plt.plot(px_e , py_e , 'r--', label='Est traj')
plt.legend(); plt.axis('equal'); plt.grid(True)
plt.title("GBP SLAM: 1 Robot + 10 Landmarks")
plt.show()
