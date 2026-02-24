import numpy as np
import pandas as pd
import os
import argparse
from OffManifoldLearning.decoders import VelocityDecoder
from OffManifoldLearning.util import calc_intrinsic_manifold, curvature_2d, calc_jerk
from joblib import delayed, parallel_backend, Parallel


class TrainController():

    def __init__(self,
                 A0: np.ndarray,
                 B: np.ndarray,
                 P: np.ndarray,
                 target: np.ndarray,
                 radius: float = 1.,
                 maxT: int = 1000,
                 n_trials: int = 1000,
                 dt: float = .01,
                 sigma_u: float = .1,
                 sigma_s: float = .01,
                 lam_u: float = .001,
                 lam_j: float = .001,
                 eta_w: float = 1e-3,
                 eta_a: float = 1e-4,
                 tol: float = .001,  # success tolerance as % of radius
                 train_W_pol: bool = False,
                 train_A: bool = False,
                 ):

        rng = np.random.default_rng()

        self.A0 = A0
        self.A = A0.copy()
        self.Nc, self.K = A0.shape
        self.B = B
        self.P = P

        self.dt = dt
        self.sigma_u = sigma_u  # noise on u update
        self.sigma_s = sigma_s  # noise on basis vector selection
        self.eps_s = rng.standard_normal(self.K)

        self.lam_u = lam_u
        self.lam_j = lam_j

        self.eta_w = eta_w  # learning rate for W_pol
        self.eta_a = eta_a

        self.W_pol = np.zeros((self.K, 2))

        self.s = np.ones(self.K)
        self.exp_noise = np.zeros(self.K)  # init exploratory noise

        self.target = target
        self.radius = radius
        self.tol = tol
        self.pos_star = np.zeros(2)

        self.n_trials = n_trials
        self.maxT = maxT

        self.mean_dist = 0.0
        self.mean_steps = 0.0
        self.mean_jerk = 0.0

        self.train_W_pol = train_W_pol
        self.train_A = train_A

        self.baseline = 0
        self.baseline_prev = 0
        self.trajectories = []
        self.distance = []

        self.velMax = np.zeros(n_trials)
        self.success = np.zeros(n_trials)
        self.meanDev = np.zeros(n_trials)
        # self.curv_mean = np.zeros(n_trials)
        # self.curv_max = np.zeros(n_trials)
        self.nsteps = np.zeros(n_trials)
        self.jerk = np.zeros(n_trials)
        self.dist_to_target = np.zeros(n_trials)

    def simulate_training(self):
        for tr in range(self.n_trials):
            dist, traj = self.simulate_trial(tr)
            mask = ~np.isnan(traj[:, 0])
            print(f'trial {tr + 1}, dist_to_target={self.dist_to_target[tr]}')
            self.jerk[tr] = calc_jerk(traj, self.dt)
            self.trajectories.append(traj)
            self.distance.append(dist)
            self._update_s(tr) if self.train_A else None

    def simulate_trial(self, tr):
        rng = np.random.default_rng()
        self.pos_star = self._calc_pos_star(tr)
        self.eps_s = rng.standard_normal(self.K)  # exploration direction
        self.exp_noise = self.sigma_s * self.eps_s
        s = self.s + self.exp_noise
        u = np.zeros(self.K)
        pos = np.zeros(2)
        traj = np.full((self.maxT, 2), np.nan)
        dist = np.full(self.maxT, np.nan)
        velMax = 0
        dev = 0
        success = False
        for t in range(self.maxT):
            f = self._simulate_force(s, u)
            pos, vel, u, e = self._update_pos(f, pos)
            traj[t] = pos  # record position
            self.dist_to_target[tr] = np.linalg.norm(e)
            dist[t] = self.dist_to_target[tr].copy()

            # updated deviation from straight line
            alpha = (pos @ self.pos_star) / ((self.pos_star @ self.pos_star) + 1e-12)
            proj = alpha * self.pos_star
            dev += np.linalg.norm(pos - proj)

            # update max velocity
            vel_mod = np.linalg.norm(vel)
            if vel_mod > velMax:
                velMax = vel_mod

            # update feedback policy
            self._update_Wpol_gd(e) if self.train_W_pol is True else None

            if self.dist_to_target[tr] < self.tol * self.radius:
                success = True
                break

        self.velMax[tr] = velMax
        self.success[tr] = success
        self.meanDev[tr] = dev / (t + 1)
        self.nsteps[tr] = t + 1

        #_, self.curv_mean[tr], self.curv_max[tr] = curvature_2d(traj, self.dt)

        return dist, traj

    def _simulate_force(self, s, u):
        self.A = self.A0 * s[None, :]  # (N, K)
        f = self.A @ u  # (N,)
        return f

    def _calc_pos_star(self, tr):
        angle = self.target[tr]
        pos_star = self.radius * np.array([np.cos(angle), np.sin(angle)], dtype=float)
        return pos_star

    def _update_pos(self, f, pos):
        rng = np.random.default_rng()
        vel = self.P @ f  # (2,)
        pos = pos + self.dt * vel  # (2,)

        # error
        e = self.pos_star - pos  # (2,)

        # --- policy: sample action u around mean mu = W_pol @ e ---
        mu = self.W_pol @ e  # (K,)
        noise = rng.standard_normal(self.K,)
        u = mu + self.sigma_u * noise # (K,)

        return pos, vel, u, e

    def _update_Wpol_gd(self, e):

        # feedback error
        J_u = (self.P @ self.A)  # (2, K)
        g_u = J_u.T @ e  # (K,)

        # effort
        #g_u -= self.lam_u * u

        # # jerk of command (3rd finite difference)
        # j_u = u - 3 * u_prev + 3 * u_prev2 - u_prev3
        # g_u -= self.lam_j * j_u  # (K,)

        self.W_pol += self.eta_w * np.outer(g_u, e)  # (K,2)

    def _update_s(self, tr):
        dist = float(self.dist_to_target[tr])
        steps = float(self.nsteps[tr])
        jerk = float(self.jerk[tr])

        cost_dist = 1 - (self.mean_dist / (self.mean_dist + dist + 1e-8))
        cost_steps = 1 - (self.mean_steps/ (self.mean_steps + steps + 1e-8))
        cost_jerk = 1 - (self.mean_jerk / (self.mean_jerk + jerk + 1e-8))

        R = -(cost_dist + cost_steps + cost_jerk)

        self.mean_dist += (dist - self.mean_dist) / (tr + 1)
        self.mean_steps += (steps - self.mean_steps) / (tr + 1)
        self.mean_jerk += (jerk - self.mean_jerk) / (tr + 1)

        adv = R - self.baseline
        self.baseline += (R - self.baseline) / (tr + 1)
        self.s += self.eta_a * adv * (self.exp_noise / (self.sigma_s ** 2))
        pass

def train_participant(group, sn, sim_rehab=False):
    rng = np.random.default_rng()

    n_trials = 1200

    d = 5

    save_dir = 'data/post_rehab' if sim_rehab is True else 'data/controller_training'
    os.makedirs(save_dir, exist_ok=True)

    # load single finger force, basis vectors and calculate intrisic manifold
    F = np.load(f'data/baseline/single_finger.pretraining.{group}.{sn + 100}.npy')
    A0 = np.load(f'data/basis_vectors/basis_vectors.{group}.{sn + 100}.npy')
    Nc = A0.shape[0]
    F_c = F.reshape(-1, Nc)
    B = calc_intrinsic_manifold(F_c, d)

    # if simulating rehabilitation, W_dec should be already saved from the training expeirment
    if sim_rehab:
        W = np.load(f'data/controller_training/W_dec.{group}.{sn + 100}.npy')
    else:
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        W = Q[:2]  # define decoder mapping
        np.save(f'{save_dir}/W_dec.{group}.{sn + 100}.npy', W)

    # try different angles with respect to intrinsic manifold
    for angle in [0, 20, 40, 60, 80]:
        print(f'doing participant {sn + 100}, {group}, angle {angle}...')

        VD = VelocityDecoder(B, W, angle=np.deg2rad(angle))
        angle_real = VD.angle_real
        P = VD.P_om

        #target = rng.choice([0, .25, .5, .75, 1, 1.25, 1.5, 1.75], size=n_trials) * np.pi
        target = rng.uniform(0, 2, size=1200) * np.pi

        TC = TrainController(A0, B, P, target=target, train_A=sim_rehab, train_W_pol=True, n_trials=n_trials, maxT=1000)
        #TC.simulate_trial(0)
        TC.simulate_training()
        trajectories = np.array(TC.trajectories)
        distance = np.array(TC.distance)

        np.save(f'{save_dir}/trajectories.{angle}.{group}.{sn + 100}.npy', trajectories)
        np.save(f'{save_dir}/distance.{angle}.{group}.{sn + 100}.npy', distance)
        np.save(f'{save_dir}/W_pol.{angle}.{group}.{sn + 100}.npy', TC.W_pol)
        if sim_rehab:
            np.save(f'{save_dir}/basis_vectors.{angle}.{group}.{sn + 100}.npy', TC.A)

        df = pd.DataFrame()
        df['success'] = TC.success
        df['nsteps'] = TC.nsteps
        df['dist_to_target'] = TC.dist_to_target
        df['meanDev'] = TC.meanDev
        df['velMax'] = TC.velMax
        #df['curvMax'] = TC.curv_max
        #df['curvMean'] = TC.curv_mean
        df['subj_id'] = sn + 100
        df['group'] = group
        df['angle'] = angle_real
        df['target'] = target
        df['TN'] = np.linspace(1, n_trials, n_trials)
        df.to_csv(f'{save_dir}/log_training.{angle}.{group}.{sn + 100}.tsv', sep='\t', index=False)


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sim_rehab",
        action=argparse.BooleanOptionalAction
    )
    args = parser.parse_args(args)

    sim_rehab = args.sim_rehab

    # dataset has 40 patients/group
    N = 20
    group = ['stroke', 'intact']

    train_participant('stroke', 4, True)
    # loop thorugh participants
    for gr in group:
        with parallel_backend("loky"):  # use threading for debug in PyCharm, for run use loky
            Parallel(n_jobs=4)(delayed(train_participant)(gr, sn, sim_rehab) for sn in range(N))
