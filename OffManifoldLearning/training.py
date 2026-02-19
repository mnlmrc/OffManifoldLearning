import numpy as np
import pandas as pd
import os
import argparse
from decoders import VelocityDecoder
from util import calc_intrinsic_manifold


class TrainController():

    def __init__(self,
                 A0: np.ndarray,
                 B: np.ndarray,
                 P: np.ndarray,
                 target: np.ndarray,
                 radius: float = -1.,
                 maxT: int = 500,
                 n_trials: int = 1000,
                 dt: float = .01,
                 sigma_u: float = .1,
                 sigma_s: float = .01,
                 eta_w: float = 1e-3,
                 eta_a: float = 1e-4,
                 tol: float = 1e-3,  # success tolerance as % of radius
                 train_W_pol: bool = False,
                 train_A: bool = False,
                 ):

        self.A0 = A0
        self.A = A0.copy()
        self.Nc, self.K = A0.shape
        self.B = B
        self.P = P

        self.dt = dt
        self.sigma_u = sigma_u  # noise on u update
        self.sigma_s = sigma_s  # noise on basis vector selection

        self.eta_w = eta_w  # learning rate for W_pol
        self.eta_a = eta_a

        self.W_pol = np.zeros((self.K, 2))

        self.s = np.ones(self.K)

        self.target = target
        self.radius = radius
        self.tol = tol
        self.pos_star = np.zeros(2)

        self.n_trials = n_trials
        self.maxT = maxT

        self.train_W_pol = train_W_pol
        self.train_A = train_A

        self.baseline = 0
        self.trajectories = []

        self.velMax = np.zeros(n_trials)
        self.success = np.zeros(n_trials)
        self.meanDev = np.zeros(n_trials)
        self.nsteps = np.zeros(n_trials)
        self.loss = np.zeros(n_trials)

    def simulate_training(self):
        for tr in range(self.n_trials):
            current_loss, t, traj = self.simulate_trial(tr)
            self.trajectories.append(traj)
            R = -current_loss  # - time_cost * steps
            self.baseline += (R - self.baseline) / (tr + 1)
            adv = R - self.baseline
            self._update_s(adv)

    def simulate_trial(self, tr):
        self.pos_star = self._calc_pos_star(tr)
        eps_s = rng.standard_normal(self.K)  # exploration direction
        s = self.s + self.sigma_s * eps_s
        u = np.zeros(self.K)
        pos = np.zeros(2)
        traj = np.zeros((self.maxT, 2))
        velMax = 0
        dev = 0
        success = False
        for t in range(self.maxT):
            f = self._simulate_force(s, u)
            pos, vel, u, e = self._update_pos(f, pos)
            traj[t] = pos
            self._update_Wpol(e) if self.train_W_pol is True else None
            current_loss = np.linalg.norm(e)

            if current_loss < self.tol * self.radius:
                success = True
                break

            alpha = (pos @ self.pos_star) / ((self.pos_star @ self.pos_star) + 1e-12)
            proj = alpha * self.target[tr]
            dev += np.linalg.norm(pos - proj)

            vel_mod = np.linalg.norm(vel)
            if vel_mod > velMax:
                velMax = vel_mod

        self.loss[tr] = current_loss
        self.velMax[tr] = velMax
        self.success[tr] = success
        self.meanDev[tr] = dev / t
        self.nsteps[tr] = t

        return current_loss, t, traj

    def _simulate_force(self, s, u):
        self.A = self.A0 * s[None, :]  # (N, K)
        f = self.A @ u  # (N,)
        return f

    def _calc_pos_star(self, tr):
        angle = self.target[tr]
        pos_star = self.radius * np.array([np.cos(angle), np.sin(angle)], dtype=float)
        return pos_star

    def _update_pos(self, f, pos):
        vel = self.P @ f  # (2,)
        pos = pos + self.dt * vel  # (2,)

        # error
        e = self.pos_star - pos  # (2,)

        # --- policy: sample action u around mean mu = W_pol @ e ---
        mu = self.W_pol @ e  # (K,)
        noise = rng.standard_normal(self.K,)
        u = mu + self.sigma_u * noise # (K,)

        return pos, vel, u, e

    def _update_Wpol(self, e):
        J_u = self.dt * (self.P @ self.A)  # (2, K)
        g_u = J_u.T @ e  # (K,)  (descending loss uses +outer(g_u,e) given your convention)
        self.W_pol += self.eta_w * np.outer(g_u, e)  # (K,2)

    def _update_s(self, adv):
        self.s += self.eta_a * adv  #* (eps_s / (sigma_s ** 2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_rehab', action='store_true')
    args = parser.parse_args()

    sim_rehab = args.sim_rehab

    save_dir = '../data/post_rehab' if sim_rehab is True else '../data/controller_training'
    os.makedirs(save_dir, exist_ok=True)

    # init rng
    rng = np.random.default_rng(0)

    # design trials
    n_trials = 1200
    target = rng.choice([0, .25, .5, .75, 1, 1.25, 1.5, 1.75], size=n_trials) * np.pi

    # dataset has 40 patients/group
    N = 2
    dataset = ['stroke', 'intact']
    mapping = ['on', 'off']

    # dimensionality of intrinsic manifold
    d = 5

    # loop thorugh participants
    for ds in dataset:
        for sn in range(N):
            print(f'doing participant {sn + 100}, {ds}...')
            # load single finger force, basis vectors and calculate intrisic manifold
            F = np.load(f'../data/baseline/single_finger.pretraining.{ds}.{sn + 100}.npy')
            A0 = np.load(f'../data/basis_vectors/basis_vectors.{ds}.{sn + 100}.npy')
            Nc = A0.shape[0]
            F_c = F.reshape(-1, Nc)
            B = calc_intrinsic_manifold(F_c, d)

            # if simulating rehabilitation, W_dec should be already saved from the training expeirment
            if sim_rehab:
                W = np.load(f'../data/controller_training/W_dec.{ds}.{sn + 100}.npy')
            else:
                Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
                W = Q[:2]  # define decoder mapping
                np.save(f'{save_dir}/W_dec.{ds}.{sn + 100}.npy', W)

            VD = VelocityDecoder(B, W, angle=np.deg2rad(60))

            # try both mappings
            for map in mapping:
                if map == 'on':
                    P = VD.P0
                elif map == 'off':
                    P = VD.P_om

                TC = TrainController(A0, B, P, target=target, train_A=sim_rehab, train_W_pol=True, n_trials=n_trials)
                TC.simulate_training()
                #A = A0 * TC.s[None, :]

                np.save(f'{save_dir}/W_pol.{map}-manifold.{ds}.{sn + 100}.npy', TC.W_pol)
                if sim_rehab:
                    np.save(f'{save_dir}/basis_vectors.{map}-manifold.{ds}.{sn + 100}.npy', TC.A)

                df = pd.DataFrame(columns=['subj_id', 'group', 'mapping', 'success', 'nsteps', 'loss', 'meanDev',
                                           'velMax'])
                df['success'] = TC.success
                df['nsteps'] = TC.nsteps
                df['loss'] = TC.loss
                df['meanDev'] = TC.meanDev
                df['velMax'] = TC.velMax
                df['subj_id'] = sn + 100
                df['group'] = ds
                df['mapping'] = map
                df['TN'] = np.linspace(1, n_trials, n_trials)
                df.to_csv(f'{save_dir}/log_training.{map}-manifold.{ds}.{sn + 100}.tsv', sep='\t',
                          index=False)
