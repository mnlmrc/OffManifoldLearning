import numpy as np
from decoders import vel_decoder
import pandas as pd
import os


def calc_manifold(F: np.ndarray,
                  d: int):
    U, S, Vt = np.linalg.svd(F, full_matrices=False)
    B = Vt[:d].T  # basis vectors for intrinsic-dimensional manifold
    Q, _ = np.linalg.qr(B, mode='complete')
    B_orth = Q[:, B.shape[1]:]

    return B, B_orth


def train_controller(A: np.ndarray,
                     B: np.ndarray,
                     W: np.ndarray,
                     sigma_u: float=.1,
                     n_trials: int=1000,
                     eta: float=.001,
                     ang: float | list | np.ndarray=None,
                     radius: float=1.,
                     maxT: int=1000,
                     dt: float=.01,
                     tol: float=.001,
                     #seed: int=0
                     ):

    rng = np.random.default_rng()

    if ang is None:
        ang = rng.uniform(0, 2 * np.pi, size=n_trials)

    N, K = A.shape
    W_pol = np.zeros((K, 2))  # * 1e-2
    J = W @ B.T @ A #* dt
    success = np.zeros((n_trials,), dtype=bool)
    nsteps = np.zeros(n_trials, dtype=int)
    loss = np.zeros(n_trials, dtype=float)
    meanDev = np.zeros_like(loss)
    velMax = np.zeros_like(loss)
    for tr in range(n_trials):
        pos_star = radius * np.array([np.cos(ang[tr]), np.sin(ang[tr])])
        u = np.zeros((K,), dtype=float)
        pos0 = np.zeros((2,), dtype=float)
        pos = pos0.copy()
        current_loss, t, velMax[tr], dev = np.inf, 0, 0, 0
        for t in range(maxT):
            f = A @ u  # (N,)
            vel = vel_decoder(f, B, W)
            pos = pos + dt * vel

            e = pos_star - pos

            u = W_pol @ e + sigma_u * rng.standard_normal(K)
            g = J.T @ e
            W_pol += eta * np.outer(g, e)

            current_loss = np.sqrt(e[0] ** 2 + e[1] ** 2)

            print(f'doing trial {tr + 1} of {n_trials}, loss = {current_loss}')

            vel_mod = np.linalg.norm(vel)
            if  vel_mod > velMax[tr]:
                velMax[tr] = vel_mod.copy()

            bb = (pos_star @ pos_star) + 1e-12
            alpha = (pos @ pos_star) / bb
            proj = alpha * pos_star
            dev += np.linalg.norm(pos - proj)

            if current_loss < tol * radius:
                success[tr] = True
                break

        steps = t + 1
        nsteps[tr] = steps
        loss[tr] = current_loss
        meanDev[tr] = dev / steps

    return W_pol, success, nsteps, loss, meanDev, velMax


def simulate_trial(A, B, W, W_pol, sigma_u = .001, radius=1., maxT=1000, dt=.01, tol=.05):
    rng = np.random.default_rng()
    N, K = A.shape
    ang = rng.uniform(0, 2 * np.pi)
    pos_star = radius * np.array([np.cos(ang), np.sin(ang)])
    u = np.zeros((K,), dtype=float)
    pos = np.zeros((2,), dtype=float)
    traj = np.full((maxT, 2), np.nan)
    for t in range(maxT):
        f = A @ u  # (N,)
        vel = vel_decoder(f, B, W)
        pos = pos + dt * vel
        traj[t] = pos
        e = pos_star - pos
        u = W_pol @ e + sigma_u * rng.standard_normal(K)
        loss = np.sqrt(e[0] ** 2 + e[1] ** 2)

        if loss < tol * radius:
            break

    return pos_star, traj, u


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    save_dir = 'data/training'
    n_trials = 1200
    ang = rng.choice([0, .25, .5, .75, 1, 1.25, 1.5, 1.75], size=n_trials) * np.pi
    os.makedirs(save_dir, exist_ok=True)
    tinfo = pd.read_csv('data/pretraining/tinfo.tsv', sep='\t')
    N = len(tinfo.subj_id.unique())
    d = 5
    dataset = ['stroke', 'intact']
    mapping = ['on', 'off']
    for ds in dataset:
        for sn in range(N):
            F = np.load(f'data/pretraining/single_finger.pretraining.{ds}.{sn+100}.npy')
            A = np.load(f'data/basis_vectors/basis_vectors.{ds}.{sn+100}.npy')
            Nc = A.shape[0]
            F_c = F.reshape(-1, Nc)
            B_on, B_off = calc_manifold(F_c, d)
            Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
            W = Q[:2]  # define decoder mapping
            np.save(f'data/pretraining/W_dec.{ds}.{sn + 100}.npy', W)
            for map in mapping:
                if map == 'on':
                    B = B_on.copy()
                elif map == 'off':
                    T = rng.standard_normal((B_off.shape[1], d))  # (q-d, d)
                    T /= np.linalg.norm(T, axis=0, keepdims=True) + 1e-12
                    B_mix = B_on + .9 * (B_off @ T)  # (q, d)
                    B_om, _ = np.linalg.qr(B_mix)
                    B = B_om[:, :d]
                W_pol, success, nsteps, loss, meanDev, velMax = train_controller(A, B, W, ang=ang, n_trials=n_trials)
                np.save(f'data/training/W_pol.{map}-manifold.{ds}.{sn+100}.npy', W_pol)

                df = pd.DataFrame(columns=['subj_id', 'group', 'mapping', 'success', 'nsteps', 'loss', 'meanDev',
                                           'velMax'])
                df['success'] = success
                df['nsteps'] = nsteps
                df['loss'] = loss
                df['meanDev'] = meanDev
                df['velMax'] = velMax
                df['subj_id'] = sn + 100
                df['group'] = ds
                df['mapping'] = map
                df['TN'] = np.linspace(1, n_trials, n_trials)
                df.to_csv(f'data/training/log_training.{map}-manifold.{ds}.{sn+100}.tsv', sep='\t',
                          index=False)
