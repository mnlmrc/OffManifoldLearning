import numpy as np
from decoders import vel_decoder_on_manifold


def train_controller(A: np.ndarray,
                     B: np.ndarray,
                     W: np.ndarray,
                     n_trials: int=100,
                     eta: float=.001,
                     radius: float=1.,
                     maxT: int=1000,
                     dt: float=.01,
                     tol: float=.05,
                     seed: int=0):

    rng = np.random.default_rng(seed)

    N, K = A.shape
    W_pol = rng.standard_normal((K, 2))  # * 1e-2
    M = dt * (W @ B.T @ A)
    nsteps = np.zeros(n_trials, dtype=int)
    loss = np.zeros(n_trials, dtype=float)
    for tr in range(n_trials):
        ang = rng.uniform(0, 2 * np.pi)
        pos_star = radius * np.array([np.cos(ang), np.sin(ang)])
        u = np.zeros((K,), dtype=float)
        pos = np.zeros((2,), dtype=float)
        for t in range(maxT):
            f = A @ u  # (N,)
            vel = vel_decoder_on_manifold(f, B, W)
            pos = pos + dt * vel
            e = pos_star - pos
            u = W_pol @ e
            g = M.T @ e
            w_t = 1 / (t + 1)
            W_pol += (w_t * eta) * np.outer(g, e)
            current_loss = np.sqrt(e[0] ** 2 + e[1] ** 2)
            print(f'doing trial {tr + 1} of {n_trials}, loss = {current_loss}')

            if current_loss < tol * radius:
                break

        nsteps[tr] = t
        loss[tr] = current_loss

    return W_pol, nsteps, loss

def simulate_trial(A, B, W, W_pol, radius=1., maxT=1000, dt=.01, tol=.05, seed=0):
    rng = np.random.default_rng(seed)
    N, K = A.shape
    ang = rng.uniform(0, 2 * np.pi)
    pos_star = radius * np.array([np.cos(ang), np.sin(ang)])
    u = np.zeros((K,), dtype=float)
    pos = np.zeros((2,), dtype=float)
    traj = np.full((maxT, 2), np.nan)
    for t in range(maxT):
        f = A @ u  # (N,)
        vel = vel_decoder_on_manifold(f, B, W)
        pos = pos + dt * vel
        traj[t] = pos
        e = pos_star - pos
        u = W_pol @ e
        loss = np.sqrt(e[0] ** 2 + e[1] ** 2)

        if loss < tol * radius:
            break

    return pos_star, traj, u