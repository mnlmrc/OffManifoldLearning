import numpy as np
import pandas as pd
import os

def make_basis_vectors(
        Nf: int=5,
        Nd: int=3,
        d: int=5,
        w_f: float=0.9,
        w_d: float=0.3,
        w_b: float=0.1,
        seed: int=0
    ):

    rng = np.random.default_rng(seed)
    N = Nf * Nd  # N channels

    single_finger = np.eye(N)  # first we have single finger "synergies"
    add_patterns = rng.standard_normal((N, d))  # then some additional patterns
    flexCh = np.arange(0, N, Nd, dtype=int)  # then the flexor bias
    flexBias = np.zeros(N)
    flexBias[flexCh] = 1
    A = np.c_[w_f * single_finger, w_d * add_patterns, w_b * flexBias]  # basis vectors for health participants

    I = np.eye(N)  # recruitment of basis vectors in each condition
    B_f = I.reshape(Nf, Nd, N)
    B_p = rng.standard_normal((Nf, Nd, d))
    B_b = np.ones((Nf, Nd))
    B = np.c_[B_f, B_p, B_b[:, :, None]]

    enslavement = np.array([.1, .1, .1, .4, .4])  # set enslavement
    cov = np.outer(enslavement, enslavement)
    C = np.eye(Nf) + cov

    return A, B, C


def make_finger_force(A: np.ndarray,
                      B: np.ndarray,
                      C: np.ndarray,
                      n_trials: int=100,
                      T: int=100,
                      Nf: int=5,
                      Nd: int=3,
                      noise: float=0.1,
                      seed: int=0):

    assert A.shape[-1] == B.shape[-1], "Last dimension of A and B must be the same (i.e., number of basis vectors)"
    assert (B.shape[0] == Nf) & (B.shape[1] == Nd), "B must be (Nf, Nd, n_basis_vectors)"
    assert C.shape == (Nf, Nf), "C must be Nf-by-Nf"

    rng = np.random.default_rng(seed)
    N = Nf * Nd

    # Create a force profile
    t = np.linspace(0, 1, T)
    profile = np.sin(np.pi * t) ** 2

    F = np.zeros((n_trials, T, N), dtype=float)
    finger = np.zeros(n_trials, dtype=int)
    direction = np.zeros((n_trials, 3), dtype=float)
    for tr in range(n_trials):
        f = rng.integers(0, Nf)  # which finger is instructed

        v = np.zeros(Nd, dtype=float)  # which direction
        direct = rng.integers(0, Nd)
        sign = rng.choice([-1, 1])
        v[direct] = float(sign)
        v_norm = v / (np.linalg.norm(v) + 1e-12)

        amp = 1 + .2 * rng.standard_normal()  # movement amplitude

        z_vec = v_norm[0] * B[f, 0] + v_norm[1] * B[f, 1] + v_norm[2] * B[f, 2]  # latent patterns
        Z = (amp * profile)[:, None] * z_vec[None, :]

        X = Z @ A.T  # map ont force channels
        X += noise * rng.standard_normal(X.shape)

        Ft = X.reshape(-1, Nd, Nf)  # add enslavement
        Ft_ens = Ft @ C.T

        F[tr] = Ft_ens.reshape(-1, N)  # store trials & info
        finger[tr] = f
        direction[tr] = v

    return F, finger, direction


if __name__ == '__main__':
    rng = np.random.default_rng(seed=0)
    dataset = ['stroke', 'intact']
    tinfo = {'finger': [], 'dirX': [], 'dirY': [], 'dirZ': [], 'group': [], 'w_f': [], 'w_b': [], 'subj_id': []}
    save_dir = 'data/'
    os.makedirs(save_dir, exist_ok=True)
    N = 100
    for ds in dataset:
        for n in range(N):
            print(f'doing dataset {ds},{n}/{N}')
            if ds == 'intact':
                w_f = rng.uniform(.8, 1.)
                w_b = rng.uniform(.1, .3)
            elif ds == 'stroke':
                w_f = rng.uniform(.1, .3)
                w_b = rng.uniform(.5, .8)
            A, B, C = make_basis_vectors(Nf=5, Nd=3, d=3, w_f=w_f, w_d=w_b)
            F, finger, direction = make_finger_force(A, B, C)
            tinfo['finger'].extend(finger)
            tinfo['dirX'].extend(direction[:, 0])
            tinfo['dirY'].extend(direction[:, 1])
            tinfo['dirZ'].extend(direction[:, 2])
            tinfo['w_f'].extend([w_f] * finger.size)
            tinfo['w_b'].extend([w_b] * finger.size)
            tinfo['subj_id'].extend([n+100] * finger.size)
            tinfo['group'].extend([ds] * finger.size)
            np.save(f'{save_dir}/basis_vectors/basis_vectors.{n + 100}.npy', A)
            np.save(f'{save_dir}/pretraining/single_finger.pretraining.{n + 100}.npy', A)
    tinfo = pd.DataFrame(tinfo)
    pd.DataFrame(tinfo).to_csv(f'{save_dir}/pretraining/tinfo.tsv', sep='\t',index=False)