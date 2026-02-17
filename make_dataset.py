import argparse

import numpy as np
import pandas as pd
import os

def make_basis_vectors(
        Nf: int=5,
        Nd: int=3,
        d: int=5,
        w_f: float=0.9,
        w_d: float=0.1,
        w_b: float=0.1,
        #seed: int=0
    ):

    rng = np.random.default_rng()
    N = Nf * Nd  # N channels

    single_finger = np.eye(N)  # first we have single finger "synergies"
    add_patterns = rng.standard_normal((N, d))  # then some additional patterns
    flexCh = np.arange(0, N, Nd, dtype=int)  # then the flexor bias
    flexBias = np.zeros(N)
    flexBias[flexCh] = 1
    A = np.c_[w_f * single_finger, w_d * add_patterns, w_b * flexBias]  # basis vectors for health participants

    return A

def make_recruitment(Nf, Nd, d):
    rng = np.random.default_rng()
    N = Nf * Nd  # N channels

    I = np.eye(N)  # recruitment of basis vectors in each condition
    B_f = I.reshape(Nf, Nd, N)
    B_p = rng.standard_normal((Nf, Nd, d))
    B_b = np.ones((Nf, Nd))
    B = np.c_[B_f, B_p, B_b[:, :, None]]

    return B


def make_enslavement(enslavement):
    Nf = len(enslavement)
    cov = np.outer(enslavement, enslavement)
    C = np.eye(Nf) + cov
    return C


def make_finger_force(A: np.ndarray,
                      B: np.ndarray,
                      C: np.ndarray,
                      n_trials: int=120,
                      T: int=100,
                      Nf: int=5,
                      Nd: int=3,
                      noise: float=0.1):

    assert A.shape[-1] == B.shape[-1], "Last dimension of A and B must be the same (i.e., number of basis vectors)"
    assert (B.shape[0] == Nf) & (B.shape[1] == Nd), "B must be (Nf, Nd, n_basis_vectors)"
    assert C.shape == (Nf, Nf), "C must be Nf-by-Nf"

    rng = np.random.default_rng()
    N = Nf * Nd

    # Create a force profile
    t = np.linspace(0, 1, T)
    profile = np.sin(np.pi * t) ** 2

    # Make trials
    conds = np.array([(f, d, s) for f in range(Nf) for d in range(Nd) for s in (-1, 1)])
    n_rep = int(np.ceil(n_trials / len(conds)))
    conds = np.tile(conds, (n_rep, 1))

    F = np.zeros((n_trials, T, N), dtype=float)
    finger = np.zeros(n_trials, dtype=int)
    direction = np.zeros((n_trials, 3), dtype=float)
    for tr in range(n_trials):

        f, direct, sign = conds[tr]

        finger[tr] = f

        v = np.zeros(Nd, dtype=float)
        v[direct] = float(sign)
        v_norm = v / (np.linalg.norm(v) + 1e-12)

        direction[tr] = v_norm

        amp = 1 + .2 * rng.standard_normal()  # movement amplitude

        z_vec = v_norm[0] * B[f, 0] + v_norm[1] * B[f, 1] + v_norm[2] * B[f, 2]  # latent patterns
        Z = (amp * profile)[:, None] * z_vec[None, :]

        X = Z @ A.T  # map ont force channels
        X += noise * rng.standard_normal(X.shape)

        Ft = X.reshape(-1, Nd, Nf)  # add enslavement
        Ft_ens = Ft @ C.T

        F[tr] = Ft_ens.reshape(-1, N)  # store trials & info

    return F, finger, direction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--post_rehab', action='store_true')
    args = parser.parse_args()

    post_rehab = args.post_rehab

    rng = np.random.default_rng(seed=0)
    dataset = ['stroke', 'intact']
    tinfo = {'finger': [], 'dirX': [], 'dirY': [], 'dirZ': [], 'group': [], 'w_f': [], 'w_b': [], 'subj_id': [],
             'mapping': [] if post_rehab else None
             }
    save_dir = 'data/'
    os.makedirs(save_dir, exist_ok=True)
    N = 40
    enslavement = np.array([.1, .1, .1, .4, .4])
    for ds in dataset:
        for sn in range(N):
            print(f'doing dataset {ds},{sn + 1}/{N}')
            B = make_recruitment(Nf=5, Nd=3, d=5)
            C = make_enslavement(enslavement)
            if post_rehab:
                A_on = np.load(f'../data/post_rehab/basis_vectors.on-manifold.{ds}.{sn + 100}.npy')
                A_off = np.load(f'../data/post_rehab/basis_vectors.off-manifold.{ds}.{sn + 100}.npy')
                F_on, finger_on, direction_on = make_finger_force(A_on, B, C)
                F_off, finger_off, direction_off = make_finger_force(A_off, B, C)
                F = np.vstack((F_on, F_off))
                finger = np.vstack((finger_on, finger_off))
                direction = np.vstack((direction_on, direction_off))
                np.save(f'{save_dir}/single_finger.post_rehab.on-manifold.{ds}.{sn + 100}.npy', F_on)
                np.save(f'{save_dir}/single_finger.post_rehab.off-manifold.{ds}.{sn + 100}.npy', F_off)
                w_b, w_f = None, None
            else:
                if ds == 'intact':
                    w_f = rng.uniform(.6, 1.)
                    w_b = rng.uniform(.05, .35)
                elif ds == 'stroke':
                    w_f = rng.uniform(.0, .3)
                    w_b = rng.uniform(.6, .9)
                else:
                    w_b, w_f = None, None
                A = make_basis_vectors(Nf=5, Nd=3, d=5, w_f=w_f, w_b=w_b)
                F, finger, direction = make_finger_force(A, B, C)
                np.save(f'{save_dir}/basis_vectors/basis_vectors.{ds}.{sn + 100}.npy', A)
                np.save(f'{save_dir}/baseline/single_finger.pretraining.{ds}.{sn + 100}.npy', F)
            tinfo['finger'].extend(finger)
            tinfo['dirX'].extend(direction[:, 0])
            tinfo['dirY'].extend(direction[:, 1])
            tinfo['dirZ'].extend(direction[:, 2])
            tinfo['w_f'].extend([w_f] * finger.size)
            tinfo['w_b'].extend([w_b] * finger.size)
            tinfo['subj_id'].extend([sn+100] * finger.size)
            tinfo['group'].extend([ds] * finger.size)
            if post_rehab:
                tinfo['mapping'].extend(['on'] * F_on.size)
                tinfo['mapping'].extend(['off'] * F_off.size)
    tinfo = pd.DataFrame(tinfo)
    cond_vec = (np.char.mod('%d', tinfo['dirX'].to_numpy()) + ',' +
                np.char.mod('%d', tinfo['dirY'].to_numpy()) + ',' +
                np.char.mod('%d', tinfo['dirZ'].to_numpy()))
    tinfo['cond_vec'] = cond_vec
    if post_rehab:
        pd.DataFrame(tinfo).to_csv(f'{save_dir}/post_rehab/tinfo.tsv', sep='\t', index=False)
    else:
        pd.DataFrame(tinfo).to_csv(f'{save_dir}/baseline/tinfo.tsv', sep='\t',index=False)