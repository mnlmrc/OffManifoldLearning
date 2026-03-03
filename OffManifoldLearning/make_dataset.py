import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import OffManifoldLearning.globals as gl
import numpy as np
import pandas as pd
import os

def make_basis_vectors(
        Nf: int=5,
        Nd: int=3,
        d: int=10,
        w_f: float=0.9,
        w_d: float=0.1,
        w_b: float=0.1,
        #seed: int=0
    ):

    rng = np.random.default_rng()
    N = Nf * Nd  # N channels
    single_finger = np.eye(N)  # first we have single finger "synergies"
    #add_patterns = rng.standard_normal((N, d))  # then some additional patterns
    base_vec = np.r_[np.zeros(5), np.ones(5), -1 * np.ones(5)]
    add_patterns = np.column_stack([np.random.permutation(base_vec) for _ in range(d)])
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
    #B_b[:, 0] *= -1
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


def make_dataset_baseline():
    rng = np.random.default_rng(seed=0)
    group = ['stroke', 'intact']
    tinfo = {'finger': [], 'dirX': [], 'dirY': [], 'dirZ': [], 'group': [], 'w_f': [], 'w_b': [], 'subj_id': [],
             'TN': []}
    save_dir = os.path.join(gl.baseDir, 'baseline')
    os.makedirs(save_dir, exist_ok=True)
    N = 20
    enslavement = np.array([.1, .1, .1, .4, .4])
    for gr in group:
        for sn in range(N):
            print(f'doing {gr},{sn + 1}/{N}')
            B = make_recruitment(Nf=5, Nd=3, d=20)
            C = make_enslavement(enslavement)
            if gr == 'intact':
                w_f = rng.uniform(.6, 1.)
                w_b = rng.uniform(.05, .35)
            elif gr == 'stroke':
                w_f = rng.uniform(.0, .3)
                w_b = rng.uniform(.6, .9)
            A = make_basis_vectors(Nf=5, Nd=3, d=20, w_f=w_f, w_b=w_b)
            F, finger, direction = make_finger_force(A, B, C)
            np.save(f'{save_dir}/basis_vectors.{gr}.{sn + 100}.npy', A)
            np.save(f'{save_dir}/single_finger.pretraining.{gr}.{sn + 100}.npy', F)
            tinfo['finger'].extend(finger)
            tinfo['dirX'].extend(direction[:, 0])
            tinfo['dirY'].extend(direction[:, 1])
            tinfo['dirZ'].extend(direction[:, 2])
            tinfo['w_f'].extend([w_f] * finger.size)
            tinfo['w_b'].extend([w_b] * finger.size)
            tinfo['subj_id'].extend([sn + 100] * finger.size)
            tinfo['group'].extend([gr] * finger.size)
            tinfo['TN'].extend(np.arange(finger.size) + 1)
    tinfo = pd.DataFrame(tinfo)
    cond_vec = (np.char.mod('%d', tinfo['dirX'].to_numpy()) + ',' +
                np.char.mod('%d', tinfo['dirY'].to_numpy()) + ',' +
                np.char.mod('%d', tinfo['dirZ'].to_numpy()))
    tinfo['cond_vec'] = cond_vec
    pd.DataFrame(tinfo).to_csv(f'{save_dir}/tinfo.tsv', sep='\t', index=False)


def make_dataset_postrehab():
    group = ['stroke', 'intact']
    tinfo = {'finger': [], 'dirX': [], 'dirY': [], 'dirZ': [], 'group': [], 'subj_id': [], 'TN': [], 'angle': []}
    save_dir = os.path.join(gl.baseDir, 'post_rehab')
    N = 20
    enslavement = np.array([.1, .1, .1, .4, .4])
    angle = [0, 30, 50, 70, 90]
    for gr in group:
        for sn in range(N):
            for ang in angle:
                print(f'doing dataset {gr},{sn + 1}/{N}')
                B = make_recruitment(Nf=5, Nd=3, d=20)
                C = make_enslavement(enslavement)
                A = np.load(os.path.join(gl.baseDir, 'post_rehab', f'basis_vectors.{ang}.{gr}.{sn + 100}.npy'))
                F, finger, direction = make_finger_force(A, B, C)
                np.save(f'{save_dir}/single_finger.post_rehab.{ang}.{gr}.{sn + 100}.npy', F)
                tinfo['finger'].extend(finger)
                tinfo['dirX'].extend(direction[:, 0])
                tinfo['dirY'].extend(direction[:, 1])
                tinfo['dirZ'].extend(direction[:, 2])
                tinfo['subj_id'].extend([sn + 100] * finger.size)
                tinfo['group'].extend([gr] * finger.size)
                tinfo['TN'].extend(np.arange(finger.size) + 1)
                tinfo['angle'].extend([ang] * finger.size)
    tinfo = pd.DataFrame(tinfo)
    cond_vec = (np.char.mod('%d', tinfo['dirX'].to_numpy()) + ',' +
                np.char.mod('%d', tinfo['dirY'].to_numpy()) + ',' +
                np.char.mod('%d', tinfo['dirZ'].to_numpy()))
    tinfo['cond_vec'] = cond_vec
    pd.DataFrame(tinfo).to_csv(f'{save_dir}/tinfo.tsv', sep='\t', index=False)


def calc_dist(session, angle=[0, 50, 70, 90]):
    dataset = ['stroke', 'intact']
    tinfo = pd.read_csv(os.path.join(gl.baseDir, session, 'tinfo.tsv'), sep='\t')
    N = len(tinfo.subj_id.unique())
    if session=='baseline':
        euc = np.zeros((2, N, 5, 6, 6))  # (groups, n_subj, n_finger, dir, dir)
        cos = np.zeros_like(euc)
        for d, ds in enumerate(dataset):
            for f, fi in enumerate(tinfo.finger.unique()):
                for s, sn in enumerate(tinfo.subj_id.unique()):
                    tinfo_s = tinfo[(tinfo.subj_id == sn) & (tinfo.group == ds)]
                    X = np.load(os.path.join(gl.baseDir, session, f'single_finger.pretraining.{ds}.{sn}.npy'))
                    X_f = X[tinfo_s.finger == f, 50]  # .mean(axis=1)
                    X_m = X_f.reshape(6, -1, 15).mean(axis=1)
                    G = X_m @ X_m.T
                    diag = np.diag(G)
                    norm = np.sqrt(np.outer(diag, diag))
                    D2 = diag[:, None] + diag[None, :] - 2 * G
                    euc[d, s, f] = np.sqrt(D2)
                    cos[d, s, f] = 1 - G / norm
    if session=='post_rehab':
        euc = np.zeros((2, len(angle), N, 5, 6, 6))  # (groups, n_subj, n_finger, dir, dir)
        cos = np.zeros_like(euc)
        for a, ang in enumerate(angle):
            for d, ds in enumerate(dataset):
                for f, fi in enumerate(tinfo.finger.unique()):
                    for s, sn in enumerate(tinfo.subj_id.unique()):
                        tinfo_s = tinfo[(tinfo.subj_id == sn) & (tinfo.group == ds) & (tinfo.angle == ang)]
                        X = np.load(os.path.join(gl.baseDir, session, f'single_finger.post_rehab.{ang}.{ds}.{sn}.npy'))
                        X_f = X[tinfo_s.finger == f, 50]  # .mean(axis=1)
                        X_m = X_f.reshape(6, -1, 15).mean(axis=1)
                        G = X_m @ X_m.T
                        diag = np.diag(G)
                        norm = np.sqrt(np.outer(diag, diag))
                        D2 = diag[:, None] + diag[None, :] - 2 * G
                        euc[d, a, s, f] = np.sqrt(D2)
                        cos[d, a, s, f] = 1 - G / norm

    np.save(os.path.join(gl.baseDir, session, f'cosine.npy'), cos)
    np.save(os.path.join(gl.baseDir, session, f'euclidean.npy'), euc)

def calc_var_expl(session, angle=[0, 30, 50, 70, 90]):
    dataset = ['stroke', 'intact']
    tinfo = pd.read_csv(os.path.join(gl.baseDir, session, 'tinfo.tsv'), sep='\t')
    N = len(tinfo.subj_id.unique())
    scaler = StandardScaler()
    pca = PCA()
    if session == 'baseline':
        var_expl = np.zeros((2, N, 15))  # (groups, n_subj, n_channels)
        for d, ds in enumerate(dataset):
            for s, sn in enumerate(tinfo.subj_id.unique()):
                X = np.load(os.path.join(gl.baseDir, 'baseline', f'single_finger.pretraining.{ds}.{sn}.npy'))  # (trials, time, channels)
                X_r = X.reshape(-1, X.shape[-1])
                X_norm = scaler.fit_transform(X_r)
                pca.fit(X_norm)
                var_expl[d, s] = pca.explained_variance_ratio_
    if session == 'post_rehab':
        scaler = StandardScaler()
        pca = PCA()
        var_expl = np.zeros((len(angle), N, 15))  # (angles, n_subj, n_channels)
        for a, ang in enumerate(angle):
            for s, sn in enumerate(tinfo.subj_id.unique()):
                X = np.load(os.path.join(gl.baseDir, 'post_rehab',
                                         f'single_finger.post_rehab.{ang}.stroke.{sn}.npy'))  # (trials, time, channels)
                X_r = X.reshape(-1, X.shape[-1])
                X_norm = scaler.fit_transform(X_r)
                pca.fit(X_norm)
                var_expl[a, s] = pca.explained_variance_ratio_

    np.save(os.path.join(gl.baseDir, session, 'var_expl.npy'), var_expl)