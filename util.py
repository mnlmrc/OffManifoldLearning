import numpy as np

def simulate_finger_force(A: np.ndarray,
                          B: np.ndarray,
                          enslavement: list | np.ndarray,
                          n_trials: int=100,
                          T: int=100,
                          Nf: int=5,
                          Nd: int=3,
                          noise: float=0.1,
                          seed: int=0):

    if isinstance(enslavement, list):
        enslavement = np.array(enslavement)

    assert A.shape[-1] == B.shape[-1], "Last dimension of A and B must be the same (i.e., number of basis vectors)"
    assert (B.shape[0] == Nf) & (B.shape[1] == Nd), "B must be (Nf, Nd, n_basis_vectors)"
    assert enslavement.size == Nf, "enslavement must be Nf-dimensional"

    cov = np.outer(enslavement, enslavement)
    C = np.eye(Nf) + cov

    rng = np.random.default_rng(seed)

    N = Nf * Nd

    # Create a force profile
    t = np.linspace(0, 1, T)
    profile = np.sin(np.pi * t) ** 2

    F = np.zeros((n_trials, T, N), dtype=float)
    fingers = np.zeros(n_trials, dtype=int)
    dirs = np.zeros((n_trials, 3), dtype=float)
    for tr in range(n_trials):
        f = rng.integers(0, Nf)  # which finger is instructed

        v = np.zeros(Nd, dtype=float)  # which direction
        direct = rng.integers(0, Nd)
        sign = rng.choice([-1, 1])
        v[direct] = float(sign)
        #v = np.array([-1, 0, 0], dtype=float)
        v_norm = v / (np.linalg.norm(v) + 1e-12)

        amp = 1 + .2 * rng.standard_normal()  # movement amplitude

        z_vec = v_norm[0] * B[f, 0] + v_norm[1] * B[f, 1] + v_norm[2] * B[f, 2]  # latent patterns
        Z = (amp * profile)[:, None] * z_vec[None, :]

        X = Z @ A.T  # map ont force channels
        X += noise * rng.standard_normal(X.shape)

        Ft = X.reshape(-1, Nd, Nf)  # add enslavement
        Ft_ens = Ft @ C.T

        F[tr] = Ft_ens.reshape(-1, N)  # store trials & info
        fingers[tr] = f
        dirs[tr] = v

    return F, fingers, dirs
