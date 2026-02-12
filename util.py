import numpy as np

def simulate_force():
    pass


def simulate_motor_output(T=5000, N=16, d=3, noise=0.05, seed=0):
    """
    Returns X with shape (T, N) and intrinsic (noise-free) dimensionality d.
    Model: X = Z @ A.T + noise
      Z: (T, d) latent timecourses
      A: (N, d) mixing matrix
    """
    if d > N:
        raise ValueError("Require d <= N.")
    rng = np.random.default_rng(seed)

    Z = rng.standard_normal((T, d))        # latents (time x dim)
    A = rng.standard_normal((N, d))        # mixing (electrode x dim)

    X = Z @ A.T                            # (T, N), rank <= d (if noise=0)
    X += noise * rng.standard_normal((T, N))
    return X, Z, A

import numpy as np

def simulate_single_finger_trials(
    n_trials=200,
    T=200,
    d=4,                 # intrinsic dimensionality (<15)
    noise=0.03,
    coupling=0.20,       # off-diagonal “enslavement” strength
    seed=0,
):
    """
    Simulate 3D forces for 5 fingers (15 channels) during single-finger trials.

    Output:
      F : (n_trials, T, 5, 3)  forces (fingers x xyz)
      meta : dict with 'finger' (0..4) and 'dir' (3,) per trial
    Key idea:
      Low-dim latent Z(t) -> mixed into 15D -> then a linear finger-coupling matrix
      (linear transforms do NOT increase rank beyond d, ignoring noise).
    """
    rng = np.random.default_rng(seed)

    Nf, Nd = 5, 3
    N = Nf * Nd
    if d > N:
        raise ValueError("Require d <= 15.")

    # Low-rank mixing from latent space to 15 channels
    A = rng.standard_normal((N, d))

    # Finger coupling matrix (5x5): diag=1, off-diagonals=coupling (optionally with small randomness)
    C = np.full((Nf, Nf), coupling, dtype=float)
    np.fill_diagonal(C, 1.0)
    # Optional: small asymmetry / heterogeneity
    C += 0.02 * rng.standard_normal((Nf, Nf))
    np.fill_diagonal(C, 1.0)
    # Keep it stable-ish
    C = np.clip(C, -0.5, 1.5)

    # A simple smooth movement profile (same for all trials; scales change trial-by-trial)
    t = np.linspace(0, 1, T)
    profile = np.sin(np.pi * t) ** 2  # 0 -> peak -> 0

    # Latent “prototypes” per (finger, axis): each is a d-vector
    # This is what ensures all outputs live in a low-d manifold.
    B = rng.standard_normal((Nf, Nd, d))

    F = np.zeros((n_trials, T, Nf, Nd), dtype=float)
    fingers = np.zeros(n_trials, dtype=int)
    dirs = np.zeros((n_trials, 3), dtype=float)

    for tr in range(n_trials):
        f = rng.integers(0, Nf)          # which finger is instructed
        v = rng.standard_normal(3)       # movement direction in 3D
        v /= (np.linalg.norm(v) + 1e-12)

        amp = 1.0 + 0.2 * rng.standard_normal()  # trial amplitude

        # Build a latent timecourse Z(t) in d dims as a weighted sum of the 3 axis prototypes
        # according to the chosen 3D direction.
        z_vec = v[0] * B[f, 0] + v[1] * B[f, 1] + v[2] * B[f, 2]   # (d,)
        Z = (amp * profile)[:, None] * z_vec[None, :]              # (T, d)

        # Mix into 15 channels
        X = Z @ A.T                                                # (T, 15)
        X += noise * rng.standard_normal(X.shape)

        # Reshape to (T, 5, 3)
        Ft = X.reshape(T, Nf, Nd)

        # Enslavement: for each axis, couple finger forces linearly
        # f_out(:,axis) = C @ f_in(:,axis)
        for ax in range(Nd):
            Ft[:, :, ax] = Ft[:, :, ax] @ C.T

        F[tr] = Ft
        fingers[tr] = f
        dirs[tr] = v

    meta = {"finger": fingers, "dir": dirs, "A": A, "B": B, "C": C, "profile": profile}
    return F, meta


if __name__ == "__main__":
    X, Z, A = simulate_motor_output(T=10000, N=16, d=10, noise=0.1, seed=1)
    print(X.shape)

    # center data
    Xc = X - X.mean(axis=0, keepdims=True)

    # SVD-based PCA
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    # variance explained by each PC
    var = S**2
    pve = var / var.sum()

    # cumulative variance
    cum_pve = np.cumsum(pve)

    print("First 16 PCs cumulative variance:")
    print(cum_pve[:16])
