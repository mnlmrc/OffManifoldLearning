import numpy as np

# ----------------------------
# 1) On-manifold position decoder
# ----------------------------
def pos_decoder_on_manifold(f_t,
                           B,
                           W,
                           mu: np.ndarray=None,
                           pos_gain: float=1.0,
                           pos_clip: float=None):

    if mu is None:
        mu = np.zeros_like(f_t)

    z = B.T @ (f_t - mu)            # (d,)
    pos = pos_gain * (W @ z)    # (2,)
    if pos_clip is not None:
        pos = np.clip(pos, -pos_clip, pos_clip)
    return pos


import numpy as np


# ----------------------------
# 2) Create a simple "intuitive" 2D readout within the manifold
# ----------------------------
def make_W_dec(d, rng):
    # Sample a random matrix in manifold space (d x d)
    R = rng.standard_normal((d, d))

    # Orthonormalize it (QR); Q has orthonormal columns
    Q, _ = np.linalg.qr(R)

    # Take first 2 rows: this makes W_dec map (d,) -> (2,)
    W_dec = Q[:2, :]                      # (2, d)

    # Return the 2D readout matrix
    return W_dec


# ----------------------------
# 3) Training loop (learn a controller W_pol for reaching in 2D)
# ----------------------------
def train_reacher_on_manifold(
    A,                 # (N, K) plant / synergy matrix (fixed baseline)
    Bb,                # (N, d) manifold basis from SVD/PCA
    rng,               # np.random.Generator
    mu=None,           # (N,) mean force used for centering
    n_episodes=2000,   # number of reaches (trials)
    T=20,              # time steps per reach
    target_radius=1.0, # radius of 2D targets
    eta=1e-2,          # learning rate for the controller
    u_clip=2.0,        # clip synergy commands
    pos_gain=1.0,      # optional gain in decoder
    pos_clip=None,     # optional bound on cursor
):
    # Dimensions
    N, K = A.shape                         # N force channels, K synergies
    d = Bb.shape[1]                        # manifold dimensionality

    # Mean force for centering (must match how you built Bb)
    if mu is None:
        mu = np.zeros((N,), dtype=float)

    # Build a fixed "intuitive" decoder inside the manifold
    W_dec = make_W_dec(d, rng)             # (2, d)

    # Initialize policy/controller weights:
    # maps cursor error e (2,) -> synergy commands u (K,)
    W_pol = np.zeros((K, 2), dtype=float)  # (K, 2)

    # Logs (for plotting learning curves later)
    loss_hist = np.zeros(n_episodes, dtype=float)
    succ_hist = np.zeros(n_episodes, dtype=bool)

    # Precompute decoder Jacobian wrt force:
    # pos = pos_gain * W_dec @ (Bb.T @ (f - mu))
    # so dpos/df = pos_gain * W_dec @ Bb.T
    J_f = (pos_gain * W_dec) @ Bb.T        # (2, N)

    # Training over many reaches
    for ep in range(n_episodes):
        # Sample a random target on a circle
        ang = rng.uniform(0, 2*np.pi)                         # random angle
        pos_star = target_radius * np.array([np.cos(ang), np.sin(ang)])     # target x, y

        # Initialize synergy command and cursor (start at origin)
        u = np.zeros((K,), dtype=float)                        # (K,)
        pos = np.zeros((2,), dtype=float)                      # (2,)

        # Rollout for T steps
        for t in range(T):
            # Plant: synergies -> forces
            f = A @ u                                          # (N,)

            # Decoder: forces -> cursor position (2D)
            pos = pos_decoder_on_manifold(
                f_t=f,
                Bb=Bb,
                W_dec=W_dec,
                mu=mu,
                pos_gain=pos_gain,
                pos_clip=pos_clip
            )                                                   # (2,)

            # Cursor error (what you want to reduce)
            e = pos_star - pos                                  # (2,)

            # Accumulate squared-error loss for this episode
            loss_hist[ep] += 0.5 * float(e @ e)

            # Policy: error -> synergy command
            u = W_pol @ e                                       # (K,)

            # Clip synergy command so it stays bounded
            u = np.clip(u, -u_clip, u_clip)

            g = A.T @ (J_f.T @ e)                              # (K,)

            W_pol += eta * np.outer(g, e)                       # (K,2)

        # Define success: did you end close enough to the target?
        succ_hist[ep] = (np.linalg.norm(pos_star - pos) < 0.1 * target_radius)

    # Return the trained controller, the decoder used, and logs
    return W_pol, W_dec, dict(loss=loss_hist, success=succ_hist)
