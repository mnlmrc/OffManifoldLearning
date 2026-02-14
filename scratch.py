import numpy as np
from decoders import pos_decoder_on_manifold


def run_training_pos_decoder(
    A_init: np.ndarray,
    B: np.ndarray,
    W_dec: np.ndarray,
    rng: np.random.Generator,
    mu: np.ndarray | None = None,
    n_episodes: int = 2000,
    T: int = 20,
    target_radius: float = 1.0,
    eta_Wpol: float = 1e-2,     # learning rate for policy weights
    eta_A: float = 0.0,         # set >0 for slow "recovery" in A
    flex_col: int = -1,
    flex_decay: float = 0.0,
    u_clip: float = 2.0,
    pos_gain: float = 1.0,
    pos_clip: float | None = None,
):
    """
    Plant: f = A u
    Decoder: pos = pos_decoder_on_manifold(f, B, W_dec, mu, ...)
    Policy: u = Wpol @ e,  e = pos* - pos

    A_init: (N, K)
    B:      (N, d)
    W_dec:  (2, d)
    """
    A = A_init.copy()
    N, K = A.shape

    if mu is None:
        mu = np.zeros((N,), dtype=float)

    Wpol = np.zeros((K, 2))  # policy weights: u = Wpol @ e

    losses = np.zeros((n_episodes,), dtype=float)
    succ = np.zeros((n_episodes,), dtype=bool)

    # Helpful precomputations for analytic gradients
    # pos = pos_gain * W_dec @ (B.T @ (A u - mu))
    # => pos = M @ u + const, where M = pos_gain * W_dec @ B.T @ A
    # For gradients wrt Wpol we need g_u = (d pos / d u)^T e = M^T e
    for ep in range(n_episodes):
        ang = rng.uniform(0, 2 * np.pi)
        pos_star = target_radius * np.array([np.cos(ang), np.sin(ang)])

        u = np.zeros((K,), dtype=float)
        pos = np.zeros((2,), dtype=float)

        for t in range(T):
            f = A @ u  # (N,)

            pos = pos_decoder_on_manifold(
                f_t=f,
                B=B,
                W=W_dec,
                mu=mu,
                pos_gain=pos_gain,
                pos_clip=pos_clip,
            )

            e = pos_star - pos  # (2,)
            losses[ep] += 0.5 * float(e @ e)

            # --- Policy step ---
            u = Wpol @ e
            u = np.clip(u, -u_clip, u_clip)

            # --- Gradient update for Wpol (one-step)
            # L = 0.5 ||e||^2, e = pos* - pos(u)
            # dL/du = -(dpos/du)^T e
            # Here we do: Wpol += eta * outer(du_dWpol, -dL/du)
            #
            # pos(u) = pos_gain * W_dec @ (B.T @ (A u - mu))
            # dpos/du = pos_gain * W_dec @ B.T @ A   => (2,K)
            M = (pos_gain * W_dec) @ (B.T @ A)      # (2,K)
            g_u = M.T @ e                           # (K,)  (this is (dpos/du)^T e)

            # Since u = Wpol @ e, gradient ascent step on reducing loss:
            # Wpol <- Wpol + eta * outer(g_u, e)
            Wpol += eta_Wpol * np.outer(g_u, e)

            # --- Optional slow plasticity of A (recovery) ---
            if eta_A > 0.0:
                # dpos/df = pos_gain * W_dec @ B.T  => (2,N)
                J_f = (pos_gain * W_dec) @ B.T      # (2,N)

                # A update: A += eta_A * outer( (dpos/df)^T e, u )
                # (N,) outer (K,) -> (N,K)
                A += eta_A * np.outer(J_f.T @ e, u)

                if flex_decay > 0.0:
                    A[:, flex_col] *= (1.0 - flex_decay)

                # keep columns bounded
                A /= (np.linalg.norm(A, axis=0, keepdims=True) + 1e-8)

        succ[ep] = (np.linalg.norm(pos_star - pos) < 0.1 * target_radius)

    logs = dict(loss=losses, success=succ)
    return A, Wpol, logs


def evaluate_pos_decoder(
    A: np.ndarray,
    B: np.ndarray,
    W_dec: np.ndarray,
    Wpol: np.ndarray,
    rng: np.random.Generator,
    mu: np.ndarray | None = None,
    n_episodes: int = 500,
    T: int = 20,
    target_radius: float = 1.0,
    u_clip: float = 2.0,
    pos_gain: float = 1.0,
    pos_clip: float | None = None,
):
    N, K = A.shape
    if mu is None:
        mu = np.zeros((N,), dtype=float)

    losses = np.zeros((n_episodes,), dtype=float)
    succ = np.zeros((n_episodes,), dtype=bool)

    for ep in range(n_episodes):
        ang = rng.uniform(0, 2 * np.pi)
        pos_star = target_radius * np.array([np.cos(ang), np.sin(ang)])

        u = np.zeros((K,), dtype=float)
        pos = np.zeros((2,), dtype=float)

        for t in range(T):
            f = A @ u
            pos = pos_decoder_on_manifold(
                f_t=f, B=B, W=W_dec, mu=mu, pos_gain=pos_gain, pos_clip=pos_clip
            )
            e = pos_star - pos
            losses[ep] += 0.5 * float(e @ e)
            u = np.clip(Wpol @ e, -u_clip, u_clip)

        succ[ep] = (np.linalg.norm(pos_star - pos) < 0.1 * target_radius)

    return dict(loss=losses, success=succ)
