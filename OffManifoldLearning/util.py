import numpy as np

def calc_intrinsic_manifold(F: np.ndarray, d: int):
    U, S, Vt = np.linalg.svd(F, full_matrices=False)
    B = Vt[:d].T  # basis vectors for intrinsic-dimensional manifold
    # Q, _ = np.linalg.qr(B, mode='complete')
    # B_orth = Q[:, B.shape[1]:]
    return B


def reshape_off_manifold(B_on, B_off, d, angle=1):
    rng = np.random.default_rng()
    T = rng.standard_normal((B_off.shape[1], d))  # (q-d, d)
    T /= np.linalg.norm(T, axis=0, keepdims=True) + 1e-12
    B_mix = B_on + angle * (B_off @ T)  # (q, d)
    B_om, _ = np.linalg.qr(B_mix)
    B_reshaped = B_om[:, :d]
    return B_reshaped


def curvature_2d(traj: np.ndarray, dt: float, eps: float = 1e-8, eps_pos: float = 1e-6):  # "almost no movement" per sample
    x, y = traj[:, 0], traj[:, 1]

    dx  = np.gradient(x, dt)
    dy  = np.gradient(y, dt)
    ddx = np.gradient(dx, dt)
    ddy = np.gradient(dy, dt)

    num = np.abs(dx * ddy - dy * ddx)
    den = (dx*dx + dy*dy)**1.5 + eps
    kappa = num / den

    kappa_mean = kappa.mean()
    kappa_max  = kappa.max()

    return kappa, kappa_mean, kappa_max


def calc_jerk(traj, dt):
    """
    traj : array (T, 2)  -> columns are x and y
    dt   : time step
    returns:
        jerk      : (T, 2) jerk vector
        jerk_mag  : (T,)   jerk magnitude
    """
    mask = ~np.isnan(traj[:, 0])

    x = traj[mask, 0]
    y = traj[mask, 1]

    dx  = np.gradient(x, dt)
    dy  = np.gradient(y, dt)

    ddx = np.gradient(dx, dt)
    ddy = np.gradient(dy, dt)

    dddx = np.gradient(ddx, dt)
    dddy = np.gradient(ddy, dt)

    return np.sum(dddx**2 + dddy**2) * dt

