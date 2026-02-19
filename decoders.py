import numpy as np

def pos_decoder(f_t,
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


def vel_decoder(f_t,
                B,
                W,
                mu: np.ndarray = None,
                vel_gain: float = 1.0,
                vel_clip: float = None):
    if mu is None:
        mu = np.zeros_like(f_t)

    z = B.T @ (f_t - mu)              # (d,)
    v = vel_gain * (W @ z)            # (2,)
    if vel_clip is not None:
        v = np.clip(v, -vel_clip, vel_clip)
    return v


def find_permutation_for_angle(P_base, make_Pprime, target_angle, tol=0.02, max_iter=10000):

    for _ in range(max_iter):
        Pp = make_Pprime()
        ang = decoder_angle(P_base, Pp)

        if abs(ang - target_angle) < tol:
            return Pp, ang

    raise RuntimeError("No permutation found within tolerance.")


def decoder_angle(P, Pp):
    num = np.sum(P * Pp)
    den = np.linalg.norm(P) * np.linalg.norm(Pp)
    cosang = np.clip(num / den, -1.0, 1.0)
    return np.arccos(cosang)   # radians

class VelocityDecoder:
    def __init__(self,
                 f_t: np.ndarray,
                 B: np.ndarray,
                 W: np.ndarray,
    ):

        P = W @ B.T






