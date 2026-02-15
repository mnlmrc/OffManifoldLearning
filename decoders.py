import numpy as np

# ----------------------------
# 1) On-manifold position decoder
# ----------------------------
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



