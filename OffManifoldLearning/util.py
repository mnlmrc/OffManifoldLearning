import numpy as np

def calc_intrinsic_manifold(F: np.ndarray, d: int):
    U, S, Vt = np.linalg.svd(F, full_matrices=False)
    B = Vt[:d].T  # basis vectors for intrinsic-dimensional manifold
    # Q, _ = np.linalg.qr(B, mode='complete')
    # B_orth = Q[:, B.shape[1]:]
    return B


def reshape_off_manifold(B_on, B_off, d, angle=.9):
    rng = np.random.default_rng()
    T = rng.standard_normal((B_off.shape[1], d))  # (q-d, d)
    T /= np.linalg.norm(T, axis=0, keepdims=True) + 1e-12
    B_mix = B_on + angle * (B_off @ T)  # (q, d)
    B_om, _ = np.linalg.qr(B_mix)
    B_reshaped = B_om[:, :d]
    return B_reshaped


