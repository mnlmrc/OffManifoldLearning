import numpy as np


class VelocityDecoder:
    def __init__(self,
                 B: np.ndarray,
                 W: np.ndarray,
                 angle: float = 0,
                 tol: float = 0.01,
                 max_iter: int = 10000
                 ):
        Nd = B.shape[1]
        perm_wm = np.random.permutation(Nd)
        self.B = B
        self.W = W
        self.P0 = W @ B.T
        self.angle = angle
        self.max_iter = max_iter
        self.tol = tol
        G_wm = self._perm_matrix(perm_wm)
        self.P_wm = W @ G_wm @ B.T
        self.P_om = self._calc_P_om()

    def _perm_matrix(self, p):
        """
        p is a permutation of np.arange(n).
        Returns n x n permutation matrix
        """
        n = len(p)
        G = np.eye(n)[p]
        return G

    def _decoder_angle(self, P_om):
        num = np.sum(self.P0 * P_om)
        den = np.linalg.norm(self.P0) * np.linalg.norm(P_om)
        cosang = np.clip(num / den, -1.0, 1.0)
        return np.arccos(cosang)  # radians

    def _calc_P_om(self):
        for _ in range(self.max_iter):
            Nc = self.B.shape[0]
            perm_om = np.random.permutation(Nc)
            G_om = self._perm_matrix(perm_om)
            P_om = self.W @ (self.B.T @ G_om)
            ang = self._decoder_angle(P_om)

            if abs(ang - self.angle) < self.tol:
                return P_om

        raise RuntimeError("No permutation found within tolerance.")
