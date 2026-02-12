import numpy as np

def simulate_emg(T=5000, N=16, d=3, noise=0.05, seed=0):
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

# Example
X, Z, A = simulate_emg(T=10000, N=16, d=10, noise=0.1, seed=1)
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
