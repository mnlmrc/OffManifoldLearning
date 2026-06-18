"""
Estimate the intrinsic dimensionality of the enslaving manifold (W)
using cross-validated factor analysis log-likelihood, following
Sadtler et al. (2014, Nature), Methods: "Estimation of intrinsic
dimensionality" / Fig. 4a-c.

Input
-----
F : ndarray, shape (n_samples, 15)
    Calibration data: rows are time bins (or trials), columns are the
    15 force/torque channels (5 fingers x 3 axes), z-scored.

Output
------
- Cross-validated log-likelihood as a function of k (number of factors)
- Estimated intrinsic dimensionality (EID) = k at the peak
- The fitted FA model (loading matrix L) at the chosen k
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import KFold


def fa_crossval(F, k_range, n_folds=4, random_state=0):
    """
    Compute cross-validated log-likelihood of factor analysis models
    with different numbers of factors k.

    Parameters
    ----------
    F : ndarray, shape (n_samples, n_channels)
        Z-scored calibration data.
    k_range : iterable of int
        Candidate numbers of factors to evaluate (e.g., range(1, 15)).
    n_folds : int
        Number of cross-validation folds (Sadtler et al. used 4).
    random_state : int
        Random seed for reproducibility of fold splits.

    Returns
    -------
    k_values : ndarray
        The candidate k values tested.
    mean_ll : ndarray
        Mean held-out log-likelihood (per sample) for each k,
        averaged across folds.
    sem_ll : ndarray
        Standard error of the mean across folds, for each k.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    k_values = np.array(list(k_range))
    fold_ll = np.full((n_folds, len(k_values)), np.nan)
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(F)):
        F_train, F_test = F[train_idx], F[test_idx]
        for ki, k in enumerate(k_values):
            fa = FactorAnalysis(n_components=k, random_state=random_state)
            fa.fit(F_train)
            fold_ll[fold_idx, ki] = fa.score(F_test) # Average log-likelihood per test sample

    mean_ll = np.nanmean(fold_ll, axis=0)
    sem_ll = np.nanstd(fold_ll, axis=0, ddof=1) / np.sqrt(n_folds)

    return k_values, mean_ll, sem_ll


def est_dim_crossval(F, k_range=range(1, 15), n_folds=4, random_state=0,):
    """
    Full pipeline: cross-validate FA over k, identify the peak
    (estimated intrinsic dimensionality, EID), and optionally plot.

    Returns
    -------
    eid : int
        The k value at the peak cross-validated log-likelihood.
    k_values, mean_ll, sem_ll : as returned by fa_crossval
    """

    k_values, mean_ll, sem_ll = fa_crossval(F, k_range, n_folds=n_folds, random_state=random_state)

    eid = k_values[np.argmax(mean_ll)]

    # if eid < 2:
    #     print("Warning: Estimated intrinsic dimensionality is < 2. Setting k=2.")
    #     eid = 2
        
    # if plot:
    #     fig, ax = plt.subplots(figsize=(5, 4))
    #     ax.errorbar(k_values, mean_ll, yerr=sem_ll, marker='o', color='k')
    #     ax.axvline(eid, color='r', linestyle='--',
    #                 label=f'EID = {eid}')
    #     ax.set_xlabel('Number of factors (k)')
    #     ax.set_ylabel('Cross-validated log-likelihood\n(per sample)')
    #     ax.set_title('Estimating intrinsic dimensionality of\nthe enslaving manifold')
    #     ax.legend()
    #     fig.tight_layout()
    #     fig.savefig('/mnt/user-data/outputs/fa_dimensionality.png', dpi=150)
    #     plt.close(fig)

    return eid, k_values, mean_ll, sem_ll


def est_intrinsic_manifold(F, k):
    """
    Fit the final factor analysis model at the chosen dimensionality k.

    Returns
    -------
    L : ndarray, shape (n_channels, k)
        Loading matrix. Its column space defines the intrinsic
        (enslaving) manifold W.
    fa : fitted FactorAnalysis object
        Provides fa.components_ (= L.T), fa.noise_variance_ (= diag(Psi)),
        and fa.mean_.
    """
    fa = FactorAnalysis(n_components=k, random_state=0)
    fa.fit(F)
    L = fa.components_.T  # sklearn stores L.T as components_
    return L, fa


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Example usage with simulated data
    # ------------------------------------------------------------------
    rng = np.random.default_rng(0)

    n_channels = 15
    true_k = 5
    n_samples = 2000

    # Simulate data from a k=5 factor model + channel noise,
    # standing in for real calibration recordings
    L_true = rng.normal(size=(n_channels, true_k))
    Z = rng.normal(size=(n_samples, true_k))
    noise = rng.normal(scale=0.5, size=(n_samples, n_channels))
    F = Z @ L_true.T + noise

    # Z-score each channel (as you would with real force data)
    F = (F - F.mean(axis=0)) / F.std(axis=0)

    eid, k_values, mean_ll, sem_ll = est_dim_crossval(
        F, k_range=range(1, 11), n_folds=4
    )
    print(f"Estimated intrinsic dimensionality (EID): {eid}")
    print(f"(Simulated data was generated with true k = {true_k})")

    L, fa = est_intrinsic_manifold(F, k=eid)
    print(f"Loading matrix L shape: {L.shape}")