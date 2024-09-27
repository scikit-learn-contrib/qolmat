"""Utils algebra functions for qolmat package."""

from typing import Optional, Tuple

import numpy as np
import scipy
from numpy.typing import NDArray


def frechet_distance_exact(
    means1: NDArray,
    cov1: NDArray,
    means2: NDArray,
    cov2: NDArray,
) -> float:
    """Compute the Fréchet distance between two dataframes df1 and df2.

    Frechet_distance = || mu_1 - mu_2 ||_2^2
        + Tr(Sigma_1 + Sigma_2 - 2(Sigma_1 . Sigma_2)^(1/2))
    It is normalized, df1 and df2 are first scaled
    by a factor (std(df1) + std(df2)) / 2
    and then centered around (mean(df1) + mean(df2)) / 2
    The result is divided by the number of samples to get
    an homogeneous result.
    Based on: Dowson, D. C., and BV666017 Landau.
        "The Fréchet distance between multivariate normal distributions."
        Journal of multivariate analysis 12.3 (1982): 450-455.

    Parameters
    ----------
    means1 : NDArray
        Means of the first distribution
    cov1 : NDArray
        Covariance matrix of the first distribution
    means2 : NDArray
        Means of the second distribution
    cov2 : NDArray
        Covariance matrix of the second distribution

    Returns
    -------
    float
        Frechet distance

    """
    n = len(means1)
    if (
        (means2.shape != (n,))
        or (cov1.shape != (n, n))
        or (cov2.shape != (n, n))
    ):
        raise ValueError("Inputs have to be of same dimensions.")

    ssdiff = np.sum((means1 - means2) ** 2.0)
    product = np.array(cov1 @ cov2)
    if product.ndim < 2:
        product = product.reshape(-1, 1)
    covmean = scipy.linalg.sqrtm(product)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    frechet_dist = ssdiff + np.trace(cov1 + cov2 - 2.0 * covmean)

    return frechet_dist / n


def kl_divergence_gaussian_exact(
    means1: NDArray, cov1: NDArray, means2: NDArray, cov2: NDArray
) -> float:
    """Compute the exact Kullback-Leibler divergence.

    This is computed between two multivariate normal distributions
    Based on https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    Parameters
    ----------
    means1: NDArray
        Mean of the first distribution
    cov1: NDArray
        Covariance matrx of the first distribution
    means2: NDArray
        Mean of the second distribution
    cov2: NDArray
        Covariance matrx of the second distribution

    Returns
    -------
    float
        Kulback-Leibler divergence

    """
    n_variables = len(means1)
    L1, _ = scipy.linalg.cho_factor(cov1)
    L2, _ = scipy.linalg.cho_factor(cov2)
    M = scipy.linalg.solve(L2, L1)
    y = scipy.linalg.solve(L2, means2 - means1)
    norm_M = (M**2).sum().sum()
    norm_y = (y**2).sum()
    term_diag_L = 2 * np.sum(np.log(np.diagonal(L2) / np.diagonal(L1)))
    div_kl = 0.5 * (norm_M - n_variables + norm_y + term_diag_L)
    return div_kl


def svdtriplet(X, row_w=None, ncp=np.inf):
    """Perform weighted SVD on matrix X with row weights.

    Parameters
    ----------
    X : ndarray
        Data matrix of shape (n_samples, n_features).
    row_w : array-like, optional
        Row weights. If None, uniform weights are assumed. Default is None.
    ncp : int
        Number of principal components to retain. Default is infinity.

    Returns
    -------
    s : ndarray
        Singular values.
    U : ndarray
        Left singular vectors.
    V : ndarray
        Right singular vectors.

    """
    if not isinstance(X, np.ndarray):
        X = np.array(X, dtype=float)
    else:
        X = X.astype(float)
    if row_w is None:
        row_w = np.ones(X.shape[0]) / X.shape[0]
    else:
        row_w = np.array(row_w, dtype=float)
        row_w /= row_w.sum()
    ncp = int(min(ncp, X.shape[0] - 1, X.shape[1]))
    # Apply weights to rows
    X_weighted = X * np.sqrt(row_w[:, None])
    # Perform SVD
    U, s, Vt = np.linalg.svd(X_weighted, full_matrices=False)
    V = Vt.T
    U = U[:, :ncp]
    V = V[:, :ncp]
    s = s[:ncp]
    # Adjust signs to ensure consistency
    mult = np.sign(np.sum(V, axis=0))
    mult[mult == 0] = 1
    U *= mult
    V *= mult
    # Rescale U by the square root of row weights
    U /= np.sqrt(row_w[:, None])
    return s, U, V
