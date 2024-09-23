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


def svdtriplet(
    X: NDArray[np.float64],
    row_weights: Optional[NDArray[np.float64]] = None,
    col_weights: Optional[NDArray[np.float64]] = None,
    ncp: int = np.inf,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Perform a weighted Singular Value Decomposition (SVD) of matrix X.

    This function computes the SVD of a weighted matrix, where weights are
    applied to both the rows and columns. Row and column weights are optional,
    and if not provided, uniform weights are applied by default.

    Parameters
    ----------
    X : NDArray[np.float64]
        Input matrix to decompose with SVD.
    row_weights : Optional[NDArray[np.float64]], optional
        Weights for the rows of the matrix, by default None (uniform weights).
    col_weights : Optional[NDArray[np.float64]], optional
        Weights for the columns of the matrix, by default None (uniform
        weights).
    ncp : int, optional
        The number of components to retain, by default np.inf. This will be
        capped at min(rows-1, cols).

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        A tuple containing:
        - Singular values (s)
        - Left singular vectors (U)
        - Right singular vectors (V)

    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("Input matrix X must be 2-dimensional")
    n_rows, n_cols = X.shape
    if row_weights is None:
        row_weights = np.ones(n_rows) / n_rows
    else:
        row_weights = np.asarray(row_weights, dtype=np.float64)
        if row_weights.shape[0] != n_rows:
            raise ValueError("Row weights must match the number of rows in X")

    if col_weights is None:
        col_weights = np.ones(n_cols)
    else:
        col_weights = np.asarray(col_weights, dtype=np.float64)
        if col_weights.shape[0] != n_cols:
            raise ValueError(
                "Column weights must match the number of columns in X"
            )

    row_weights /= row_weights.sum()
    X_weighted = X * np.sqrt(col_weights)  # Column weights
    X_weighted *= np.sqrt(row_weights[:, None])  # Row weights

    ncp = min(ncp, n_rows - 1, n_cols)

    if n_cols <= n_rows:
        U, s, Vt = np.linalg.svd(X_weighted, full_matrices=False)
        V = Vt.T
    else:
        Vt, s, U = np.linalg.svd(X_weighted.T, full_matrices=False)
        V = Vt.T
        U = U.T

    # Truncate U, V, and s to the top ncp components
    U, V, s = U[:, :ncp], V[:, :ncp], s[:ncp]

    sign_correction = np.sign(np.sum(V, axis=0))
    sign_correction[sign_correction == 0] = 1
    U *= sign_correction
    V *= sign_correction
    U /= np.sqrt(row_weights[:, None])
    V /= np.sqrt(col_weights[:, None])

    return s, U, V
