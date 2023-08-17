"""
Modular utility functions for RPCA
"""


import numpy as np
from numpy.typing import NDArray
import scipy
from scipy.linalg import toeplitz
from scipy import sparse as sps


def approx_rank(
    M: NDArray,
    threshold: float = 0.95,
) -> int:
    """
    Estimate a bound on the rank of an array by SVD.

    Parameters
    ----------
    M : NDArray
        Matrix which rank should be estimated
    threshold : float, Optional
        fraction of the cumulative sum of the singular values, by default 0.95

    Returns
    -------
    int: Approximated rank of M

    """
    if threshold == 1:
        return min(M.shape)
    _, values_singular, _ = np.linalg.svd(M, full_matrices=True)

    cum_sum = np.cumsum(values_singular) / np.sum(values_singular)
    rank = np.argwhere(cum_sum > threshold)[0][0] + 1

    return rank


def soft_thresholding(
    X: NDArray,
    threshold: float,
) -> NDArray:
    """
    Shrinkage operator (i.e. soft thresholding) on the elements of X.

    Parameters
    ----------
    X : NDArray
        Matrix which elements should be shrinked
    threshold : float
        Shrinking factor

    Returns
    -------
    NDArray
        Array V such that V = sign(X) * max(abs(X - threshold,0)
    """
    return np.sign(X) * np.maximum(np.abs(X) - threshold, 0)


def svd_thresholding(X: NDArray, threshold: float) -> NDArray:
    """
    Apply the shrinkage operator to the singular values obtained from the SVD of X.

    Parameters
    ----------
    X : NDArray
        Matrix which singular values should be shrinked
    threshold : float
        Shrinking factor

    Returns
    -------
    NDArray
        Array obtained by computing U * shrink(s) * V where
            U is the array of left singular vectors of X
            V is the array of the right singular vectors of X
            s is the array of the singular values as a diagonal array
    """

    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    s = soft_thresholding(s, threshold)
    return U @ (np.diag(s) @ Vh)


def l1_norm(M: NDArray) -> float:
    """
    L1 norm of an array

    Parameters
    ----------
    M : NDArray
        Matrix which norm should be computed

    Returns
    -------
    float
        L1 norm of M
    """
    return np.sum(np.abs(M))


def toeplitz_matrix(T: int, dimension: int) -> NDArray:
    """
    Create a sparse Toeplitz square matrix H to take into account temporal correlations in the RPCA
    H=Toeplitz(0,1,-1), in which the central diagonal is defined as ones and
    the T upper diagonal is defined as minus ones.

    Parameters
    ----------
    T : int
        diagonal offset
    dimension : int
        dimension of the matrix to create

    Returns
    -------
    NDArray
        Sparse Toeplitz matrix using scipy format
    """

    n_lags = dimension - T
    diagonals = [np.ones(n_lags), -np.ones(n_lags)]
    H_top = sps.diags(diagonals, offsets=[0, T], shape=(n_lags, dimension), format="csr")
    H = sps.dok_matrix((dimension, dimension))
    H[:n_lags] = H_top
    return H
