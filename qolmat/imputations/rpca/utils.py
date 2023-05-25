"""
Modular utility functions for RPCA
"""

import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
from numpy.typing import NDArray
from scipy.linalg import toeplitz
from sklearn.neighbors import kneighbors_graph


def fold_signal(X: NDArray, n_rows: int) -> NDArray:
    """
    Reshape a time series into a 2D-array

    Parameters
    ----------
    X : NDArray
    n_rows : int
        Number of rows of the 2D-array

    Returns
    -------
    Tuple[NDArray, int]
        Array and number of added nan's fill it

    Raises
    ------
    ValueError
        if X is not a 1D array
    """
    if len(X.shape) != 2 or X.shape[0] != 1:
        raise ValueError("'X' should be 2D with a single line")

    if (X.size % n_rows) > 0:
        X = X[0]
        X = np.append(X, [np.nan] * (n_rows - (X.size % n_rows)))
    X = X.reshape(n_rows, -1)

    return X


def approx_rank(
    M: NDArray,
    threshold: float = 0.95,
) -> int:
    """
    Estimate a bound on the rank of an array by SVD.

    Parameters
    ----------
    M : NDArray
    threshold : float, Optional
        fraction of the cumulative sum of the singular values, by default 0.95

    Returns
    -------
    int: Approximated rank of M

    """
    if threshold == 1:
        return min(M.shape)
    _, svd, _ = np.linalg.svd(M, full_matrices=True)
    nuclear = np.sum(svd)
    cum_sum = np.cumsum([sv / nuclear for sv in svd])
    return np.argwhere(cum_sum > threshold)[0][0] + 1


def soft_thresholding(
    X: NDArray,
    threshold: float,
) -> NDArray:
    """
    Shrinkage operator (i.e. soft thresholding) on the elements of X.

    Parameters
    ----------
    X : NDArray
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


def impute_nans(M: NDArray, method: str = "zeros") -> NDArray:
    """
    Impute the M's nan with the specified method

    Parameters
    ----------
    M : NDArray
        Array to impute
    method : str
        'mean', 'median', or 'zeros'

    Returns
    -------
    NDArray
        Imputed Array
    Raises
    ------
        ValueError
            if ``method`` is not
            in 'mean', 'median' or 'zeros']

    """
    if method == "mean":
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = np.where(np.isnan(M), np.resize(np.nanmean(M, axis=0), M.shape), M)
        result = np.where(np.isnan(result), np.nanmean(result), result)
    elif method == "median":
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = np.where(np.isnan(M), np.resize(np.nanmedian(M, axis=0), M.shape), M)
        result = np.where(np.isnan(result), np.nanmedian(result), result)
    elif method == "zeros":
        result = np.where(np.isnan(M), 0, M)
    else:
        raise ValueError("'method' should be 'mean', 'median' or 'zeros'.")
    return result


def l1_norm(M: NDArray) -> float:
    """
    L1 norm of an array

    Parameters
    ----------
    M : NDArray

    Returns
    -------
    float
        L1 norm of M
    """
    return np.sum(np.abs(M))


def toeplitz_matrix(T: int, dimension: int, model: str) -> NDArray:
    """
    Create a matrix Toeplitz matrix H to take into account temporal correlation via HX
    H=Toeplitz(0,1,-1), in which the central diagonal is defined as ones and
    the T upper diagonal is defined as minus ones.

    Parameters
    ----------
    T : int
        diagonal offset
    dimension : int
        second dimension of H = first dimension of X
    model: str
        "column" or "row"

    Returns
    -------
    NDArray
        Toeplitz matrix
    """
    first_row = np.zeros((dimension,))
    first_row[0] = 1
    first_row[T] = -1

    first_col = np.zeros((dimension,))
    first_col[0] = 1

    H = toeplitz(first_col, first_row)
    if model == "row":
        return H[:-T, :]
    elif model == "column":
        return H[:, T:]
    else:
        raise ValueError("Parameter `model`should be 'row' of 'column'")
