from __future__ import annotations
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


def proximal_operator(U: np.ndarray, X: np.ndarray, threshold: float) -> np.ndarray:
    """Compute the proximal operator with L1 norm

    Parameters
    ----------
    U : np.ndarray
                U (np.ndarray): observations

    X : np.ndarray
        [description]
    threshold : float
        [description]

    Returns
    -------
    np.ndarray
        array V such that V[i,j] = X[i,j] + max(abs(X[i,j]) - tau,0)
    """

    return X + np.sign(U - X) * np.maximum(np.abs(U - X) - threshold, 0)


def soft_thresholding(X: np.ndarray, threshold: float) -> np.ndarray:
    """Apply the shrinkage operator (i.e. soft thresholding) to the elements of X.

    Parameters
    ----------
    X : np.ndarray
        observed data
    threshold : float
        scaling parameter to shrink the function

    Returns
    -------
    np.ndarray
        array V such that V[i,j] = max(abs(X[i,j]) - tau,0)
    """

    return np.sign(X) * np.maximum(np.abs(X) - threshold, 0)


def svd_thresholding(X: np.ndarray, threshold: float) -> np.ndarray:
    """Apply the shrinkage operator to the singular values obtained from the SVD of X.

    Parameters
    ----------
    X : np.ndarray
        observation
    threshold : float
        scaling parameter to shrink the function

    Returns
    -------
    np.ndarray
        array obtained by computing U * shrink(s) * V where
            U are the left singular vectors of X
            V are the right singular vectors of X
            s are the singular values as a diagonal matrix
    """

    U, s, Vh = np.linalg.svd(X, full_matrices=False, compute_uv=True)
    s = soft_thresholding(s, threshold)
    return np.multiply(U, s) @ Vh


def impute_nans(M: np.ndarray, method: Optional[str] = None) -> np.ndarray:
    """Impute the M's np.nan with the specified method


    Parameters
    ----------
    M : np.ndarray
        array with nan values
    method : Optional[str], optional
        mean or median, by default None

    Returns
    -------
    np.ndarray
        array with imputed nan
    """

    if method == "mean":
        return np.where(np.isnan(M), np.tile(np.nanmean(M, axis=0), (M.shape[0], 1)), M)
    if method == "median":
        return np.where(np.isnan(M), np.tile(np.nanmedian(M, axis=0), (M.shape[0], 1)), M)
    return np.where(np.isnan(M), 0, M)


def ortho_proj(M: np.ndarray, omega: np.ndarray, inv: Optional[int] = 0) -> np.ndarray:
    """orthogonal projection of matrix M onto the linear space of matrices supported on omega

    Parameters
    ----------
    M : np.ndarray
        array to be projected
    omega : np.ndarray
        projector
    inv : Optional[int], optional
        if 0, get projection on omega; if 1, projection on omega^C, by default 0

    Returns
    -------
    np.ndarray
        M' projection on omega
    """
    if inv == 1:
        return M * (np.ones(omega.shape) - omega)
    else:
        return M * omega


def l1_norm(M: np.ndarray) -> float:
    """L1 norm of a matrix seen as a long vector 1 x (np.product(M.shape))

    Parameters
    ----------
    M : np.ndarray


    Returns
    -------
    float
        L1 norm of M seen as a vector
    """
    return np.sum(np.abs(M))


def toeplitz_matrix(T: int, dimension: int) -> np.ndarray:
    """Create a matrix Toeplitz matrix H to take into account temporal correlation via HX
    H=Toeplitz(0,1,-1), in which the central diagonal is defined as ones and
    the T upper diagonal is defined as negative ones.

    Parameters
    ----------
    T : int
        "period", diagonal offset
    dimension : int
        second dimension of H = first dimension of X

    Returns
    -------
    np.ndarray
        Toeplitz matrix
    """

    H = np.eye(dimension - T, dimension)
    H[: dimension - T, T:] = H[: dimension - T, T:] - np.eye(dimension - T, dimension - T)
    return H
