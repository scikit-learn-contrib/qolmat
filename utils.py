from __future__ import annotations
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

def proximal_operator(U: np.ndarray, X: np.ndarray, threshold: float) -> np.ndarray:
    """
    Compute the proximal operator with L1 norm

    Args:
        U (np.ndarray): observations
        X (np.ndarray): [description]
        threshold (float): [description]

    Returns:
        np.ndarray: V such that V[i,j] = X[i,j] + max(abs(X[i,j]) - tau,0)
    """

    return X + np.sign(U - X) * np.maximum(np.abs(U - X) - threshold, 0)


def soft_thresholding(X: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply the shrinkage operator (i.e. soft thresholding) to the elements of X.

    Args:
        X (np.ndarray): observed data
        threshold (float): scaling parameter to shrink the function

    Returns:
        np.ndarray: V such that V[i,j] = max(abs(X[i,j]) - tau,0)
    """

    return np.sign(X) * np.maximum(np.abs(X) - threshold, 0)


def svd_thresholding(X: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply the shrinkage operator to the singular values obtained from the SVD of X.

    Args:
        X (np.array): observed data
        threshold (float): scaling parameter to shrink the function

    Returns:
        np.array: matrix obtained by computing U * shrink(s) * V where
                U are the left singular vectors of X
                V are the right singular vectors of X
                s are the singular values as a diagonal matrix
    """
    
    U, s, Vh = np.linalg.svd(X, full_matrices=False, compute_uv=True)
    s = soft_thresholding(s, threshold)
    return np.multiply(U, s) @ Vh


def impute_nans(M: np.ndarray, method: Optional[str]=None) -> np.ndarray:    
    """Impute the M's np.nan with the specified method

    Args:
        M (np.array): array with nan values
        method (string, optional): mean or median. Defaults to None.

    Returns:
        [np.array: array with imputed nan
    """
    
    if method == 'mean':
        return np.where(np.isnan(M), np.tile(np.nanmean(M, axis=0), (M.shape[0], 1)), M)
    if method == 'median':
        return np.where(np.isnan(M), np.tile(np.nanmedian(M, axis=0), (M.shape[0], 1)), M)
    return np.where(np.isnan(M), 0, M)


def ortho_proj(M: np.ndarray, omega: np.ndarray, inv: Optional[int]=0) -> np.ndarray:
    """
    orthogonal projection of matrix M onto the linear space of matrices supported on omega
    params:
        M: np.ndarray
        omega: np.ndarray
    return
        np.ndarray
    """
    if inv == 1:
        return M * (np.ones(omega.shape)-omega)
    else:
        return M * omega
    

def l1_norm(M: np.ndarray) -> float:
    """L1 norm of a matrix seen as a long vector

    Args:
        M (np.array): 

    Returns:
        (float): L1 norm of M seen as a vectors
    """
    return np.sum(np.abs(M))

