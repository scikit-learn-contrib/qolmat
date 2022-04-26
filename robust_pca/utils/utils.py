"""
General utility functions for rpca
"""

from __future__ import annotations
from typing import Optional, Tuple, List
from numpy.typing import ArrayLike, NDArray

import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
import scipy
from statsmodels import robust

def get_period(signal: ArrayLike, max_period: Optional[int] = None) -> int:
    """
    Retrieve the "period" of a series based on the ACF

    Parameters
    ----------
    signal : List
        time series

    Returns
    -------
    int
        time series' "period" 
    """
    signal_ts = pd.Series(signal)
    if max_period is None:
        max_period = len(signal_ts)
    auto_correlations = [signal_ts.autocorr(lag=lag) for lag in range(1,max_period)]
    return 1 + np.argmax(auto_correlations)

def signal_to_matrix(signal: ArrayLike, period: int) -> Tuple[np.ndarray, int]:
    """
    Shape a time series into a matrix

    Parameters
    ----------
    signal : List
        time series
    period : int
        time series' period, it corresponds to the number of colummns in the resulting matrix

    Returns
    -------
    Tuple[np.ndarray, int]
        matrix and number of added values to match the size (if len(signal)%period != 0)
    """
    n_rows = (len(signal)//period) + 1
    D_signal = np.full((n_rows, period), np.nan)
    D_signal.flat[:len(signal)]=signal                                                
    return D_signal

def approx_rank(D: np.ndarray, threshold: Optional[float]=0.95) -> int:
    """
    Estimate a superior rank of a matrix M by SVD

    Parameters
    ----------
    M : np.ndarray
        matrix 
    threshold : float, optional
        fraction of the cumulative sum of the singular values, by default 0.95
    """
    if threshold == 1:
        return min(D.shape)
    else:
        _, s, _ = np.linalg.svd(D, full_matrices=True)
        nuclear = np.sum(s)
        cum_sum = np.cumsum([i / nuclear for i in s])
        k = np.argwhere(cum_sum > threshold)[0][0] + 1
        return k

def proximal_operator(U: np.ndarray, X: np.ndarray, threshold: float) -> np.ndarray:
    """
    Compute the proximal operator with L1 norm

    Parameters
    ----------
    U : np.ndarray
        observations

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


def impute_nans(D: np.ndarray, method: Optional[str] = "zeros") -> np.ndarray:
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
    if method is None:
        method = "zeros"
    if method == "mean":
        return np.where(np.isnan(D), np.tile(np.nanmean(D, axis=0), (D.shape[0], 1)), D)
    elif method == "median":
        return np.where(np.isnan(D), np.tile(np.nanmedian(D, axis=0), (D.shape[0], 1)), D)
    elif method == "zeros":
        return np.where(np.isnan(D), 0, D)
    else:
        raise ValueError("Invalid method")


def ortho_proj(D: np.ndarray, omega: np.ndarray, inv: Optional[bool] = False) -> np.ndarray:
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
    if inv:
        return D * (~omega)
    else:
        return D * omega

def l1_norm(D: np.ndarray) -> float:
    """
    L1 norm of a matrix

    Parameters
    ----------
    M : np.ndarray


    Returns
    -------
    float
        L1 norm
    """
    return np.sum(np.abs(D))


def toeplitz_matrix(T: int, dimension: int) -> np.ndarray:
    """
    Create a matrix Toeplitz matrix H to take into account temporal correlation via HX
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

def construct_graph(
    X: np.ndarray,
    n_neighbors: Optional[int]=10,
    distance: Optional[str]="euclidean",
    n_jobs: Optional[int]=1
) -> np.ndarray:
    """Construct a graph based on the distance (similarity) between data

    Parameters
    ----------
    X : np.ndarray
        Observations
    n_neighbors : int, optional
        Number of neighbors for each node, by default 10
    distance : str, optional
        Method to construct the weight of the links, by default 'euclidean'
    n_jobs : int, optional
        Number of jobs to run in parallel, by default 1

    Returns
    -------
    np.ndarray
        Graph's adjacency matrix 
    """
    
    G_bin = kneighbors_graph(X, n_neighbors=n_neighbors, metric=distance, mode='connectivity', n_jobs=n_jobs).toarray()
    G_val = kneighbors_graph(X, n_neighbors=n_neighbors, metric=distance, mode='distance', n_jobs=n_jobs).toarray()
    G_val = np.exp(-G_val)
    G_val[~np.array(G_bin, dtype=np.bool)] = 0  
    return G_val
    
def get_laplacian(
    M: np.ndarray,
    normalised: Optional[bool]=True
) -> np.ndarray:
    """Return the Laplacian matrix of a directed graph.

    Parameters
    ----------
    M : np.ndarray
        [description]
    normalised : Optional[bool], optional
        If True, then compute symmetric normalized Laplacian, by default True

    Returns
    -------
    np.ndarray
        Laplacian matrix
    """
    
    return scipy.sparse.csgraph.laplacian(M, normed=normalised)

def get_anomaly(A, X, e=3):
    """
    Filter the matrix A to get anomalies

    Args:
        A (np.nadarray): matrix of "unfiltered" anomalies
        X (np.nadarray): matrix of smooth signal
        e (int, optional): deviation from 0. Defaults to 3.

    Returns:
        np.ndarray: filtered A
        np.ndarray: noise
    """
    mad = robust.mad(X)
    return np.where(np.abs(A) > (e * mad), A, 0), np.where(np.abs(A) <= (e * mad), A, 0)

def resultRPCA_to_signal(
    M1: np.ndarray,
    M2: np.ndarray,
    M3: np.ndarray,
) -> Tuple[List, List, List]:
    """Convert the resulting matrices from RPCA to lists. 
    It makes sense if time series version

    Parameters
    ----------
    M1 : np.ndarray
        Observations
    M2 : np.ndarray
        Low-rank matrix
    M3 : np.ndarray
        Sparse matrix
    ret : int
        Number of added values to form a full matrix. by default 0

    Returns
    -------
    Tuple[List, List, List]
        results of RPCA
    """
    s1 = M1.flatten()
    s2 = M2.flatten()
    s3 = M3.flatten()
    return s1, s2, s3