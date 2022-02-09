"""
General utility functions for rpca
"""

from __future__ import annotations
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
import scipy

def get_period(signal: List) -> int:
    """Retrieve the "period" of a series based on the ACF

    Parameters
    ----------
    signal : List
        time series

    Returns
    -------
    int
        time series' "period" 
    """
    ss = pd.Series(signal)
    val = []
    for i in range(len(signal)):
        val.append(ss.autocorr(lag=i))

    ind_sort = sorted(range(len(val)), key=lambda k: val[k])
    period = ind_sort[::-1][1]
    
    return period


def signal_to_matrix(signal: List, period: int) -> Tuple[np.ndarray, int]:
    """Shape a time series into a matrix

    Parameters
    ----------
    signal : List
        time series
    period : int
        time series' period, it corresponds to the number of colummns the resulting matrix

    Returns
    -------
    Tuple[np.ndarray, int]
        matrix and number of added values to match the size (if len(signal)%period != 0)
    """

    modulo = len(signal) % period
    nb_add_val = (period - modulo) % period
    signal += [np.nan] * nb_add_val

    M = np.array(signal).reshape(-1, period)
    return M, nb_add_val

def approx_rank(M: np.ndarray, th: Optional[float]=0.95) -> int:
    """Estimate a superior rank of a matrix M by SVD

    Parameters
    ----------
    M : np.ndarray
        matrix 
    th : float, optional
        fraction of the cumulative sum of the singular values, by default 0.95
    """
    _, s, _ = np.linalg.svd(M, full_matrices=True)
    nuclear = np.sum(s)
    cum_sum = np.cumsum([i / nuclear for i in s])
    k = np.argwhere(cum_sum > th)[0][0] + 1
    return k

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


def resultRPCA_to_signal(
    M1: np.ndarray,
    M2: np.ndarray,
    M3: np.ndarray,
    ret: Optional[int]=0 
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
        results of RPCA in list form
    """
    
    if ret > 0:
        s1 = M1.flatten().tolist()[:-ret]
        s2 = M2.flatten().tolist()[:-ret]
        s3 = M3.flatten().tolist()[:-ret]
    else:
        s1 = M1.flatten().tolist()
        s2 = M2.flatten().tolist()
        s3 = M3.flatten().tolist()
        
    return s1, s2, s3