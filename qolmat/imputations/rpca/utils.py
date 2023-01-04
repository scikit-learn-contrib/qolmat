"""
Modular utility functions for RPCA
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import toeplitz
from sklearn.neighbors import kneighbors_graph


def get_period(signal: np.ndarray, max_period: Optional[int] = None) -> int:
    """
    Retrieve the "period" of a series based on the ACF,
    in an optional given range.

    Parameters
    ----------
    signal : np.ndarray
        time series

    Returns
    -------
    int
        time series' "period"
    """
    ts = pd.Series(signal)
    max_period = len(ts) if max_period is None else max_period
    acf = [round(ts.autocorr(lag=lag), 2) for lag in range(1, max_period + 1)]
    return np.argmax(acf) + 1


def signal_to_matrix(signal: np.ndarray, n_rows: int) -> Tuple[np.ndarray, int]:
    """
    Reshape a time series into a 2D array

    Parameters
    ----------
    signal : np.ndarray
        time series
    n_rows : int
        Number of colummns of the intermediate matrix
        Becareful: the returned matrix is the transposed one.
        That is why n_rows and n_cols are inverted.

    Returns
    -------
    Tuple[NDArray, int]
        matrix and number of added nan's to match the size
        (if len(signal)%period != 0)
    """
    if (len(signal) % n_rows) > 0:
        M = np.append(signal, [np.nan] * (n_rows - (len(signal) % n_rows)))
    else:
        M = signal.copy()
    M = M.reshape(-1, n_rows)
    nb_add_val = (M.shape[0] * M.shape[1]) - len(signal)
    return M.T, nb_add_val


def approx_rank(M: np.ndarray, threshold: Optional[float] = 0.95) -> int:
    """
    Estimate a superior rank of a matrix M by SVD

    Parameters
    ----------
    M : NDArray
        matrix
    th : float, optional
        fraction of the cumulative sum of the singular values, by default 0.95
    """
    if threshold == 1:
        return min(M.shape)
    _, svd, _ = np.linalg.svd(M, full_matrices=True)
    nuclear = np.sum(svd)
    cum_sum = np.cumsum([sv / nuclear for sv in svd])
    return np.argwhere(cum_sum > threshold)[0][0] + 1


def proximal_operator(U: np.ndarray, X: np.ndarray, threshold: float) -> np.ndarray:
    """
    Compute the proximal operator with L1 norm

    Parameters
    ----------
    U : np.ndarray
    X : np.ndarray
    threshold : float

    Returns
    -------
    np.ndarray
        Array V such that V = X + sign(U-X) * max(abs(U-X) - threshold, 0)
    """

    return X + np.sign(U - X) * np.maximum(np.abs(U - X) - threshold, 0)


def soft_thresholding(X: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply the shrinkage operator (i.e. soft thresholding) to the elements of X.

    Parameters
    ----------
    X : np.ndarray
        Observed data
    threshold : float
        Scaling parameter to shrink the function

    Returns
    -------
    np.ndarray
        Array V such that V = sign(X) * max(abs(X - threshold,0)
    """
    return np.sign(X) * np.maximum(np.abs(X) - threshold, 0)


def svd_thresholding(X: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply the shrinkage operator to the singular values obtained from the SVD of X.

    Parameters
    ----------
    X : NDArray
        observation
    threshold : float
        scaling parameter to shrink the function

    Returns
    -------
    NDArray
        Array obtained by computing U * shrink(s) * V where
            U are the left singular vectors of X
            V are the right singular vectors of X
            s are the singular values as a diagonal matrix
    """

    U, s, Vh = np.linalg.svd(X, full_matrices=False)  # , compute_uv=True)
    s = soft_thresholding(s, threshold)
    return np.dot(U, np.dot(np.diag(s), Vh))  # np.multiply(U, SVD) @ Vh


def impute_nans(M: np.ndarray, method: str = "zeros") -> np.ndarray:
    """
    Impute the M's nan with the specified method

    Parameters
    ----------
    M : NDArray
        Array with nan values
    method : str
        'mean' or 'median', or 'zeros'

    Returns
    -------
    NDArray
        Array with imputed nan
    """

    if method == "mean":
        result = np.where(np.isnan(M), np.tile(np.nanmean(M, axis=0), (M.shape[0], 1)), M)
        result = np.where(np.isnan(result), np.nanmean(result), result)
    elif method == "median":
        result = np.where(np.isnan(M), np.tile(np.nanmedian(M, axis=0), (M.shape[0], 1)), M)
        result = np.where(np.isnan(result), np.nanmedian(result), result)
    elif method == "zeros":
        result = np.where(np.isnan(M), 0, M)
    else:
        raise ValueError("'method' should be 'mean', 'median' or 'zeros'.")
    return result


def ortho_proj(M: np.ndarray, omega: np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    Orthogonal projection of matrix M onto the linear space of matrices supported on omega

    Parameters
    ----------
    M : np.ndarray
        array to be projected
    omega : np.ndarray
        projector
    inverse : bool
        if False, get projection on omega; if True, projection on omega^C, by default False.

    Returns
    -------
    np.ndarray
        M' projection on omega
    """
    if inverse:
        return M * (~omega)
    else:
        return M * omega


def l1_norm(M: np.ndarray) -> float:
    """
    L1 norm of a matrix

    Parameters
    ----------
    M : np.ndarray


    Returns
    -------
    float
        L1 norm of M
    """
    return np.sum(np.abs(M))


def toeplitz_matrix(T: int, dimension: int, model: str) -> np.ndarray:
    """
    Create a matrix Toeplitz matrix H to take into account temporal correlation via HX
    H=Toeplitz(0,1,-1), in which the central diagonal is defined as ones and
    the T upper diagonal is defined as minus ones.
    Dimensions depend on the modelisation: if each observation is a row or a column
    if row: H to be HX, if column: H to be XH

    Parameters
    ----------
    T : int
        "period", diagonal offset
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


def construct_graph(
    X: np.ndarray,
    n_neighbors: Optional[int] = 10,
    distance: Optional[str] = "euclidean",
    n_jobs: Optional[int] = 1,
) -> np.ndarray:
    """
    Construct a graph based on the distance (similarity) between data

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
    NDArray
        Graph's adjacency matrix
    """
    G_val = kneighbors_graph(
        X=X, n_neighbors=n_neighbors, metric=distance, mode="distance", n_jobs=n_jobs
    ).toarray()

    G_val = np.exp(-G_val)
    G_val[G_val < 1.0] = 0
    return G_val


def get_laplacian(M: np.ndarray, normalised: Optional[bool] = True) -> np.ndarray:
    """
    Return the Laplacian matrix of a directed graph.

    Parameters
    ----------
    M : np.ndarray
        Adjacency matrix of the directed graph??
    normalised : Optional[bool]
        If True, then compute symmetric normalized Laplacian,
        by default True

    Returns
    -------
    np.ndarray
        Laplacian matrix
    """
    return scipy.sparse.csgraph.laplacian(M, normed=normalised)


def get_anomaly(A: np.ndarray, X: np.ndarray, e: Optional[float] = 3):
    """
    Filter the matrix A to get anomalies

    Args:
        A : np.ndarray
            matrix of "unfiltered" anomalies
        X : np.ndarray
            matrix of smooth signal
        e : Optional[float]
            deviation from 0. Defaults to 3.

    Returns:
        np.ndarray: filtered A
        np.ndarray: noise
    """
    # mad = robust.mad(X, axis=1)
    mad = np.median(np.abs(X), axis=1)
    filtered_A = np.where(np.abs(A) > (e * mad), A, 0)
    noise = np.where(np.abs(A) <= (e * mad), A, 0)
    return filtered_A, noise


def solve_proj2(
    m: np.ndarray,
    U: np.ndarray,
    lam1: float,
    lam2: float,
    maxIter: Optional[int] = 10_000,
    tol: Optional[float] = 1e-6,
):
    """
    solve the problem:
    min_{v, s} 0.5*|m-Uv-s|_2^2 + 0.5*lambda1*|v|^2 + lambda2*|s|_1

    solve the projection by APG

    Parameters
    ----------
    m : np.ndarray
        (n x 1) vector, vector/sample to project
    U : np.ndarray
        (n x p) matrix, basis
    lam1 : float
        tuning param for the nuclear norm in the initial problem
    lam2 : float
        tuning param for the L1 norm of anomalies in the initial problem
    maxIter : int, optional
        maximum number of iterations before stopping, by default 10_000
    tol : float, optional
        tolerance for convergence, by default 1e-6

    Returns
    -------
    np.ndarray, np.ndarray
        vectors: coefficients, sparse part
    """
    n, p = U.shape
    v = np.zeros(p)
    s = np.zeros(n)
    Identity = np.identity(p)

    UUt = np.linalg.inv(U.transpose().dot(U) + lam1 * Identity).dot(U.transpose())
    for _ in range(maxIter):
        vtemp = v.copy()
        v = UUt.dot(m - s)
        stemp = s
        s = soft_thresholding(m - U.dot(v), lam2)
        stopc = max(np.linalg.norm(v - vtemp), np.linalg.norm(s - stemp)) / n
        if stopc < tol:
            break
    return v, s


def solve_projection(
    z: np.ndarray,
    L: np.ndarray,
    lam1: float,
    lam2: float,
    list_lams: List[float],
    list_periods: List[int],
    X: np.ndarray,
    maxIter: Optional[int] = 10_000,
    tol: Optional[float] = 1e-6,
):
    """
    solve the problem:
    min_{v, s} 0.5*|m-Uv-s|_2^2 + 0.5*lambda1*|v|^2 +
    lambda2*|s|_1 + sum_k eta_k |Lq-Lq_{-T_k}|_2^2

    projection with temporal regularisations

    solve the projection by APG

    Parameters
    ----------
    z : np.ndarray
        vector to project
    L : np.ndarray
        basis
    lam1 : float
        tuning param for the nuclear norm in the initial problem
    lam2 : float
        tuning param for the L1 norm of anomalies in the initial problem
    list_lams : list[float]
        tuning param for the L2 norm of temporal regularizations in the initial problem
    list_periods : list[int]
        list of "periods" for the Toeplitz matrices in the initial problem
    X : np.ndarray
        low rank part already computed (during the burnin phase)
    maxIter : int, optional
        maximum number of iterations before stopping, by default 10_000
    tol : float, optional
        tolerance for convergence, by default 1e-6

    Returns
    -------
    np.ndarray, np.ndarray
        vectors: coefficients, sparse part
    """
    n, p = L.shape
    r = np.zeros(p)
    e = np.zeros(n)
    Identity = np.identity(p)

    sums = 2.0 * np.sum(list_lams)
    sums_rk = np.zeros(n)
    for a, b in zip(list_lams, list_periods):
        sums_rk += 2 * a * X[:, -b]

    tmp = np.linalg.inv(L.T @ L + lam1 * Identity + L.T @ L * sums)

    for _ in range(maxIter):
        rtemp = r
        etemp = e

        r = tmp @ L.T @ (z - e + sums_rk)
        e = soft_thresholding(z - L.dot(r), lam2)

        stopc = max(np.linalg.norm(r - rtemp), np.linalg.norm(e - etemp)) / n
        if stopc < tol:
            break

    return r, e


def update_col(lam: float, U: np.ndarray, A: np.ndarray, B: np.ndarray):
    """
    Update column of matrix U
    See Algo 2. p5 of Feng, Jiashi, Huan Xu, and Shuicheng Yan.
    "Online robust pca via stochastic optimization.""
    Advances in Neural Information Processing Systems. 2013.

    Block-coordinate descent with warm restarts

    Parameters
    ----------
    lam: float
        tuning param for the nuclear norm in the initial problem
    U : np.ndarray
        matrix to update
    A : np.ndarray
        see algorithm 1 in ...
    B : np.ndarray
        see algorithm 1 in ...

    Returns
    -------
    np.ndarray
        updated matrix
    """

    _, r = U.shape
    A = A + lam * np.identity(r)
    for j in range(r):
        bj = B[:, j]
        uj = U[:, j]
        aj = A[:, j]
        temp = (bj - U.dot(aj)) / A[j, j] + uj
        U[:, j] = temp / max(np.linalg.norm(temp), 1)
    return U
