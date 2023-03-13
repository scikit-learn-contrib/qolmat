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


def get_period(
    signal: NDArray,
    max_period: Optional[int] = None,
) -> int:
    """
    Return the lag of maximum auto-correlation based on the Auto-Correlation Function,
    in an possibly given range.

    Parameters
    ----------
    signal : NDArray
        time series

    Returns
    -------
    int
        lag of maximum auto-correlation
        of the time series
    """
    ts = pd.Series(signal)
    max_period = len(ts) // 3 if max_period is None else max_period
    acf = [round(ts.autocorr(lag=lag), 2) for lag in range(1, max_period + 1)]
    return np.argmax(acf) + 1


def fold_signal(X: NDArray, n_rows: int) -> Tuple[NDArray, int]:
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


def proximal_operator(U: NDArray, X: NDArray, threshold: float) -> NDArray:
    """
    Compute the proximal operator with L1-norm.

    Parameters
    ----------
    U : NDArray
    X : NDArray
    threshold : float

    Returns
    -------
    NDArray
        Array V such that V = X + sign(U-X) * max(abs(U-X) - threshold, 0)
    """
    return X + np.sign(U - X) * np.maximum(np.abs(U - X) - threshold, 0)


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


def ortho_proj(
    M: NDArray,
    omega: NDArray,
    inverse: bool = False,
) -> NDArray:
    """
    Orthogonal projection of the array M
    nullified out of omega.

    Parameters
    ----------
    M : NDArray
        Array to be projected
    omega : NDArray
        Non-null space
    inverse : bool
        If ``False``, projection on omega
        If ``True``, projection on omega^C
        By default ``False``.

    Returns
    -------
    NDArray
       Array equals to M on omega
       and zero elsewhere
    """
    if inverse:
        return M * (~omega)
    else:
        return M * omega


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


def construct_graph(
    X: NDArray,
    n_neighbors: int = 10,
    distance: str = "euclidean",
    n_jobs: int = 1,
) -> NDArray:
    """
    Construct a graph based on the distance (similarity) between data

    Parameters
    ----------
    X : NDArray
    n_neighbors : int
        Number of neighbors for each node, by default 10
    distance : str
        Method to construct the weight of the ties, by default 'euclidean'
    n_jobs : int
        Number of jobs for parallel processing
    Returns
    -------
    NDArray
        Graph's adjacency matrix
    """
    G_val = kneighbors_graph(
        X=X,
        n_neighbors=n_neighbors,
        metric=distance,
        mode="distance",
        n_jobs=n_jobs,
    ).toarray()

    G_val = np.exp(-G_val)
    G_val[G_val < 1.0] = 0
    return G_val


def get_laplacian(M: NDArray, normalised: bool = True) -> NDArray:
    """
    Return the Laplacian matrix of a directed graph.

    Parameters
    ----------
    M : NDArray
        Adjacency matrix of the directed graph.
    normalised : bool
        Argument 'normed' of scipy.sparse.csgraph.laplacian.

    Returns
    -------
    NDArray
        Laplacian array
    """
    return scipy.sparse.csgraph.laplacian(M, normed=normalised)


def solve_proj2(
    M: NDArray,
    U: NDArray,
    lam1: float,
    lam2: float,
    max_iter: int = int(1e4),
    tol: float = 1e-6,
) -> Tuple[NDArray, NDArray]:
    """
    solve the problem:
    min_{v, s} 0.5*|m-Uv-s|_2^2 + 0.5*lambda1*|v|^2 + lambda2*|s|_1

    solve the projection by APG

    Parameters
    ----------
    M : NDArray of shape (n, )
        Array to project
    U : NDArray of shape (n, p)
        Basis
    lam1 : float
        Tuning parameter for the nuclear norm.
    lam2 : float
        Tuning parameter for the L1 norm of anomalies.
    max_iter : int
        Maximum number of iterations, by default 1e4
    tol : float
        tolerance for convergence, by default 1e-6

    Returns
    -------
    NDArray, NDArray
        coefficients and sparse part
    """
    n, p = U.shape
    v = np.zeros(p)
    s = np.zeros(n)
    identity = np.identity(p)

    UUt = np.linalg.inv(U.transpose().dot(U) + lam1 * identity).dot(U.transpose())
    for _ in range(max_iter):
        vtemp = v.copy()
        v = UUt.dot(M - s)
        stemp = s.copy()
        s = soft_thresholding(M - U.dot(v), lam2)
        stopc = max(np.linalg.norm(v - vtemp) / p, np.linalg.norm(s - stemp) / n)
        if stopc < tol:
            break
    return v, s


def solve_projection(
    Z: NDArray,
    L: NDArray,
    lam1: float,
    lam2: float,
    list_lams: List[float],
    list_periods: List[int],
    X: NDArray,
    max_iter: int = int(1e4),
    tol: float = 1e-6,
):
    """
    Solve the problem:
    min_{v, s} 0.5*|m-Uv-s|_2^2 + 0.5*lambda1*|v|^2 +
    lambda2*|s|_1 + sum_k eta_k |Lq-Lq_{-T_k}|_2^2

    Projection with temporal regularizations

    Solve the projection by APG

    Parameters
    ----------
    Z : NDArray
        vector to project
    L : NDArray
        Basis
    lam1 : float
        Tuning parameter for the nuclear norm
    lam2 : float
        Tuning parameter for the L1 norm of anomalies
    list_lams : list[float]
        Tuning parameters for the L2 norm of temporal regularizations
    list_periods : list[int]
        Shifts for the Toeplitz matrices
    X : NDArray
        Low rank part already computed (at burnin step)
    max_iter : int
        Maximum number of iterations, by default 1e4
    tol : float
        Tolerance for convergence, by default 1e-6

    Returns
    -------
    NDArray, NDArray
        coefficients, sparse part
    """
    n, p = L.shape
    r = np.zeros(p)
    e = np.zeros(n)
    identity = np.identity(p)

    sums = 2.0 * np.sum(list_lams)
    sums_rk = np.zeros(n)
    for a, b in zip(list_lams, list_periods):
        sums_rk += 2 * a * X[:, -b]

    tmp = np.linalg.inv(L.T @ L + lam1 * identity + L.T @ L * sums)

    for _ in range(max_iter):
        rtemp = r.copy()
        etemp = e.copy()

        r = tmp @ L.T @ (Z - e + sums_rk)
        e = soft_thresholding(Z - L.dot(r), lam2)

        stopc = max(np.linalg.norm(r - rtemp) / p, np.linalg.norm(e - etemp) / n)
        if stopc < tol:
            break

    return r, e


def update_col(lam: float, U: NDArray, A: NDArray, B: NDArray):
    """
    Update column of matrix U
    See Algo 2. p5 of Feng, Jiashi, Huan Xu, and Shuicheng Yan.
    "Online robust pca via stochastic optimization.""
    Advances in Neural Information Processing Systems. 2013.

    Block-coordinate descent with warm restarts

    Parameters
    ----------
    lam: float
        Tuning parameter for the nuclear norm
    U : NDArray
    A : NDArray
    B : NDArray

    Returns
    -------
    NDArray
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
