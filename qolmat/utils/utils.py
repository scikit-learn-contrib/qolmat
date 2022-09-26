"""
General utility functions for rpca
"""

from __future__ import annotations
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from numpy.typing import NDArray
import scipy
import torch


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


def approx_rank(M: np.ndarray, th: Optional[float] = 1) -> int:
    """Estimate a superior rank of a matrix M by SVD

    Parameters
    ----------
    M : np.ndarray
        matrix
    th : float, optional
        fraction of the cumulative sum of the singular values, by default 0.95
    """
    if th == 1:
        return min(M.shape)
    else:
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
        return np.where(
            np.isnan(M), np.tile(np.nanmedian(M, axis=0), (M.shape[0], 1)), M
        )
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
    H[: dimension - T, T:] = H[: dimension - T, T:] - np.eye(
        dimension - T, dimension - T
    )
    return H


def construct_graph(
    X: NDArray,
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
    np.ndarray
        Graph's adjacency matrix
    """

    G_bin = kneighbors_graph(
        X, n_neighbors=n_neighbors, metric=distance, mode="connectivity", n_jobs=n_jobs
    ).toarray()
    G_val = kneighbors_graph(
        X, n_neighbors=n_neighbors, metric=distance, mode="distance", n_jobs=n_jobs
    ).toarray()
    G_val = np.exp(-G_val)
    G_val[~np.array(G_bin, dtype=np.bool)] = 0
    return G_val


def get_laplacian(M: np.ndarray, normalised: Optional[bool] = True) -> np.ndarray:
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
    mad = np.median(np.abs(X - np.median(X)))
    return np.where(np.abs(A) > (e * mad), A, 0), np.where(np.abs(A) <= (e * mad), A, 0)


def resultRPCA_to_signal(
    M1: np.ndarray, M2: np.ndarray, M3: np.ndarray, ret: Optional[int] = 0
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


def compute_distri(sample):
    return np.unique(sample, return_counts=True)[1] / len(sample)


def KL(P: pd.Series, Q: pd.Series) -> float:
    """
    Compute the Kullback-Leibler divergence between distributions P and Q
    Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0.

    Parameters
    ----------
    P : pd.Series
        "true" distribution
    Q : pd.Series
        sugggesetd distribution

    Return
    ------
    float
        KL(P,Q)
    """
    epsilon = 0.00001

    P = P.copy() + epsilon
    Q = Q.copy() + epsilon

    return np.sum(P * np.log(P / Q))


#################################################################################################
# Missing data mechanisms with numpy or pytorch outputs                                         #
# codes from BorisMuzellec: https://github.com/BorisMuzellec/MissingDataOT/blob/master/utils.py #
#################################################################################################


def nanmean(v, *args, **kwargs):
    """
    A Pytorch version on Numpy's nanmean
    """
    v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


#### Quantile ######
def quantile(X, q, dim=None):
    """
    Returns the q-th quantile.

    Parameters
    ----------
    X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
        Input data.

    q : float
        Quantile level (starting from lower values).

    dim : int or None, default = None
        Dimension allong which to compute quantiles. If None, the tensor is flattened and one value is returned.


    Returns
    -------
        quantiles : torch.DoubleTensor

    """
    return X.kthvalue(int(q * len(X)), dim=dim)[0]


#### Automatic selection of the regularization parameter ####
def pick_epsilon(X, quant=0.5, mult=0.05, max_points=2000):
    """
        Returns a quantile (times a multiplier) of the halved pairwise squared distances in X.
        Used to select a regularization parameter for Sinkhorn distances.

    Parameters
    ----------
    X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
        Input data on which distances will be computed.

    quant : float, default = 0.5
        Quantile to return (default is median).

    mult : float, default = 0.05
        Mutiplier to apply to the quantiles.

    max_points : int, default = 2000
        If the length of X is larger than max_points, estimate the quantile on a random subset of size max_points to
        avoid memory overloads.

    Returns
    -------
        epsilon: float

    """
    means = nanmean(X, 0)
    X_ = X.clone()
    mask = torch.isnan(X_)
    X_[mask] = (mask * means)[mask]

    idx = np.random.choice(len(X_), min(max_points, len(X_)), replace=False)
    X = X_[idx]
    dists = ((X[:, None] - X) ** 2).sum(2).flatten() / 2.0
    dists = dists[dists > 0]

    return quantile(dists, quant, 0).item() * mult


#### Accuracy Metrics ####
def MAE(X, X_true, mask):
    """
    Mean Absolute Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.

    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.

    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)

    Returns
    -------
        MAE : float

    """
    if torch.is_tensor(mask):
        mask_ = mask.bool()
        return torch.abs(X[mask_] - X_true[mask_]).sum() / mask_.sum()
    else:  # should be an ndarray
        mask_ = mask.astype(bool)
        return np.absolute(X[mask_] - X_true[mask_]).sum() / mask_.sum()


def RMSE(X, X_true, mask):
    """
    Root Mean Squared Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.

    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.

    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)

    Returns
    -------
        RMSE : float

    """
    if torch.is_tensor(mask):
        mask_ = mask.bool()
        return (((X[mask_] - X_true[mask_]) ** 2).sum() / mask_.sum()).sqrt()
    else:  # should be an ndarray
        mask_ = mask.astype(bool)
        return np.sqrt(((X[mask_] - X_true[mask_]) ** 2).sum() / mask_.sum())


##################### MISSING DATA MECHANISMS #############################

##### Missing At Random ######


def MAR_mask(X, p, p_obs):
    """
    Missing at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated. If a numpy array is provided,
        it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    p_obs : float
        Proportion of variables with *no* missing values that will be used for the logistic masking model.

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    to_torch = torch.is_tensor(X)  ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_obs = max(
        int(p_obs * d), 1
    )  ## number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs  ## number of variables that will have missing values

    ### Sample variables that will all be observed, and those with missing values:
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    ### Other variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask


##### Missing not at random ######


def MNAR_mask_logistic(X, p, p_params=0.3, exclude_inputs=True):
    """
    Missing not at random mechanism with a logistic masking model. It implements two mechanisms:
    (i) Missing probabilities are selected with a logistic model, taking all variables as inputs. Hence, values that are
    inputs can also be missing.
    (ii) Variables are split into a set of intputs for a logistic model, and a set whose missing probabilities are
    determined by the logistic model. Then inputs are then masked MCAR (hence, missing values from the second set will
    depend on masked values.
    In either case, weights are random and the intercept is selected to attain the desired proportion of missing values.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    p_params : float
        Proportion of variables that will be used for the logistic masking model (only if exclude_inputs).

    exclude_inputs : boolean, default=True
        True: mechanism (ii) is used, False: (i)

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    to_torch = torch.is_tensor(X)  ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_params = (
        max(int(p_params * d), 1) if exclude_inputs else d
    )  ## number of variables used as inputs (at least 1)
    d_na = (
        d - d_params if exclude_inputs else d
    )  ## number of variables masked with the logistic model

    ### Sample variables that will be parameters for the logistic regression:
    idxs_params = (
        np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
    )
    idxs_nas = (
        np.array([i for i in range(d) if i not in idxs_params])
        if exclude_inputs
        else np.arange(d)
    )

    ### Other variables will have NA proportions selected by a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_params, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_params], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_params].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    ## If the inputs of the logistic model are excluded from MNAR missingness,
    ## mask some values used in the logistic model at random.
    ## This makes the missingness of other variables potentially dependent on masked values

    if exclude_inputs:
        mask[:, idxs_params] = torch.rand(n, d_params) < p

    return mask


def MNAR_self_mask_logistic(X, p):
    """
    Missing not at random mechanism with a logistic self-masking model. Variables have missing values probabilities
    given by a logistic model, taking the same variable as input (hence, missingness is independent from one variable
    to another). The intercepts are selected to attain the desired missing rate.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    to_torch = torch.is_tensor(X)  ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    ### Variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, self_mask=True)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X, coeffs, p, self_mask=True)

    ps = torch.sigmoid(X * coeffs + intercepts)

    ber = torch.rand(n, d) if to_torch else np.random.rand(n, d)
    mask = ber < ps if to_torch else ber < ps.numpy()

    return mask


def MNAR_mask_quantiles(X, p, q, p_params, cut="both", MCAR=False):
    """
    Missing not at random mechanism with quantile censorship. First, a subset of variables which will have missing
    variables is randomly selected. Then, missing values are generated on the q-quantiles at random. Since
    missingness depends on quantile information, it depends on masked values, hence this is a MNAR mechanism.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    q : float
        Quantile level at which the cuts should occur

    p_params : float
        Proportion of variables that will have missing values

    cut : 'both', 'upper' or 'lower', default = 'both'
        Where the cut should be applied. For instance, if q=0.25 and cut='upper', then missing values will be generated
        in the upper quartiles of selected variables.

    MCAR : bool, default = True
        If true, masks variables that were not selected for quantile censorship with a MCAR mechanism.

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """
    n, d = X.shape

    to_torch = torch.is_tensor(X)  ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_na = max(int(p_params * d), 1)  ## number of variables that will have NMAR values

    ### Sample variables that will have imps at the extremes
    idxs_na = np.random.choice(
        d, d_na, replace=False
    )  ### select at least one variable with missing values

    ### check if values are greater/smaller that corresponding quantiles
    if cut == "upper":
        quants = quantile(X[:, idxs_na], 1 - q, dim=0)
        m = X[:, idxs_na] >= quants
    elif cut == "lower":
        quants = quantile(X[:, idxs_na], q, dim=0)
        m = X[:, idxs_na] <= quants
    elif cut == "both":
        u_quants = quantile(X[:, idxs_na], 1 - q, dim=0)
        l_quants = quantile(X[:, idxs_na], q, dim=0)
        m = (X[:, idxs_na] <= l_quants) | (X[:, idxs_na] >= u_quants)

    ### Hide some values exceeding quantiles
    ber = torch.rand(n, d_na)
    mask[:, idxs_na] = (ber < p) & m

    if MCAR:
        ## Add a mcar mecanism on top
        mask = mask | (torch.rand(n, d) < p)

    return mask


def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
    n, d = X.shape
    if self_mask:
        coeffs = torch.randn(d)
        Wx = X * coeffs
        coeffs /= torch.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = torch.randn(d_obs, d_na)
        Wx = X[:, idxs_obs].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
    return coeffs


def fit_intercepts(X, coeffs, p, self_mask=False):
    if self_mask:
        d = len(coeffs)
        intercepts = torch.zeros(d)
        for j in range(d):

            def f(x):
                return torch.sigmoid(X * coeffs[j] + x).mean().item() - p

            intercepts[j] = scipy.optimize.bisect(f, -50, 50)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = torch.zeros(d_na)
        for j in range(d_na):

            def f(x):
                return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p

            intercepts[j] = scipy.optimize.bisect(f, -50, 50)
    return intercepts


# Function produce_NA for generating missing values ------------------------------------------------------


def produce_NA(X, p_miss, mecha="MCAR", opt=None, p_obs=None, q=None):
    """
    Generate missing values for specifics missing-data mechanism and proportion of missing values.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p_miss : float
        Proportion of missing values to generate for variables which will have missing values.
    mecha : str,
            Indicates the missing-data mechanism to be used. "MCAR" by default, "MAR", "MNAR" or "MNARsmask"
    opt: str,
         For mecha = "MNAR", it indicates how the missing-data mechanism is generated: using a logistic regression ("logistic"), quantile censorship ("quantile") or logistic regression for generating a self-masked MNAR mechanism ("selfmasked").
    p_obs : float
            If mecha = "MAR", or mecha = "MNAR" with opt = "logistic" or "quanti", proportion of variables with *no* missing values that will be used for the logistic masking model.
    q : float
        If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.

    Returns
    ----------
    A dictionnary containing:
    'X_init': the initial data matrix.
    'X_incomp': the data with the generated missing values.
    'mask': a matrix indexing the generated missing values.s
    """

    to_torch = torch.is_tensor(X)  ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = X.astype(np.float32)
        X = torch.from_numpy(X)

    if mecha == "MAR":
        mask = MAR_mask(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "logistic":
        mask = MNAR_mask_logistic(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "quantile":
        mask = MNAR_mask_quantiles(X, p_miss, q, 1 - p_obs).double()
    elif mecha == "MNAR" and opt == "selfmasked":
        mask = MNAR_self_mask_logistic(X, p_miss).double()
    else:
        mask = (torch.rand(X.shape) < p_miss).double()

    X_nas = X.clone()
    X_nas[mask.bool()] = np.nan

    return {"X_init": X.double(), "X_incomp": X_nas.double(), "mask": mask}
