from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

from numpy.typing import NDArray
from sklearn.base import check_array

from qolmat.utils.exceptions import NotDimension2, SignalTooShort

HyperValue = Union[int, float, str]


def _get_numerical_features(df1: pd.DataFrame) -> List[str]:
    """Get numerical features from dataframe

    Parameters
    ----------
    df1 : pd.DataFrame

    Returns
    -------
    List[str]
        List of numerical features

    Raises
    ------
    Exception
        No numerical feature is found
    """
    cols_numerical = df1.select_dtypes(include=np.number).columns.tolist()
    if len(cols_numerical) == 0:
        raise Exception("No numerical feature is found.")
    else:
        return cols_numerical


def _get_categorical_features(df1: pd.DataFrame) -> List[str]:
    """Get categorical features from dataframe

    Parameters
    ----------
    df1 : pd.DataFrame

    Returns
    -------
    List[str]
        List of categorical features

    Raises
    ------
    Exception
        No categorical feature is found
    """

    cols_numerical = df1.select_dtypes(include=np.number).columns.tolist()
    cols_categorical = [col for col in df1.columns.to_list() if col not in cols_numerical]
    if len(cols_categorical) == 0:
        raise Exception("No categorical feature is found.")
    else:
        return cols_categorical


def _validate_input(X: NDArray) -> pd.DataFrame:
    """
    Checks that the input X can be converted into a DataFrame, and returns the corresponding
    dataframe.

    Parameters
    ----------
    X : NDArray
        Array-like to process

    Returns
    -------
    pd.DataFrame
        Formatted dataframe, if the input had no column names then the dataframe columns are
        integers
    """
    check_array(X, force_all_finite="allow-nan", dtype=None)
    if not isinstance(X, pd.DataFrame):
        X_np = np.array(X)
        if len(X_np.shape) == 0:
            raise ValueError
        if len(X_np.shape) == 1:
            X_np = X_np.reshape(-1, 1)
        df = pd.DataFrame(X_np, columns=[i for i in range(X_np.shape[1])])
        df = df.infer_objects()
    else:
        df = X
    # df = df.astype(float)

    return df


def progress_bar(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 100,
    fill: str = "█",
):
    """Call in a loop to create terminal progress bar

    Parameters
    ----------
    iteration : int
        current iteration
    total : int
        total iterations
    prefix : str
        prefix string, by default ""
    suffix : str
        suffix string, by default ""
    decimals : int
        positive number of decimals in percent complete, by default 1
    length : int
        character length of bar, by default 100
    fill : str
        bar fill character, by default "█"
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="\r")
    if iteration == total:
        print()


def acf(values: pd.Series, lag_max: int = 30) -> pd.Series:
    """Correlation series of dataseries

    Parameters
    ----------
    values : pd.Series
        dataseries
    lag_max : int, optional
        the maximum lag, by default 30

    Returns
    -------
    pd.Series
        correlation series of value
    """
    acf = pd.Series(0, index=range(lag_max))
    for lag in range(lag_max):
        acf[lag] = values.corr(values.shift(lag))
    return acf


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
    n_rows, n_cols = M.shape
    result = M.copy()
    for i_col in range(n_cols):
        values = M[:, i_col]
        isna = np.isnan(values)
        nna = np.sum(isna)
        if method == "mean":
            value_imputation = np.nanmean(M) if nna == n_rows else np.nanmean(values)
        elif method == "median":
            value_imputation = np.nanmedian(M) if nna == n_rows else np.nanmedian(values)
        elif method == "zeros":
            value_imputation = 0
        else:
            raise ValueError("'method' should be 'mean', 'median' or 'zeros'.")
        result[isna, i_col] = value_imputation

    return result


def linear_interpolation(X: NDArray) -> NDArray:
    """
    Impute missing data with a linear interpolation, column-wise

    Parameters
    ----------
    X : NDArray
        array with missing values

    Returns
    -------
    X_interpolated : NDArray
        imputed array, by linear interpolation
    """
    n_rows, n_cols = X.shape
    indices = np.arange(n_rows)
    X_interpolated = X.copy()
    median = np.nanmedian(X)
    for i_col in range(n_cols):
        values = X[:, i_col]
        mask_isna = np.isnan(values)
        if np.all(mask_isna):
            X_interpolated[:, i_col] = median
        elif np.any(mask_isna):
            values_interpolated = np.interp(
                indices[mask_isna], indices[~mask_isna], values[~mask_isna]
            )
            X_interpolated[mask_isna, i_col] = values_interpolated
    return X_interpolated


def fold_signal(X: NDArray, period: int) -> NDArray:
    """
    Reshape a time series into a 2D-array

    Parameters
    ----------
    X : NDArray
    period : int
        Period used to fold the signal of the 2D-array

    Returns
    -------
    Tuple[NDArray, int]
        Array and number of added nan's fill it

    Raises
    ------
    ValueError
        if X is not a 1D array
    """
    if len(X.shape) != 2:
        raise NotDimension2(X.shape)
    n_rows, n_cols = X.shape
    n_cols_new = n_cols * period

    X = X.flatten()
    n_required_nans = (-X.size) % n_cols_new
    X = np.append(X, [np.nan] * n_required_nans)
    X = X.reshape(-1, n_cols_new)

    return X


def prepare_data(X: NDArray, period: int = 1) -> NDArray:
    """
    Transform signal to 2D-array in case of 1D-array.
    """
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    X_fold = fold_signal(X, period)
    return X_fold


def get_shape_original(M: NDArray, shape: tuple) -> NDArray:
    """Shapes an output matrix from the RPCA algorithm into the original shape.

    Parameters
    ----------
    M : NDArray
        Matrix to reshape
    X : NDArray
        Matrix of the desired shape

    Returns
    -------
    NDArray
        Reshaped matrix
    """
    size = np.prod(shape)
    M_flat = M.flatten()[:size]
    return M_flat.reshape(shape)


def create_lag_matrices(X: NDArray, p: int) -> Tuple[NDArray, NDArray]:
    n_rows, _ = X.shape
    n_rows_new = n_rows - p
    list_X_lag = [np.ones((n_rows_new, 1))]
    for lag in range(p):
        X_lag = X[p - lag - 1 : n_rows - lag - 1, :]
        list_X_lag.append(X_lag)

    Z = np.concatenate(list_X_lag, axis=1)
    Y = X[-n_rows_new:, :]
    return Z, Y


def nan_mean_cov(X: NDArray) -> Tuple[NDArray, NDArray]:
    _, n_variables = X.shape
    means = np.nanmean(X, axis=0)
    cov = np.ma.cov(np.ma.masked_invalid(X), rowvar=False).data
    cov = cov.reshape(n_variables, n_variables)
    return means, cov
