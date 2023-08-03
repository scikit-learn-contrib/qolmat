from typing import Optional, Union
import warnings

import numpy as np
import pandas as pd

from numpy.typing import NDArray

from qolmat.utils.exceptions import NotDimension2, SignalTooShort

HyperValue = Union[int, float, str]


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
    indices = np.arange(n_cols)
    X_interpolated = X.copy()
    median = np.nanmedian(X)
    for i_row in range(n_rows):
        values = X[i_row]
        mask_isna = np.isnan(values)
        if np.all(mask_isna):
            X_interpolated[i_row] = median
        elif np.any(mask_isna):
            values_interpolated = np.interp(
                indices[mask_isna], indices[~mask_isna], values[~mask_isna]
            )
            X_interpolated[i_row, mask_isna] = values_interpolated
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
    n_rows_new = n_rows * period

    X = X.flatten()
    n_required_nans = (-X.size) % n_rows_new
    X = np.append(X, [np.nan] * n_required_nans)
    X = X.reshape(n_rows_new, -1)

    return X


def prepare_data(X: NDArray, period: int = 1) -> NDArray:
    """
    Transform signal to 2D-array in case of 1D-array.
    """
    if len(X.shape) == 1:
        X = X.reshape(1, -1)

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
