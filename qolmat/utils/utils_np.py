from typing import Optional

import numpy as np
from numpy.typing import NDArray


def _linear_interpolation(X: NDArray) -> NDArray:
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
    for i_row in range(n_rows):
        values = X[i_row]
        mask_isna = np.isnan(values)
        values_interpolated = np.interp(
            indices[mask_isna], indices[~mask_isna], values[~mask_isna]
        )
        X_interpolated[i_row, mask_isna] = values_interpolated
    return X_interpolated


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


def _prepare_data(X: NDArray, period: Optional[int] = None) -> NDArray:
    """
    Transform signal to 2D-array in case of 1D-array.
    """
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    n_rows_X, n_cols_X = X.shape
    if n_rows_X == 1:
        if period is None:
            raise ValueError("`period` must be specified when imputing 1D data.")
        elif period >= n_cols_X:
            raise ValueError("`period` must be smaller than the signals duration.")
        return fold_signal(X, period)
    else:
        if period is None:
            return X.copy()
        else:
            raise ValueError("`period` should not be specified when imputing 2D data.")


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
