"""Utils for qolmat package."""

from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import check_array

from qolmat.utils.exceptions import NotDimension2

HyperValue = Union[int, float, str]


def _get_numerical_features(df1: pd.DataFrame) -> List[str]:
    """Get numerical features from dataframe.

    Parameters
    ----------
    df1 : pd.DataFrame
        Input dataframe.

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
    """Get categorical features from dataframe.

    Parameters
    ----------
    df1 : pd.DataFrame
        Input dataframe.

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
    cols_categorical = [
        col for col in df1.columns.to_list() if col not in cols_numerical
    ]
    if len(cols_categorical) == 0:
        raise Exception("No categorical feature is found.")
    else:
        return cols_categorical


def _validate_input(X: NDArray) -> pd.DataFrame:
    """Calidate the input array.

    Checks that the input X can be converted into a DataFrame,
    and returns the corresponding dataframe.

    Parameters
    ----------
    X : NDArray
        Array-like to process

    Returns
    -------
    pd.DataFrame
        Formatted dataframe, if the input had no column names
        then the dataframe columns are integers

    """
    check_array(X, force_all_finite="allow-nan", dtype=None)
    if not isinstance(X, pd.DataFrame):
        X_np = np.array(X)
        if len(X_np.shape) == 0:
            raise ValueError
        if len(X_np.shape) == 1:
            X_np = X_np.reshape(-1, 1)
        df = pd.DataFrame(X_np, columns=list(range(X_np.shape[1])))
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
    """Call in a loop to create terminal progress bar.

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
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total))
    )
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="\r")
    if iteration == total:
        print()


def acf(values: pd.Series, lag_max: int = 30) -> pd.Series:
    """Correlation series of dataseries.

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
    """Impute the M's nan with the specified method.

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
            value_imputation = (
                np.nanmean(M) if nna == n_rows else np.nanmean(values)
            )
        elif method == "median":
            value_imputation = (
                np.nanmedian(M) if nna == n_rows else np.nanmedian(values)
            )
        elif method == "zeros":
            value_imputation = 0
        else:
            raise ValueError("'method' should be 'mean', 'median' or 'zeros'.")
        result[isna, i_col] = value_imputation

    return result


def linear_interpolation(X: NDArray) -> NDArray:
    """Impute missing data with a linear interpolation, column-wise.

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
    """Reshape a time series into a 2D-array.

    Parameters
    ----------
    X : NDArray
        Input array to be reshaped.
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
    """Reshape a time series into a 2D-array.

    Parameters
    ----------
    X : NDArray
        Input array to be reshaped.
    period : int, optional
        Period used to fold the signal. Defaults to 1.

    Returns
    -------
    NDArray
        Reshaped array.

    """
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    X_fold = fold_signal(X, period)
    return X_fold


def get_shape_original(M: NDArray, shape: Tuple[int, int]) -> NDArray:
    """Shapes an output matrix from the RPCA algorithm into the original shape.

    Parameters
    ----------
    M : NDArray
        Matrix to reshape
    shape : Tuple[int, int]
        Desired shape

    Returns
    -------
    NDArray
        Reshaped matrix

    """
    size: int = int(np.prod(shape))
    M_flat = M.flatten()[:size]
    return M_flat.reshape(shape)


def create_lag_matrices(X: NDArray, p: int) -> Tuple[NDArray, NDArray]:
    """Create lag matrices for the VAR(p).

    Parameters
    ----------
    X : NDArray
        Input matrix
    p : int
        Number of lags

    Returns
    -------
    Tuple[NDArray, NDArray]
        Z and Y

    """
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
    """Compute mean and covariance matrix.

    Parameters
    ----------
    X : NDArray
        Input matrix

    Returns
    -------
    Tuple[NDArray, NDArray]
        Means and covariance matrix

    """
    _, n_variables = X.shape
    means = np.nanmean(X, axis=0)
    cov = np.ma.cov(np.ma.masked_invalid(X), rowvar=False).data
    cov = cov.reshape(n_variables, n_variables)
    return means, cov


def moy_p(V, weights):
    """Compute the weighted mean of a vector, ignoring NaNs.

    Parameters
    ----------
    V : array-like
        Input vector with possible NaN values.
    weights : array-like
        Weights corresponding to each element in V.

    Returns
    -------
    float
        Weighted mean of non-NaN elements.

    """
    mask = ~np.isnan(V)
    total_weight = np.sum(weights[mask])
    if total_weight == 0:
        return 0.0  # or use np.finfo(float).eps for a small positive value
    return np.sum(V[mask] * weights[mask]) / total_weight


def tab_disjonctif_NA(df):
    """Create a disjunctive (one-hot encoded).

    Parameters
    ----------
    df : DataFrame
        Input DataFrame with categorical and numeric variables.

    Returns
    -------
    DataFrame
        Disjunctive table with one-hot encoding.

    """  # noqa: E501
    df_encoded_list = []
    for col in df.columns:
        if df[col].dtype.name == "category" or df[col].dtype == object:
            df[col] = df[col].astype("category")
            # Include '__MISSING__' as a category if not already present
            if "__MISSING__" not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(["__MISSING__"])
            # Fill missing values with '__MISSING__'
            df[col] = df[col].fillna("__MISSING__")
            # One-hot encode the categorical variable
            encoded = pd.get_dummies(
                df[col],
                prefix=col,
                prefix_sep="_",
                dummy_na=False,
                dtype=float,
            )
            df_encoded_list.append(encoded)
        else:
            # Numeric column; keep as is
            df_encoded_list.append(df[[col]])
    # Concatenate all encoded columns
    df_encoded = pd.concat(df_encoded_list, axis=1)
    return df_encoded


def tab_disjonctif_prop(df, seed=None):
    """Perform probabilistic imputation for categorical columns using observed
    value distributions, without creating a separate missing category.

    Parameters
    ----------
    df : DataFrame
        DataFrame with categorical columns to impute.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    DataFrame
        Disjunctive coded DataFrame with missing values probabilistically
        imputed.

    """  # noqa: D205
    if seed is not None:
        np.random.seed(seed)
    df = df.copy()
    df_encoded_list = []
    for col in df.columns:
        if df[col].dtype.name == "category" or df[col].dtype == object:
            # Ensure categories are strings
            df[col] = df[col].cat.rename_categories(
                df[col].cat.categories.astype(str)
            )
            observed = df[col][df[col].notna()]
            categories = df[col].cat.categories.tolist()
            # Get observed frequencies
            freqs = observed.value_counts(normalize=True)
            # Impute missing values based on observed frequencies
            missing_indices = df[col][df[col].isna()].index
            if len(missing_indices) > 0:
                imputed_values = np.random.choice(
                    freqs.index, size=len(missing_indices), p=freqs.values
                )
                df.loc[missing_indices, col] = imputed_values
            # One-hot encode without creating missing category
            encoded = pd.get_dummies(
                df[col],
                prefix=col,
                prefix_sep="_",
                dummy_na=False,
                dtype=float,
            )
            col_names = [f"{col}_{cat}" for cat in categories]
            encoded = encoded.reindex(columns=col_names, fill_value=0.0)
            df_encoded_list.append(encoded)
        else:
            df_encoded_list.append(df[[col]])
    df_encoded = pd.concat(df_encoded_list, axis=1)
    return df_encoded


def find_category(df_original, tab_disj):
    """Reconstruct the original categorical variables from the disjunctive.

    Parameters
    ----------
    df_original : DataFrame
        Original DataFrame with categorical variables.
    tab_disj : DataFrame
        Disjunctive table after imputation.

    Returns
    -------
    DataFrame
        Reconstructed DataFrame with imputed categorical variables.

    """
    df_reconstructed = df_original.copy()
    start_idx = 0
    for col in df_original.columns:
        if (
            df_original[col].dtype.name == "category"
            or df_original[col].dtype == object
        ):  # noqa: E501
            categories = df_original[col].cat.categories.tolist()
            if "__MISSING__" in categories:
                missing_cat_index = categories.index("__MISSING__")
            else:
                missing_cat_index = None
            num_categories = len(categories)
            sub_tab = tab_disj.iloc[:, start_idx : start_idx + num_categories]
            if missing_cat_index is not None:
                sub_tab.iloc[:, missing_cat_index] = -np.inf
            # Find the category with the maximum value for each row
            max_indices = sub_tab.values.argmax(axis=1)
            df_reconstructed[col] = [categories[idx] for idx in max_indices]
            # Replace '__MISSING__' back to NaN
            df_reconstructed[col].replace("__MISSING__", np.nan, inplace=True)
            start_idx += num_categories
        else:
            # For numeric variables, keep as is
            start_idx += 1  # Increment start_idx by 1 for numeric columns
    return df_reconstructed