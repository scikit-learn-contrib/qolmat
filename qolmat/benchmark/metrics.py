from typing import Optional, Union, Literal, List

import pandas as pd
import numpy as np
from collections import Counter

import scipy
from sklearn import metrics as skm
from sklearn.preprocessing import StandardScaler

EPS = np.finfo(float).eps

###########################
# Column-wise metris      #
###########################

def mean_squared_error(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> pd.Series:
    """Mean squared error between two dataframes.

    Parameters
    ----------
    df1 : pd.DataFrame
        True dataframe
    df2 : pd.DataFrame
        Predicted dataframe

    Returns
    -------
    pd.Series
    """
    return pd.Series(skm.mean_squared_error(df1, df2, multioutput='raw_values'), index= df1.columns.to_list())


def root_mean_squared_error(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> pd.Series:
    """Root mean squared error between two dataframes.

    Parameters
    ----------
    df1 : pd.DataFrame
        True dataframe
    df2 : pd.DataFrame
        Predicted dataframe

    Returns
    -------
    pd.Series
    """
    return pd.Series(skm.mean_squared_error(df1, df2, squared=False, multioutput='raw_values'), index=df1.columns.to_list())


def mean_absolute_error(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> pd.Series:
    """Mean absolute error between two dataframes.

    Parameters
    ----------
    df1 : pd.DataFrame
        True dataframe
    df2 : pd.DataFrame
        Predicted dataframe

    Returns
    -------
    pd.Series
    """
    return pd.Series(skm.mean_absolute_error(df1, df2, multioutput='raw_values'), index=df1.columns.to_list())


def weighted_mean_absolute_percentage_error(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> pd.Series:
    """Weighted mean absolute percentage error between two dataframes.

    Parameters
    ----------
    Parameters
    ----------
    df1 : pd.DataFrame
        True dataframe
    df2 : pd.DataFrame
        Predicted dataframe

    Returns
    -------
    Union[float, pd.Series]
    """
    return pd.Series((df1 - df2).abs().sum() / df1.abs().sum(), index=df1.columns.to_list())

def wasser_distance(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> pd.Series:
    """Wasserstein distances between columns of 2 dataframes.
    Wasserstein distance can only be computed columnwise

    Parameters
    ----------
    df1 : pd.DataFrame
    df2 : pd.DataFrame

    Returns
    -------
    wasserstein distances : pd.Series
    """
    cols = df1.columns.tolist()
    wd = [scipy.stats.wasserstein_distance(df1[col].dropna(), df2[col].dropna()) for col in cols]
    return pd.Series(wd, index=cols)

def kl_divergence(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    columnwise_evaluation: Optional[bool] = True,
) -> Union[float, pd.Series]:
    """Kullback-Leibler divergence between distributions
    If multivariate normal distributions:
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    Parameters
    ----------
    df1 : pd.DataFrame
    df2 : pd.DataFrame
    columnwise_evaluation: Optional[bool]
        if the evalutation is computed column-wise. By default, is set to False

    Returns
    -------
    Kullback-Leibler divergence : Union[float, pd.Series]
    """
    cols = df1.columns.tolist()
    if columnwise_evaluation or df1.shape[1] == 1:
        list_kl = []
        for col in cols:
            min_val = min(df1[col].min(), df2[col].min())
            max_val = min(df1[col].max(), df2[col].max())
            bins = np.linspace(min_val, max_val, 20)
            p = np.histogram(df1[col].dropna(), bins=bins, density=True)[0]
            q = np.histogram(df2[col].dropna(), bins=bins, density=True)[0]
            list_kl.append(scipy.stats.entropy(p + EPS, q + EPS))
        return pd.Series(list_kl, index=cols)
    else:
        df_1 = StandardScaler().fit_transform(df1)
        df_2 = StandardScaler().fit_transform(df2)

        n = df_1.shape[0]
        mu_true = np.nanmean(df_1, axis=0)
        sigma_true = np.ma.cov(np.ma.masked_invalid(df_1), rowvar=False).data
        mu_pred = np.nanmean(df_2, axis=0)
        sigma_pred = np.ma.cov(np.ma.masked_invalid(df_2), rowvar=False).data
        diff = mu_true - mu_pred
        inv_sigma_pred = np.linalg.inv(sigma_pred)
        quad_term = diff.T @ inv_sigma_pred @ diff
        trace_term = np.trace(inv_sigma_pred @ sigma_true)
        det_term = np.log(np.linalg.det(sigma_pred) / np.linalg.det(sigma_true))
        kl = 0.5 * (quad_term + trace_term + det_term - n)
        return pd.Series(kl, index=cols)

def _get_numerical_features(df1: pd.DataFrame):
    cols_numerical = df1.select_dtypes(include=np.number).columns.tolist()
    if len(cols_numerical) == 0:
        raise Exception("No numerical feature is found.")
    else:
        return cols_numerical


def _get_categorical_features(df1: pd.DataFrame):
    cols_numerical = df1.select_dtypes(include=np.number).columns.tolist()
    cols_categorical = [col for col in df1.columns.to_list() if col not in cols_numerical]
    if len(cols_categorical) == 0:
        raise Exception("No categorical feature is found.")
    else:
        return cols_categorical


def kolmogorov_smirnov_test(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.Series:
    """Kolmogorov Smirnov Test for numerical features. Lower score means better performance.

    Parameters
    ----------
    df1 : pd.DataFrame
        true dataframe
    df2 : pd.DataFrame
        predicted dataframe

    Returns
    -------
    float
        KS test statistic
    """
    cols_numerical = _get_numerical_features(df1)
    ks_test_statistic = [scipy.stats.ks_2samp(df1[col], df2[col])[0] for col in cols_numerical]

    return pd.Series(ks_test_statistic, index=cols_numerical)


def total_variance_distance(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.Series:
    """Total variance distance for categorical features, based on TVComplement in https://github.com/sdv-dev/SDMetrics

    Parameters
    ----------
    df1 : pd.DataFrame
        true dataframe
    df2 : pd.DataFrame
        predicted dataframe

    Returns
    -------
    pd.Series
        Total variance distance
    """

    cols_categorical = _get_categorical_features(df1)
    total_variance_distance = {}
    for col in cols_categorical:
        list_categories = list(set(df1[col].unique() + df2[col].unique()))
        freqs1 = df1.groupby(col).count() / len(df1)
        freqs1 = freqs1.reindex(list_categories, fill_value=0)
        freqs2 = df2.groupby(col).count() / len(df2)
        freqs2 = freqs2.reindex(list_categories, fill_value=0)
        total_variance_distance[col] = (freqs1 - freqs2).abs().sum()

    return pd.Series(total_variance_distance)


def _get_correlation_pearson_matrix(data: pd.DataFrame, use_p_value: bool = True) -> pd.DataFrame:
    cols = data.columns.tolist()
    matrix = np.zeros((len(data.columns), len(data.columns)))
    for idx_1, col_1 in enumerate(cols):
        for idx_2, col_2 in enumerate(cols):
            res = scipy.stats.mstats.pearsonr(
                data[col_1].array.reshape(-1, 1), data[col_2].array.reshape(-1, 1)
            )
            if use_p_value:
                matrix[idx_1, idx_2] = res[1]
            else:
                matrix[idx_1, idx_2] = res[0]
    return pd.DataFrame(matrix, index=cols, columns=cols)


def mean_difference_correlation_matrix_numerical_features(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    use_p_value: bool = True,
) -> pd.Series:
    """_summary_

    Parameters
    ----------
    df1 : pd.DataFrame
        true dataframe
    df2 : pd.DataFrame
        predicted dataframe
    method : _type_
        _description_

    Returns
    -------
    float
        Mean absolute differences between correlation matrix of df1 and df2
    """
    if df1.shape != df2.shape:
        raise Exception("inputs have to be of same dimensions.")
    cols_numerical = _get_numerical_features(df1)
    df_corr1 = _get_correlation_pearson_matrix(df1[cols_numerical], use_p_value=use_p_value)
    df_corr2 = _get_correlation_pearson_matrix(df2[cols_numerical], use_p_value=use_p_value)

    diff_corr =  (df_corr1 - df_corr2).abs().mean(axis=1)
    return pd.Series(diff_corr, index=cols_numerical)


def _get_correlation_chi2_matrix(data: pd.DataFrame, use_p_value: bool = True) -> pd.DataFrame:
    cols = data.columns.tolist()
    matrix = np.zeros((len(data.columns), len(data.columns)))
    for idx_1, col_1 in enumerate(cols):
        for idx_2, col_2 in enumerate(cols):
            freq = data.pivot_table(
                index=col_1, columns=col_2, aggfunc="size", fill_value=0
            ).to_numpy()
            res = scipy.stats.chi2_contingency(freq)
            if use_p_value:
                matrix[idx_1, idx_2] = res[1]
            else:
                matrix[idx_1, idx_2] = res[0]
    return pd.DataFrame(matrix, index=cols, columns=cols)


def mean_difference_correlation_matrix_categorical_features(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    use_p_value: bool = True,
) -> pd.Series:
    """_summary_

    Parameters
    ----------
    df1 : pd.DataFrame
        true dataframe
    df2 : pd.DataFrame
        predicted dataframe
    method : _type_
        _description_

    Returns
    -------
    float
        Mean absolute differences between correlation matrix of df1 and df2
    """
    if df1.shape != df2.shape:
        raise Exception("inputs have to be of same dimensions.")
    cols_categorical = _get_categorical_features(df1)
    df_corr1 = _get_correlation_chi2_matrix(df1[cols_categorical], use_p_value=use_p_value)
    df_corr2 = _get_correlation_chi2_matrix(df2[cols_categorical], use_p_value=use_p_value)

    diff_corr =  (df_corr1 - df_corr2).abs().mean(axis=1)
    return pd.Series(diff_corr, index=cols_categorical)


def _get_correlation_f_oneway_matrix(
    data: pd.DataFrame,
    cols_categorical: List[str],
    cols_numerical: List[str],
    use_p_value: bool = True,
) -> pd.DataFrame:
    matrix = np.zeros((len(cols_categorical), len(cols_numerical)))
    for idx_cat, col_cat in enumerate(cols_categorical):
        for idx_num, col_num in enumerate(cols_numerical):
            category_group_lists = data.groupby(col_cat)[col_num].apply(list)
            try:
                res = scipy.stats.f_oneway(*category_group_lists)
                if use_p_value:
                    matrix[idx_cat, idx_num] = res[1]
                else:
                    matrix[idx_cat, idx_num] = res[0]
            except ValueError:
                matrix[idx_cat, idx_num] = 0.0
    return pd.DataFrame(matrix, index=cols_categorical, columns=cols_numerical)

def mean_difference_correlation_matrix_categorical_vs_numerical_features(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    use_p_value: bool = True,
) -> pd.Series:
    """_summary_

    Parameters
    ----------
    df1 : pd.DataFrame
        true dataframe
    df2 : pd.DataFrame
        predicted dataframe
    method : _type_
        _description_

    Returns
    -------
    float
        Mean absolute differences between correlation matrix of df1 and df2
    """
    if df1.shape != df2.shape:
        raise Exception("inputs have to be of same dimensions.")
    cols_categorical = _get_categorical_features(df1)
    cols_numerical = _get_numerical_features(df1)
    df_corr1 = _get_correlation_f_oneway_matrix(
        df1, cols_categorical, cols_numerical, use_p_value=use_p_value
    )
    df_corr2 = _get_correlation_f_oneway_matrix(
        df2, cols_categorical, cols_numerical, use_p_value=use_p_value
    )

    diff_corr =  (df_corr1 - df_corr2).abs().mean(axis=1)
    return pd.Series(diff_corr, index=cols_categorical)

###########################
# Row-wise metris         #
###########################

def _sum_distance_col(col: pd.Series, col_size: int) -> pd.Series:
    """_summary_

    Parameters
    ----------
    col : pd.Series
        _description_
    col_size : int
        _description_

    Returns
    -------
    _type_
        _description_
    """
    col = col.sort_values(ascending=True)
    sums_partial = col.shift().fillna(0.0).cumsum()
    differences_partial = col * np.arange(col_size) - sums_partial
    res = differences_partial.sum()
    return res


def _sum_manhattan_distances(df1: pd.DataFrame) -> float:
    """Sum Manhattan distances. It is based on https://www.geeksforgeeks.org/sum-manhattan-distances-pairs-points/

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    """
    cols = df1.columns.tolist()
    sum = 0.0
    for col in cols:
        sum += _sum_distance_col(df1[col], len(df1))
    return sum


def sum_energy_distances(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """Sum of energy distances between df1 and df2. It is based on https://dcor.readthedocs.io/en/latest/theory.html#

    Parameters
    ----------
    df1 : pd.DataFrame
        true dataframe
    df2 : pd.DataFrame
        _description_

    Returns
    -------
    _type_
        _description_
    """

    sum_distances_df1 = _sum_manhattan_distances(
        df1
    )  # sum of (len_df1 * (len_df1 - 1) / 2) distances for df1
    sum_distances_df2 = _sum_manhattan_distances(df2)

    df = pd.concat([df1, df2])
    sum_distances_df1_df2 = _sum_manhattan_distances(df)
    sum_distance = 2 * sum_distances_df1_df2 - 4 * sum_distances_df1 - 4 * sum_distances_df2

    return pd.Series(sum_distance, index=["All"])

def sum_pairwise_distances(df1: pd.DataFrame, df2: pd.DataFrame, metric: str = "cityblock"):
    """Sum of pairwise distances based on a predefined metric

    Parameters
    ----------
    df1 : pd.DataFrame
        true dataframe
    df2 : pd.DataFrame
        predicted dataframe
    metric : str, optional
        distance metric, by default 'cityblock'

    Returns
    -------
    _type_
        _description_
    """
    distances = np.sum(scipy.spatial.distance.cdist(df1, df2, metric=metric))

    return pd.Series(distances, index=["All"])

###########################
# Dataframe-wise metris   #
###########################

def frechet_distance(
    df1: pd.DataFrame, df2: pd.DataFrame, normalized: Optional[bool] = False
) -> pd.Series:
    """Compute the Fréchet distance between two dataframes df1 and df2
    frechet_distance = || mu_1 - mu_2 ||_2^2 + Tr(Sigma_1 + Sigma_2 - 2(Sigma_1 . Sigma_2)^(1/2))
    if normalized, df1 and df_ are first scaled by a factor
        (std(df1) + std(df2)) / 2
    and then centered around
        (mean(df1) + mean(df2)) / 2

    Dowson, D. C., and BV666017 Landau. "The Fréchet distance between multivariate normal
    distributions."
    Journal of multivariate analysis 12.3 (1982): 450-455.

    Parameters
    ----------
    df1 : pd.DataFrame
        true dataframe
    df2 : pd.DataFrame
        predicted dataframe
    normalized: Optional[bool]
        if the data has to be normalised. By default, is set to False

    Returns
    -------
    frechet_distance : float
    """

    if df1.shape != df2.shape:
        raise Exception("inputs have to be of same dimensions.")

    df_true = df1.copy()
    df_pred = df2.copy()

    if normalized:
        std = (np.std(df_true) + np.std(df_pred) + EPS) / 2
        mu = (np.nanmean(df_true, axis=0) + np.nanmean(df_pred, axis=0)) / 2
        df_true = (df_true - mu) / std
        df_pred = (df_pred - mu) / std

    mu_true = np.nanmean(df_true, axis=0)
    sigma_true = np.ma.cov(np.ma.masked_invalid(df_true), rowvar=False).data
    mu_pred = np.nanmean(df_pred, axis=0)
    sigma_pred = np.ma.cov(np.ma.masked_invalid(df_pred), rowvar=False).data

    ssdiff = np.sum((mu_true - mu_pred) ** 2.0)
    product = np.array(sigma_true @ sigma_pred)
    if product.ndim < 2:
        product = product.reshape(-1, 1)
    covmean = scipy.linalg.sqrtm(product)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    frechet_dist = ssdiff + np.trace(sigma_true + sigma_pred - 2.0 * covmean)

    if normalized:
        return pd.Series((frechet_dist / df_true.shape[0]), index=["All"])
    else:
        return pd.Series(frechet_dist, index=["All"])