from typing import Optional, Union, Literal, List

import pandas as pd
import numpy as np
from collections import Counter

import scipy
from sklearn import metrics as skm
from sklearn.preprocessing import StandardScaler
import scipy.spatial as spatial

EPS = np.finfo(float).eps


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
    return skm.mean_squared_error(df1, df2)


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
    return skm.mean_squared_error(df1, df2, squared=False)


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
    return skm.mean_absolute_error(df1, df2)


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
    return (df1 - df2).abs().mean(axis=0) / df1.abs().mean(axis=0)


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


def frechet_distance(
    df1: pd.DataFrame, df2: pd.DataFrame, normalized: Optional[bool] = False
) -> float:
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
        return pd.Series(np.repeat(frechet_dist / df_true.shape[0], len(df1.columns)))
    else:
        return pd.Series(np.repeat(frechet_dist, len(df1.columns)))


def kolmogorov_smirnov_test(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.Series:
    """Kolmogorov Smirnov Test for numerical features

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
    numerical_cols = df1.select_dtypes(include=np.number).columns.tolist()
    if len(numerical_cols) == 0:
        raise Exception("No numerical feature is found.")

    cols = df1.columns.tolist()
    ks_test_statistic = [scipy.stats.ks_2samp(df1[col], df2[col])[0] for col in cols]

    return pd.Series(ks_test_statistic, index=cols)


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

    numerical_cols = df1.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [col for col in df1.columns.to_list() if col not in numerical_cols]
    if len(categorical_cols) == 0:
        raise Exception("No categorical feature is found.")

    total_variance_distance = {}
    for col in categorical_cols:
        f_df1 = []
        f_df2 = []
        cats_df1 = Counter(df1[col])
        cats_df2 = Counter(df2[col])

        for value in cats_df2:
            if value not in cats_df1:
                cats_df1[value] = 1e-6

        for value in cats_df1:
            f_df1.append(cats_df1[value] / sum(cats_df1.values()))
            f_df2.append(cats_df2[value] / sum(cats_df2.values()))

        score = 0
        for i in range(len(f_df1)):
            score += abs(f_df1[i] - f_df2[i])

        total_variance_distance[col] = score

    return pd.Series(total_variance_distance)


def get_correlation_pearson_matrix(data: pd.DataFrame, use_p_value: bool = True):
    corr = np.zeros((len(data.columns), len(data.columns)))
    for idx_1, col_1 in enumerate(data.columns):
        for idx_2, col_2 in enumerate(data.columns):
            res = scipy.stats.mstats.pearsonr(
                data[col_1].array.reshape(-1, 1), data[col_2].array.reshape(-1, 1)
            )
            if use_p_value:
                corr[idx_1, idx_2] = res[1]
            else:
                corr[idx_1, idx_2] = res[0]
    return corr


def get_correlation_chi2_matrix(data: pd.DataFrame, use_p_value: bool = True):
    corr = np.zeros((len(data.columns), len(data.columns)))
    for idx_1, col_1 in enumerate(data.columns):
        for idx_2, col_2 in enumerate(data.columns):
            freq = data.pivot_table(
                index=col_1, columns=col_2, aggfunc="size", fill_value=0
            ).to_numpy()
            res = scipy.stats.chi2_contingency(freq)
            if use_p_value:
                corr[idx_1, idx_2] = res[1]
            else:
                corr[idx_1, idx_2] = res[0]
    return corr


def get_correlation_f_oneway_matrix(
    data: pd.DataFrame,
    categorical_cols: List[str],
    numerical_cols: List[str],
    use_p_value: bool = True,
):

    corr = np.zeros((len(categorical_cols), len(numerical_cols)))
    for idx_cat, col_cat in enumerate(categorical_cols):
        for idx_num, col_num in enumerate(numerical_cols):
            category_group_lists = data.groupby(col_cat)[col_num].apply(list)
            try:
                res = scipy.stats.f_oneway(*category_group_lists)
                if use_p_value:
                    corr[idx_cat, idx_num] = 0.0 if np.isnan(res[1]) else res[1]
                else:
                    corr[idx_cat, idx_num] = 0.0 if np.isnan(res[1]) else res[0]
            except ValueError:
                corr[idx_cat, idx_num] = 0.0
    return corr


def mean_difference_correlation_matrix(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    method: Literal["pearson", "chi2", "f_oneway"],
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
    numerical_cols = df1.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [col for col in df1.columns.to_list() if col not in numerical_cols]

    if df1.shape != df2.shape:
        raise Exception("inputs have to be of same dimensions.")
    if method == "pearson":
        if len(numerical_cols) != 0:
            corr_df1 = get_correlation_pearson_matrix(df1[numerical_cols], use_p_value=use_p_value)
            corr_df2 = get_correlation_pearson_matrix(df2[numerical_cols], use_p_value=use_p_value)

            return pd.Series(np.mean(np.abs(corr_df1 - corr_df2), axis=1), index=numerical_cols)
        else:
            raise Exception("No numerical feature is found.")

    elif method == "chi2":
        if len(categorical_cols) != 0:
            corr_df1 = get_correlation_chi2_matrix(df1[categorical_cols], use_p_value=use_p_value)
            corr_df2 = get_correlation_chi2_matrix(df2[categorical_cols], use_p_value=use_p_value)

            return pd.Series(np.mean(np.abs(corr_df1 - corr_df2), axis=1), index=categorical_cols)
        else:
            raise Exception("No categorical feature is found.")

    elif method == "f_oneway":
        if len(numerical_cols) != 0 and len(categorical_cols) != 0:
            corr_df1 = get_correlation_f_oneway_matrix(
                df1, categorical_cols, numerical_cols, use_p_value=use_p_value
            )
            corr_df2 = get_correlation_f_oneway_matrix(
                df2, categorical_cols, numerical_cols, use_p_value=use_p_value
            )

            corr_diff = np.abs(corr_df1 - corr_df2)

            corr_diff_dict = {}
            for c, v in zip(categorical_cols, np.mean(corr_diff, axis=1)):
                corr_diff_dict[c] = v
            for c, v in zip(numerical_cols, np.mean(corr_diff, axis=0)):
                corr_diff_dict[c] = v

            return pd.Series(corr_diff_dict)
        else:
            raise Exception("No numerical/categorical feature is found.")
    else:
        raise Exception("Method is not found. Our methods are [pearson, chi2, f_oneway].")


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
    distances = spatial.distance.cdist(df1, df2, metric=metric)

    return np.sum(distances)


def sum_manhattan_distances(df: pd.DataFrame):
    """Sum Manhattan distances. It is based on https://www.geeksforgeeks.org/sum-manhattan-distances-pairs-points/

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    """

    def _sum_distance_col(col: np.array, col_size: int):
        col.sort()

        res = 0.0
        sum = 0.0
        for i in range(col_size):
            res += col[i] * i - sum
            sum += col[i]
        return res

    cols = df.columns.tolist()
    size = len(df)
    sum = 0.0
    for col in cols:
        sum += _sum_distance_col(df[col].to_numpy(), size)
    return sum


def sum_energy_distances(df1: pd.DataFrame, df2: pd.DataFrame, alpha: int = 1):
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

    df = pd.concat([df1, df2])
    sum_distances_df1 = sum_manhattan_distances(
        df1
    )  # sum of (len_df1 * (len_df1 - 1) / 2) distances for df1
    sum_distances_df2 = sum_manhattan_distances(df2)
    sum_distances_df1_df2 = (
        sum_manhattan_distances(df) - sum_distances_df1 - sum_distances_df2
    )  # sum of (len_df1 * len_df2) distances between df1 and df2
    return 2 * sum_distances_df1_df2 - sum_distances_df1 - sum_distances_df2
