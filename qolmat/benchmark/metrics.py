from typing import Optional, Union

import pandas as pd
import numpy as np

import scipy
from sklearn import metrics as skm
from sklearn.preprocessing import StandardScaler

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
    return (df1 - df2).abs().mean(axis=0)
    # return pd.Series(skm.mean_absolute_error(df1, df2, multioutput='raw_values'), index=df1.columns.to_list())


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
