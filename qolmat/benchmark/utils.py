from typing import Dict, List, Optional, Tuple, Union
from skopt.space import Categorical, Real, Integer
import pandas as pd
import numpy as np
from sklearn.utils import resample
from math import floor
import scipy
from scipy.optimize import lsq_linear, Bounds
import scipy.sparse as sparse
from sklearn.preprocessing import StandardScaler
from . import missing_patterns

BOUNDS = Bounds(1, np.inf, keep_feasible=True)
EPS = np.finfo(float).eps


def get_search_space(tested_model: any, search_params: Dict) -> Union[None, List]:
    """Construct the search space for the tested_model
    based on the search_params

    Parameters
    ----------
    tested_model : any
        imputation model
    search_params : Dict

    Returns
    -------
    Union[None, List]
        search space

    """
    search_space = None
    if str(type(tested_model).__name__) in search_params.keys():
        search_space = []
        for name_param, vals_params in search_params[
            str(type(tested_model).__name__)
        ].items():

            if str(type(tested_model).__name__) == "ImputeRPCA":
                if hasattr(tested_model.rpca, name_param):
                    raise ValueError(
                        f"Sorry, you set a value to {name_param} and asked for a search..."
                    )
            elif hasattr(tested_model, name_param):
                raise ValueError(
                    f"Sorry, you set a value to {name_param} and asked for a search..."
                )

            if vals_params["type"] == "Integer":
                search_space.append(
                    Integer(
                        low=vals_params["min"], high=vals_params["max"], name=name_param
                    )
                )
            elif vals_params["type"] == "Real":
                search_space.append(
                    Real(
                        low=vals_params["min"], high=vals_params["max"], name=name_param
                    )
                )
            elif vals_params["type"] == "Categorical":
                search_space.append(
                    Categorical(categories=vals_params["categories"], name=name_param)
                )

    return search_space


def custom_groupby(
    df: pd.DataFrame, groups: List[str]
) -> Union[pd.DataFrame, pd.core.groupby.DataFrameGroupBy]:
    """Groupby on dataframe

    Parameters
    ----------
    df : pd.DataFrame
    groups : List[str]
        list of columns for grouping
    Returns
    -------
    Union[pd.DataFrame, pd.core.groupby.DataFrameGroupBy]
        initial dataframe or initial dataframe group by the specified groups
    """
    if len(groups) > 0:
        # groupby = []
        # for g in groups:
        # groupby.append(eval("df." + g))
        # groupby.append(df[g])
        return df.groupby(groups)
    else:
        return df


def choice_with_mask(
    df: pd.DataFrame,
    mask: pd.DataFrame,
    ratio: float,
    filter_value: Optional[float] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Create missing values in a dataframe

    Parameters
    ----------
    df : pd.DataFrame
        initial dataframe with missing values
    mask : pd.DataFrame
        mask of nan values of the initial dataframe df
    ratio : float
        ratio of missing values to be created
    filter_value : Optional[float], optional
        values above this filter_value are also considerd as nan, by default None
    random_state : Optional[int], optional
        random state for replicability, by default None

    Returns
    -------
    pd.DataFrame
        initial dataframe with additional missing values
    """

    mask = mask.to_numpy().flatten()
    if filter_value:
        mask_filter = (df.values > filter_value).flatten()
        mask += mask_filter

    indices = np.argwhere(mask > 0)[:, 0]
    indices = resample(
        indices,
        replace=False,
        n_samples=floor(len(indices) * ratio),
        random_state=random_state,
        stratify=None,
    )

    choosen = np.full(df.shape, False, dtype=bool)
    choosen.flat[indices] = True
    return pd.DataFrame(
        choosen.reshape(df.shape), index=df.index, columns=df.columns, dtype=bool
    )


def create_missing_values(
    df: pd.DataFrame,
    cols_to_impute: List[str],
    markov: Optional[bool] = True,
    ratio_missing: Optional[float] = 0.1,
    missing_mechanism: Optional[str] = "MCAR",
    opt: Optional[str] = "selfmasked",
    p_obs: Optional[float] = 0.1,
    quantile: Optional[float] = 0.3,
    corruption: Optional[str] = "missing",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create missing values in a dataframe

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to be corrupted
    cols_to_impute : List[str],
    markov : Optional[bool] = True,
    ratio_missing : Optional[float] = 0.1,
    missing_mechanism : Optional[str] = "MCAR",
    opt : Optional[str] = "selfmasked",
    p_obs : Optional[float] = 0.1,
    quantile : Optional[float] = 0.3,
    corruption : Optional[str] = "missing",
    """

    df_corrupted_select = df[cols_to_impute].copy()

    if markov:
        res = missing_patterns.produce_NA_markov_chain(
            df_corrupted_select, columnwise_missing=False
        )

    else:
        res = missing_patterns.produce_NA_mechanism(
            df_corrupted_select,
            ratio_missing,
            mecha=missing_mechanism,
            opt=opt,
            p_obs=p_obs,
            q=quantile,
        )

    df_is_altered = res["mask"]
    if corruption == "missing":
        df_corrupted_select[df_is_altered] = np.nan
    elif corruption == "outlier":
        df_corrupted_select[df_is_altered] = np.random.randint(
            0, high=3 * np.max(df), size=(int(len(df) * ratio_missing))
        )

    df_corrupted = df.copy()
    df_corrupted[cols_to_impute] = df_corrupted_select

    return (df_is_altered, df_corrupted)


######################
# Evaluation metrics #
######################


def mean_squared_error(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    squared: Optional[bool] = True,
    columnwise_evaluation: Optional[bool] = False,
) -> Union[float, pd.Series]:
    """Mean squared error between two dataframes.

    Parameters
    ----------
    df1 : pd.DataFrame
        True dataframe
    df2 : pd.DataFrame
        Predicted dataframe
    squared : Optional[bool], optional
        wheter returns MSE or RMSE, by default True
    columnwise_evaluation : Optional[bool], optional
        whether the metric should be calculated column-wise or not, by default False

    Returns
    -------
    Union[float, pd.Series]
    """
    if columnwise_evaluation:
        squared_errors = ((df1 - df2) ** 2).sum()
    else:
        squared_errors = ((df1 - df2) ** 2).sum().sum()
    if squared:
        return squared_errors
    else:
        return np.sqrt(squared_errors)


def mean_absolute_error(
    df1: pd.DataFrame, df2: pd.DataFrame, columnwise_evaluation: Optional[bool] = False
) -> Union[float, pd.Series]:
    """Mean absolute error between two dataframes.

    Parameters
    ----------
    df1 : pd.DataFrame
        True dataframe
    df2 : pd.DataFrame
        Predicted dataframe
    columnwise_evaluation : Optional[bool], optional
        whether the metric should be calculated column-wise or not, by default False

    Returns
    -------
    Union[float, pd.Series]
    """
    if columnwise_evaluation:
        return (df1 - df2).abs().sum()
    else:
        return (df1 - df2).abs().sum().sum()


def weighted_mean_absolute_percentage_error(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    columnwise_evaluation: Optional[bool] = False,
) -> Union[float, pd.Series]:
    """Weighted mean absolute percentage error between two dataframes.

    Parameters
    ----------
    Parameters
    ----------
    df1 : pd.DataFrame
        True dataframe
    df2 : pd.DataFrame
        Predicted dataframe
    columnwise_evaluation : Optional[bool], optional
        whether the metric should be calculated column-wise or not, by default False

    Returns
    -------
    Union[float, pd.Series]
    """
    if columnwise_evaluation:
        return (df1 - df2).abs().mean() / df1.abs().mean()
    else:
        return ((df1 - df2).abs().mean() / df1.abs().mean()).mean()


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
    wd = [
        scipy.stats.wasserstein_distance(df1[col].dropna(), df2[col].ffill().bfill())
        for col in cols
    ]
    return pd.Series(wd, index=cols)


def kl_divergence(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    columnwise_evaluation: Optional[bool] = False,
) -> Union[float, pd.Series]:
    """Kullback-Leibler divergence between distributions
    If multivariate normal distributions: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

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

    if columnwise_evaluation or df1.shape[1] == 1:
        cols = df1.columns.tolist()
        kl = []
        for col in cols:
            p = np.histogram(df1[col].dropna(), bins=20, density=True)[0]
            q = np.histogram(df2[col].dropna(), bins=20, density=True)[0]
            kl.append(scipy.stats.entropy(p + EPS, q + EPS))
        return pd.Series(kl, index=cols)
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
        return 0.5 * (quad_term + trace_term + det_term - n)


def frechet_distance(
    df1: pd.DataFrame, df2: pd.DataFrame, normalized: Optional[bool] = False
) -> float:
    """Compute the Fréchet distance between two dataframes df1 and df2
        frechet_distance = || mu_1 - mu_2 ||_2^2 + Tr(Sigma_1 + Sigma_2 - 2(Sigma_1 . Sigma_2)^(1/2))
    if normalized, df1 and df_ are first scaled by a factor
        (std(df1) + std(df2)) / 2
    and then centered around
        (mean(df1) + mean(df2)) / 2

    Dowson, D. C., and BV666017 Landau. "The Fréchet distance between multivariate normal distributions."
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
        return frechet_dist / df_true.shape[0]
    else:
        return frechet_dist


###########################
# Aggregation and entropy #
###########################


def get_agg_matrix(x, y, freq):
    delta = pd.Timedelta(freq)
    x = x.reshape(-1, 1)
    x = np.tile(x, (1, len(y)))
    y = np.tile(y, (len(x), 1))

    increasing = 1 + ((y - x) / delta)
    increasing = np.logical_and(x - delta <= y, y < x) * increasing
    decreasing = 1 + ((x - y) / delta)
    decreasing[-1, :] = 1
    decreasing = np.logical_and(x <= y, y < x + delta) * decreasing

    A = sparse.csc_matrix(decreasing + increasing)
    return A


def agg_df_values(df, target, agg_time):
    df_dt = df["datetime"]
    df_dt_agg = np.sort(df["datetime"].dt.floor(agg_time).unique())
    A = get_agg_matrix(x=df_dt_agg, y=df_dt, freq=agg_time)
    df_values_agg = A.dot(df[target].values)
    df_res = pd.DataFrame({"agg_time": df_dt_agg, "agg_values": df_values_agg})
    return df_res


def aggregate_time_data(df, target, agg_time):
    df.loc[:, "datetime"] = df.datetime.dt.tz_localize(tz=None)
    df["day_SNCF"] = (df["datetime"] - pd.Timedelta("4H")).dt.date
    df_aggregated = df.groupby("day_SNCF").apply(
        lambda x: agg_df_values(x, target, agg_time)
    )
    return df_aggregated


def cross_entropy(t, t_hyp):
    loss = np.sum(t * np.log(t / t_hyp))
    jac = np.log(t / t_hyp) - 1
    return loss, jac


def hessian(t):
    return np.diag(1 / t)


def impute_by_max_entropy(
    df_dt,
    df_dt_agg,
    df_values_agg,
    freq,
    df_values_hyp,
):
    A = get_agg_matrix(x=df_dt_agg, y=df_dt, freq=freq)
    np.random.seed(42)
    res = lsq_linear(A, b=df_values_agg, bounds=(1, np.inf), method="trf", tol=1e-10)
    df_res = pd.DataFrame({"datetime": df_dt, "impute": res.x})
    return df_res


def is_in_a_slot(df_dt, df_dt_agg, freq):
    delta = pd.Timedelta(freq)
    df_dt_agg = df_dt_agg.values.reshape(-1, 1)
    df_dt_agg = np.tile(df_dt_agg, (1, len(df_dt)))
    df_dt = df_dt.values
    df_dt = np.tile(df_dt, (len(df_dt_agg), 1))

    lower_bound = df_dt >= df_dt_agg - delta
    upper_bound = df_dt < df_dt_agg + delta

    in_any_slots = np.logical_and(lower_bound, upper_bound)
    in_any_slots = np.any(in_any_slots, axis=0)
    return in_any_slots


def impute_entropy_day(df, target, ts_agg, agg_time, zero_soil=0.0):
    col_name = ts_agg.name

    df_day = df.drop_duplicates(subset=["datetime"])
    ts_agg = ts_agg.to_frame().reset_index()
    ts_agg = ts_agg.loc[
        (ts_agg.agg_time >= df_day.datetime.min())
        & (ts_agg.agg_time <= df_day.datetime.max())
    ]
    if len(ts_agg) < 2:
        df_day = pd.DataFrame({"datetime": df_day.datetime.values})
        df_day["impute"] = np.nan
        df_res = df.merge(df_day[["datetime", "impute"]], on="datetime", how="left")
        return df_res

    df_day["datetime_round"] = df_day.datetime.dt.round(agg_time)
    df_day["n_train"] = df_day.groupby("datetime_round")[target].transform(
        lambda x: x.shape[0]
    )

    df_day["hyp_values"] = (
        df_day[["datetime_round"]]
        .merge(ts_agg, left_on="datetime_round", right_on="agg_time", how="left")[
            col_name
        ]
        .values
    )

    df_day["hyp_values"] = df_day["hyp_values"] / df_day["n_train"]
    df_day.loc[df_day[target].notna(), "hyp_values"] = df_day.loc[
        df_day[target].notna(), target
    ]
    ts_agg_zeros = ts_agg.loc[ts_agg[col_name] <= zero_soil, "agg_time"]
    is_in_zero_slot = is_in_a_slot(
        df_dt=df_day["datetime"], df_dt_agg=ts_agg_zeros, freq=agg_time
    )

    df_day["impute"] = np.nan
    df_day.loc[is_in_zero_slot, "impute"] = 0

    non_zero_impute = impute_by_max_entropy(
        df_dt=df_day.loc[~is_in_zero_slot, "datetime"].values,
        df_dt_agg=ts_agg.loc[ts_agg[col_name] > zero_soil, "agg_time"].values,
        df_values_agg=ts_agg.loc[ts_agg[col_name] > zero_soil, col_name].values,
        freq=agg_time,
        df_values_hyp=df_day.loc[~is_in_zero_slot, "hyp_values"].values,
    )

    df_day.loc[~is_in_zero_slot, "impute"] = (
        df_day.loc[~is_in_zero_slot, ["datetime"]]
        .merge(non_zero_impute, on="datetime", how="left")["impute"]
        .values
    )

    df_res = df.merge(df_day[["datetime", "impute"]], on="datetime", how="left")

    return df_res
