from typing import Dict, List, Optional, Union, Literal

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sparse
from scipy.optimize import Bounds, lsq_linear
from sklearn.preprocessing import StandardScaler
from skopt.space import Categorical, Dimension, Integer, Real
from collections import Counter

# from sdmetrics.single_table import BNLogLikelihood, GMLogLikelihood

BOUNDS = Bounds(1, np.inf, keep_feasible=True)
EPS = np.finfo(float).eps

# def has_given_attribute(tested_model, name_param):
#     has_attribute = hasattr(tested_model, name_param) and (getattr(tested_model, name_param) is not None)

#     if ((name_param[0] == "(") and (name_param[-1] == ")") and ("," in name_param)):
#         name_param_col = eval(name_param)[1]
#         has_attribute = (has_attribute
#         or (hasattr(tested_model, name_param_col) and (getattr(tested_model, name_param_col) is not None))
#         )
#     return has_attribute


def get_dimension(dict_bounds: Dict, name_dimension: str) -> Dimension:
    if dict_bounds["type"] == "Integer":
        return Integer(low=dict_bounds["min"], high=dict_bounds["max"], name=name_dimension)
    elif dict_bounds["type"] == "Real":
        return Real(low=dict_bounds["min"], high=dict_bounds["max"], name=name_dimension)
    elif dict_bounds["type"] == "Categorical":
        return Categorical(categories=dict_bounds["categories"], name=name_dimension)


def get_search_space(search_params: Dict) -> List[Dimension]:
    """Construct the search space for the tested_model
    based on the search_params

    Parameters
    ----------
    search_params : Dict

    Returns
    -------
    List[Dimension]
        search space

    """
    list_spaces = []

    for name_hyperparam, value in search_params.items():
        # space common for all columns
        if "type" in value:
            list_spaces.append(get_dimension(value, name_hyperparam))
        else:
            for col, dict_bounds in value.items():
                name = f"{name_hyperparam}/{col}"
                list_spaces.append(get_dimension(dict_bounds, name))

    return list_spaces


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

    # put index as columns
    df_out = df.reset_index().copy()
    df_out.index = df.index
    if len(groups) > 0:
        return df.groupby(groups, group_keys=False)
    else:
        return df


######################
# Evaluation metrics #
######################


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
    return ((df1 - df2) ** 2).mean(axis=0)


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
    mse = mean_squared_error(df1, df2)
    return mse.pow(0.5)


def mean_absolute_error(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.Series:
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
        return frechet_dist / df_true.shape[0]
    else:
        return frechet_dist


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
    df1: pd.DataFrame, df2: pd.DataFrame, method: Literal["pearson"], use_p_value: bool = True
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
    match method:
        case "pearson":
            if len(numerical_cols) != 0:
                corr_df1 = get_correlation_pearson_matrix(
                    df1[numerical_cols], use_p_value=use_p_value
                )
                corr_df2 = get_correlation_pearson_matrix(
                    df2[numerical_cols], use_p_value=use_p_value
                )

                return pd.Series(
                    np.mean(np.abs(corr_df1 - corr_df2), axis=1), index=numerical_cols
                )
            else:
                raise Exception("No numerical feature is found.")

        case "chi2":
            if len(categorical_cols) != 0:
                corr_df1 = get_correlation_chi2_matrix(
                    df1[categorical_cols], use_p_value=use_p_value
                )
                corr_df2 = get_correlation_chi2_matrix(
                    df2[categorical_cols], use_p_value=use_p_value
                )

                return pd.Series(
                    np.mean(np.abs(corr_df1 - corr_df2), axis=1), index=categorical_cols
                )
            else:
                raise Exception("No categorical feature is found.")

        case "f_oneway":
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


def coverage_ratio(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.Series:
    """_summary_

    Parameters
    ----------
    df1 : pd.DataFrame
        _description_
    df2 : pd.DataFrame
        _description_

    Returns
    -------
    float
        _description_
    """

    numerical_cols = df1.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [col for col in df1.columns.to_list() if col not in numerical_cols]

    coverage_cols = {}
    for col in numerical_cols:
        min_r = df1[col].min()
        max_r = df1[col].max()
        min_s = df2[col].min()
        max_s = df2[col].max()
        if max_r - min_r == 0:
            coverage_cols[col] = 0
        else:
            normalized_min = max((min_s - min_r) / (max_r - min_r), 0)
            normalized_max = max((max_r - max_s) / (max_r - min_r), 0)
            coverage_cols[col] = max(1 - (normalized_min + normalized_max), 0)

    for col in categorical_cols:
        real_data_value = df1[col].nunique()
        synthetic_data_value = df2[col].nunique()
        coverage_cols[col] = synthetic_data_value / real_data_value

    return pd.Series(coverage_cols)


# def likelihood(df1: pd.DataFrame, df2: pd.DataFrame, retries: int = 5) -> pd.Series:
#     """_summary_

#     Parameters
#     ----------
#     df1 : pd.DataFrame
#         _description_
#     df2 : pd.DataFrame
#         _description_
#     retries : int, optional
#         _description_, by default 5

#     Returns
#     -------
#     float
#         _description_
#     """

#     categorical_cols = [col for col in df1.columns.to_list() if 'category' in df1.dtypes[col].name]
#     numerical_cols = [col for col in df1.columns.to_list() if col not in categorical_cols]

#     likelihoods = {}
#     for col in numerical_cols:
#         real_data_col= df1[[col]]
#         synthetic_data_col = df2[[col]]

#         metadata = {'fields':{col: {'type':'numerical', 'subtype':'float'}}}
#         try:
#             likelihoods[col] =  GMLogLikelihood.compute(real_data_col, synthetic_data_col, metadata=metadata, retries=retries)
#         except:
#             likelihoods[col] = np.nan

#     for col in categorical_cols:
#         real_data_col= df1[[col]]
#         synthetic_data_col = df2[[col]]

#         metadata = {'fields':{col: {'type':'categorical'}}}
#         try:
#             likelihoods[col] =  BNLogLikelihood.compute(real_data_col, synthetic_data_col, metadata=metadata)
#         except:
#             likelihoods[col] = np.nan

#     return pd.Series(likelihoods)

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
    df_aggregated = df.groupby("day_SNCF").apply(lambda x: agg_df_values(x, target, agg_time))
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
        (ts_agg.agg_time >= df_day.datetime.min()) & (ts_agg.agg_time <= df_day.datetime.max())
    ]
    if len(ts_agg) < 2:
        df_day = pd.DataFrame({"datetime": df_day.datetime.values})
        df_day["impute"] = np.nan
        df_res = df.merge(df_day[["datetime", "impute"]], on="datetime", how="left")
        return df_res

    df_day["datetime_round"] = df_day.datetime.dt.round(agg_time)
    df_day["n_train"] = df_day.groupby("datetime_round")[target].transform(lambda x: x.shape[0])

    df_day["hyp_values"] = (
        df_day[["datetime_round"]]
        .merge(ts_agg, left_on="datetime_round", right_on="agg_time", how="left")[col_name]
        .values
    )

    df_day["hyp_values"] = df_day["hyp_values"] / df_day["n_train"]
    df_day.loc[df_day[target].notna(), "hyp_values"] = df_day.loc[df_day[target].notna(), target]
    ts_agg_zeros = ts_agg.loc[ts_agg[col_name] <= zero_soil, "agg_time"]
    is_in_zero_slot = is_in_a_slot(df_dt=df_day["datetime"], df_dt_agg=ts_agg_zeros, freq=agg_time)

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
