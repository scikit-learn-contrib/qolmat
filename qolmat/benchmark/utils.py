from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy.optimize import Bounds, lsq_linear
from sklearn.preprocessing import StandardScaler

BOUNDS = Bounds(1, np.inf, keep_feasible=True)
EPS = np.finfo(float).eps


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
