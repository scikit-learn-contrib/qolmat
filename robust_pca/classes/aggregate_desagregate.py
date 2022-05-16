from contextlib import _AsyncGeneratorContextManager
from operator import is_
import numpy as np
import pandas as pd
from scipy.optimize import minimize, lsq_linear
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds
import scipy.sparse as sparse

def get_agg_matrix(target, source, delta):
    target = target.reshape(-1, 1)
    target = np.tile(target, (1, len(source)))
    source = np.tile(source, (len(target), 1))

    increasing = (source - target) / delta
    increasing[0,:] = 1
    increasing[-1,:] = 1
    increasing = np.logical_and(target <= source, source < target + delta) * increasing
    decreasing = 1 + ((target - source) / delta)
    decreasing[0, :] = 0
    decreasing[-1, :] = 0
    decreasing = np.logical_and(target <= source, source < target + delta) * decreasing
    A = sparse.csc_array(decreasing + increasing)
    return A

def agg_df_values(df, target, agg_time, weighted = False):
    df_dt = df["datetime"].values
    df_dt_agg = pd.date_range(start=df["datetime"].dt.floor(agg_time).min(),
                              end=df["datetime"].dt.floor(agg_time).max(),
                              freq=agg_time,
                              tz=None,
                              normalize=False,
                              inclusive="both").values
    A = get_agg_matrix(target=df_dt_agg, source=df_dt, delta = pd.Timedelta(agg_time))
    if weighted:
        w_num = A.dot(np.ones(len(df_dt)))
        w_denom = A.dot((~np.isnan(df[target].values)))
        weight = w_num / np.maximum(w_denom, 1)
        weight = np.where(w_denom == 0.0, np.nan, weight)
        weight = np.where(w_num == 0.0, 0.0, weight)
        df_values_agg = weight * A.dot(df[target].fillna(0).values)
    else:
        df_values_agg = A.dot(df[target].values)
    df_res = pd.DataFrame({"agg_time": df_dt_agg, "agg_values": df_values_agg})
    return df_res


def cross_entropy(t, t_hyp):
    loss = np.sum((t + 1) * np.log((t + 1) / (t_hyp + 1)))
    jac = np.log((t + 1) / (t_hyp+1)) - 1
    return loss, jac


def impute_by_max_entropy(
    df_dt,
    df_dt_agg,
    df_values_agg,
    df_values_hyp,
    freq,
    norm,
):

    A = get_agg_matrix(target=df_dt_agg, source=df_dt, delta = pd.Timedelta(freq)
)
    if norm == "L2":
        res = lsq_linear(
            A, b=df_values_agg, bounds=(0, np.inf), method="trf", tol=1e-10
        )
    elif norm == "entropy":
        # constr = LinearConstraint(A, df_values_agg, df_values_agg, keep_feasible=False)
        # bounds = tuple([(1, None)] * len(x0))
        constr = {"type": "ineq", "fun": lambda x: A.dot(x) - df_values_agg}
        x0 = np.random.uniform(low=0.8, high=1.0, size=len(df_values_hyp))
        x0 = x0 * df_values_hyp
        bounds = tuple([(0, None)] * len(x0))
        res = minimize(
            fun=lambda x: cross_entropy(x, df_values_hyp),
            jac=True,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constr,
            tol=1e-10,
            options={"disp": False},
        )
    df_res = pd.DataFrame({"datetime": df_dt, "impute": res.x})
    return df_res


def is_in_a_slot(is_in, ref, freq):
    delta = pd.Timedelta(freq)
    ref = ref.reshape(-1, 1)
    ref = np.tile(ref, (1, len(is_in)))
    is_in = np.tile(is_in, (len(ref), 1))
    lower_bound = is_in > ref - delta
    upper_bound = is_in < ref + delta

    in_any_slots = np.logical_and(lower_bound, upper_bound)
    in_any_slots = np.any(in_any_slots, axis=0)
    return in_any_slots


def impute_entropy_day(
    df, target, ts_agg, agg_time, zero_soil=0.0, fill="non_zeros", norm="L2"
):
    df_day = df.drop_duplicates(subset=["datetime"])
    ts_agg = ts_agg.loc[
        (ts_agg.agg_time >= df_day.datetime.min())
        & (ts_agg.agg_time <= df_day.datetime.max())
    ]
    if len(ts_agg) == 0:
        df_day = pd.DataFrame(
            {"datetime": df_day.datetime.values, "impute": df_day[target].values}
        )
        df_res = df.merge(df_day[["datetime", "impute"]], on="datetime", how="left")
        return df_res

    df_day["datetime_round"] = df_day.datetime.dt.round(agg_time)
    df_day["n_train"] = df_day.groupby("datetime_round")[target].transform(
        lambda x: x.shape[0]
    )

    df_day["hyp_values"] = (
        df_day[["datetime_round"]]
        .merge(ts_agg, left_on="datetime_round", right_on="agg_time", how="left")[
            "impute_agg_values"
        ]
        .values
    )
    df_day["hyp_values"] = df_day["hyp_values"] / df_day["n_train"]
    df_day.loc[df_day[target].notna(), "hyp_values"] = df_day.loc[
        df_day[target].notna(), target
    ]

    df_day["hyp_values"] = 1.0

    if fill == "non_zeros":
        matrix_agg = get_agg_matrix(target=ts_agg["agg_time"].values, source = df_day["datetime"].values, delta = pd.Timedelta(agg_time))
        df_values_known = df_day[target].fillna(0)
        ts_agg_known = matrix_agg.dot(df_values_known)
        ts_agg_residual = np.maximum(ts_agg["impute_agg_values"].values - ts_agg_known, 0)
        ts_agg_non_zeros = ts_agg.loc[ts_agg_residual > zero_soil, :]
        
        not_in_zero_slot = is_in_a_slot(
            is_in=df_day["datetime"].values, ref=ts_agg_non_zeros["agg_time"].values, freq=agg_time
        )
        df_dt = df_day.loc[not_in_zero_slot, "datetime"].values
        df_values_hyp = df_day.loc[not_in_zero_slot, "hyp_values"].values
        df_dt_agg = ts_agg_non_zeros["agg_time"].values
        df_values_agg = ts_agg_non_zeros["impute_agg_values"].values

    elif fill == "nan":
        df_dt = df_day.loc[df_day[target].isna(), "datetime"].values
        is_in_nan_slot_agg = is_in_a_slot(
            is_in=ts_agg["agg_time"].values, ref=df_dt, freq=agg_time
        )
        df_dt_agg = ts_agg.loc[is_in_nan_slot_agg, "agg_time"].values
        df_values_agg = ts_agg.loc[is_in_nan_slot_agg, "impute_agg_values"].values

        is_in_nan_slot_dt = is_in_a_slot(
            is_in=df_day["datetime"].values, ref=df_dt_agg, freq=agg_time
        )
        df_dt = df_day.loc[is_in_nan_slot_dt, "datetime"].values
        df_values_hyp = df_day.loc[is_in_nan_slot_dt, "hyp_values"].values

    elif fill == "all":
        df_dt = df_day["datetime"].values
        df_dt_agg = ts_agg["agg_time"].values
        df_values_agg = ts_agg["impute_agg_values"].values
        df_values_hyp = df_day["hyp_values"].values

    if (len(df_dt) > 0) and (len(df_dt_agg) > 0):
        non_zero_impute = impute_by_max_entropy(
            df_dt=df_dt,
            df_dt_agg=df_dt_agg,
            df_values_agg=df_values_agg,
            df_values_hyp=df_values_hyp,
            freq=agg_time,
            norm=norm,
        )
        df_day = df_day.merge(non_zero_impute, on="datetime", how="left")
    else:
        df_day["impute"] = np.nan
    if fill == "non_zeros":
        df_day["impute"] = df_day["impute"].fillna(0)
    elif fill == "nan":
        df_day["impute"] = df_day["impute"].fillna(df_day[target])

    df_res = df.merge(df_day[["datetime", "impute"]], on="datetime", how="left")
    print(df_res.head(30))
    return df_res
