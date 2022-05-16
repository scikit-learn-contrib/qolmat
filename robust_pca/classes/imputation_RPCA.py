import gc
from multiprocessing.sharedctypes import Value
import sklearn
from math import floor

from skopt import space
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from robust_pca.classes.evaluate_imputer import EvaluateImputor, eval
from robust_pca.classes.optimize_imputor import rpca_optimizer
from robust_pca.classes.temporal_rpca import TemporalRPCA, OnlineTemporalRPCA


def impute_RPCA(
    df_cht,
    target="load",
    epoch=2,
    n_jobs=1,
    agg_time="10T",
    D_n_rows=24 * 4,
    search_space=None,
    na_handling="agg_to_nan",
    **params,
):
    signal_ref = df_cht.reset_index()
    signal_ref.loc[:, "datetime"] = signal_ref.datetime.dt.tz_localize(tz=None)
    signal = signal_ref.copy()
    signal_agg = signal.loc[:, ["datetime", target]].set_index("datetime")
    signal_agg_nan = signal_agg.copy()

    if na_handling == "ignore":
        signal_agg = signal_agg.resample(agg_time, axis=0, closed="right")[target].sum()
    elif na_handling == "agg_to_nan":
        agg_sum = lambda x: x.sum(skipna=False, min_count=0)
        signal_agg = signal_agg.resample(agg_time, axis=0, closed="right")[
            target
        ].apply(agg_sum)
    elif na_handling == "fillna":
        signal_agg["mean_agg"] = signal_agg.resample(agg_time, axis=0, closed="right")[
            target
        ].transform("mean")
        signal_agg.loc[:, target] = np.where(
            np.isnan(signal_agg.loc[:, target]),
            signal_agg.loc[:, "mean_agg"],
            signal_agg.loc[:, target],
        )
        signal_agg = signal_agg.resample(agg_time, axis=0, closed="right")[target].sum()
    else:
        raise ValueError("'na_handling' should be in ['igore', 'agg_to_nan', 'fillna']")

    custom_sum = lambda x: x.sum(skipna=False, min_count=1)
    signal_agg_nan = signal_agg_nan.resample(agg_time, axis=0, closed="right")[
        target
    ].apply(custom_sum)

    indices_to_nan = list(signal_agg_nan.loc[signal_agg_nan > 10.0].reset_index().index)
    print("AAA")
    print(f"len_indices_to_nan = {len(indices_to_nan)}")

    if len(params) == 0:
        eval_rpca = EvaluateImputor(
            signal=signal_agg,
            indices_to_nan=indices_to_nan,
            prop=0.05,
            cv=3,
            random_state=42,
        )

        if search_space is None:
            search_space_exp = [
                space.Real(low=-7, high=7, name="tau"),
                space.Real(low=-7, high=7, name="lam"),
            ]
            best_imputor, _, best_params = rpca_optimizer(
                TemporalRPCA(n_rows=D_n_rows, norm="L2"),
                imputor_eval=eval_rpca,
                space=search_space_exp,
                scoring="mae",
                exp_variables=True,
                n_random_starts=max(1, epoch // 5),
                epoch=epoch,
                n_jobs=n_jobs,
                verbose=False,
                return_signal=False,
            )
            gc.collect()
            print(best_imputor.get_params())
            print(best_params)

            tau_approx = np.exp(best_params["tau"])
            lam_approx = np.exp(best_params["lam"])

            space_tau = space.Real(
                low=tau_approx * 0.8, high=tau_approx * 1.2, name="tau"
            )
            space_lam = space.Real(
                low=lam_approx * 0.8, high=lam_approx * 1.2, name="lam"
            )

            search_space = [space_tau, space_lam]

        best_imputor, _, best_params, transform_signal, anomalies = rpca_optimizer(
            TemporalRPCA(n_rows=D_n_rows, norm="L2"),
            imputor_eval=eval_rpca,
            space=search_space,
            scoring="mae",
            exp_variables=False,
            n_random_starts=max(1, epoch // 5),
            epoch=epoch,
            n_jobs=n_jobs,
            verbose=False,
        )
        gc.collect()
        print(best_imputor.get_params())
        print(best_params)

    else:

        imputor = TemporalRPCA(n_rows=D_n_rows, norm="L2", **params)
        transform_signal, anomalies, _ = imputor.fit_transform(signal=signal_agg)

    ts_impute = pd.Series(
        transform_signal,
        index=signal_agg.index,
    )
    ts_anomalies = pd.Series(
        anomalies,
        index=signal_agg.index,
    )
    ts_impute.index = ts_impute.index.shift(1, freq=agg_time)
    ts_anomalies.index = ts_anomalies.index.shift(1, freq=agg_time)

    ts_impute_agg = ts_impute.to_frame(name=f"impute_{agg_time}")
    # ts_impute_agg.loc[:, f"impute_{agg_time}"] = np.where(
    #     ts_impute_agg.loc[:, f"impute_{agg_time}"] >= 0,
    #     ts_impute_agg.loc[:, f"impute_{agg_time}"],
    #     0,
    # )

    ts_impute_agg[f"anomalies_{agg_time}"] = ts_anomalies

    ts_impute_agg.index.name = "datetime_agg"
    ts_impute_agg = ts_impute_agg.sort_index()

    # ts_impute_agg["cum_sum_impute"] = ts_impute_agg.groupby(
    #     ts_impute_agg.datetime.dt.date
    # )[f"impute_{agg_time}"].transform(lambda x: x.cumsum())
    # df_res = signal_ref.merge(ts_impute_agg, on="datetime", how="outer")
    # df_res = df_res.sort_values(by="datetime")
    # df_res = df_res.set_index("datetime")
    # df_res["piecewise_lin"] = df_res.groupby(df_res.index.date)[
    #     "cum_sum_impute"
    # ].transform(lambda x: x.interpolate(method="time", limit_direction="both"))

    # df_res.loc[:, "piecewise_lin"] = np.where(
    #     df_res.loc[:, "piecewise_lin"] >= 0, df_res.loc[:, "piecewise_lin"], 0
    # )

    df_res = signal_ref.copy()
    df_res["datetime_agg"] = df_res.datetime.dt.ceil(agg_time)
    df_res = df_res.merge(ts_impute_agg, on="datetime_agg", how="left")
    df_res = df_res.sort_index()
    # df_res = df_res.loc[signal_ref.datetime, :]
    # df_res = df_res.reset_index().sort_values(by="datetime")
    # df_res["piecewise_lin_shift"] = df_res.groupby(df_res.datetime.dt.date)[
    #     "piecewise_lin"
    # ].shift(1)
    # df_res["impute"] = (
    #     df_res.loc[:, "piecewise_lin"].values
    #     - df_res.loc[:, "piecewise_lin_shift"].values
    # )
    # df_res.loc[:, "impute"] = df_res["impute"].fillna(0)
    # df_res = df_res.set_index("datetime").sort_index()
    return df_res, ts_impute_agg
