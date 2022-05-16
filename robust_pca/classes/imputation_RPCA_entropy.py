from cProfile import label
import gc
from multiprocessing.sharedctypes import Value
import sklearn
from math import floor
from datetime import datetime
from skopt import space
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from robust_pca.classes import aggregate_desagregate as ag
from robust_pca.classes.evaluate_imputer import EvaluateImputor, eval
from robust_pca.classes.optimize_imputor import rpca_optimizer
from robust_pca.classes.temporal_rpca import TemporalRPCA, OnlineTemporalRPCA


def impute_RPCA(
    imputor,
    df_cht,
    search_space,
    target="load",
    epoch=2,
    n_jobs=1,
    agg_time="10T",
    fill="non_zeros",
    norm="L2",
):

    signal = df_cht.copy()
    signal.loc[:, "datetime"] = signal.datetime.dt.tz_localize(tz=None)
    signal = signal.drop_duplicates(subset = ["datetime"], keep = "first")
    signal["day_SNCF"] = (signal["datetime"] - pd.Timedelta("3H")).dt.date
    signal_agg = signal.groupby("day_SNCF", group_keys=False).apply(
        lambda df: ag.agg_df_values(df, target, agg_time, weighted = False).set_index("agg_time")
    )
    signal_agg_weighted = signal.groupby("day_SNCF", group_keys=False).apply(
        lambda df: ag.agg_df_values(df, target, agg_time, weighted = True).set_index("agg_time")
    ).rename(columns = {"agg_values":"weighted_agg_values"})
    signal_agg = pd.concat([signal_agg, signal_agg_weighted], axis=1)
    signal_agg = signal_agg.sort_index()
    new_index = pd.date_range(
        start=signal_agg.index.min(),
        end=signal_agg.index.max(),
        freq=agg_time,
        tz=None,
        normalize=True,
        name="agg_time",
        inclusive="both",
    )
    signal_agg = signal_agg.reindex(new_index, fill_value=0.0)
    signal_agg_nan_indices = signal_agg.reset_index()
    indices_to_nan = list(
        signal_agg_nan_indices.loc[signal_agg_nan_indices["weighted_agg_values"] > 10.0].index
    )
    print(f"len_indices_to_nan = {len(indices_to_nan)}")
    eval_rpca = EvaluateImputor(
        signal=signal_agg.loc[:, "weighted_agg_values"],
        indices_to_nan=indices_to_nan,
        prop=0.05,
        cv=5,
        random_state=42,
    )
    best_imputor, _, best_params, transform_signal, _ = rpca_optimizer(
        imputor,
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

    transform_signal = pd.Series(transform_signal, index=signal_agg.index)
    ts_impute = pd.DataFrame(
        {"impute_agg_values": signal_agg["agg_values"].values}, index=signal_agg.index
    )
    ts_impute["impute_agg_values"].fillna(transform_signal)
    ts_impute = ts_impute.reset_index()

    print(datetime.now())
    df_res = signal.groupby("day_SNCF", group_keys=False).apply(
        lambda df: ag.impute_entropy_day(
            df,
            target,
            ts_agg=ts_impute,
            agg_time=agg_time,
            zero_soil=0.0,
            fill=fill,
            norm=norm,
        )
    )
    df_res = df_res[
        [
            "datetime",
            "impute",
        ]
    ].sort_values(by="datetime")

    return df_res, ts_impute, signal_agg
