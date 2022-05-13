from ast import parse
import os, sys
import tarfile
sys.path.append("/Users/tmorzadec/Missions/aifluence-now/src/aifluence/pipelines/imputation")
import gc
import numpy as np
from math import floor
import pandas as pd
from skopt import space
from sklearn.utils import resample

import matplotlib.pyplot as plt

from robust_pca.utils import drawing, utils
from robust_pca.classes.rpca import RPCA
from robust_pca.classes.temporal_rpca import TemporalRPCA, OnlineTemporalRPCA
from robust_pca.classes.evaluate_imputer import EvaluateImputor, eval
from robust_pca.classes.optimize_imputor import rpca_optimizer
import datetime
import imputation_RPCA_entropy

if False:

    data_all = pd.read_parquet(os.path.join("..", "DATA", "H.parq"), engine="pyarrow").reset_index()
    data_all  = data_all.loc[(data_all.direction == "Paris")
                            &(data_all.station == "SAINT-DENIS")
                            &(data_all.datetime_theo_pdt.dt.year == 2019), :]

    gc.collect()

    custom_sum = lambda x: x.sum(skipna=False, min_count=1)

    data_ref = data_all.copy()
    data_ref["mean_time_loading"] = data_all.groupby(["datetime_theo_pdt", "train"])["loading"].transform("mean")
    data_ref["mean_time_unloading"] = data_all.groupby(["datetime_theo_pdt", "train"])["unloading"].transform("mean")
    data_ref["mean_time_load"] = data_all.groupby(["datetime_theo_pdt", "train"])["load"].transform("mean")

    data_ref.loc[:, "load"] = data_ref["load"].fillna(data_ref["mean_time_load"])
    data_ref.loc[:, "loading"] = data_ref["loading"].fillna(data_ref["mean_time_loading"])
    data_ref.loc[:, "unloading"] = data_ref["unloading"].fillna(data_ref["mean_time_unloading"])

    data_series = data_ref.groupby(["datetime_theo_pdt", "train"]).agg(
        {"load": custom_sum,
        "loading": custom_sum,
        "unloading": custom_sum,
        "datetime":"first"}).reset_index()

    data_series.loc[:,"datetime"] = data_series["datetime"].fillna(data_series["datetime_theo_pdt"])
    data_series.to_csv("data_series_load_before_nan.csv")

data_series = pd.read_csv("data_series_load_before_nan.csv", parse_dates=["datetime"])
indices_to_nan = data_series.loc[data_series["load"]>10, :].index
indices_to_nan = resample(
        indices_to_nan,
        replace=False,
        n_samples=floor(len(indices_to_nan) * 0.05),
        random_state=42,
        stratify=None,
    )
data_series["nan"] = False
data_series.loc[indices_to_nan, "nan"] = True
indices_to_nan = data_series.loc[data_series.nan,:].index

data_series_nan = data_series.copy()
data_series_nan.loc[np.isin(data_series.index, indices_to_nan),"load"] = np.nan
print("start agg")
space_tau = space.Real(low=0.05, high=10, name="tau")
space_lam = space.Real(low=0.05, high=10, name="lam")
df_impute, ts_X, ts_A = imputation_RPCA_entropy.impute_RPCA(data_series_nan,
                                              target = "load",
                                              agg_time="15T",
                                              D_n_rows=24*4*7,
                                              search_space = [space_tau, space_lam],
                                              epoch = 5,
                                              n_jobs = 4)


df_comp = data_series[["datetime", "load"]].rename(columns={"load":"load_init"})
df_impute = df_impute.merge(df_comp, on = "datetime", how = "left")
print(df_impute)
rmse, mae, mape, wmape = eval(
        df_impute.loc[df_impute.nan,"load_init"], df_impute.loc[df_impute.nan,"impute"]
    )
print(f"rmse = {rmse}\nmae = {mae}\nmape = {mape}\nwmape = {wmape}")

df_impute.to_csv("St_Denis_load_to_Paris_week_train_by_train_15T.csv")
ts_X.to_csv("St_Denis_load_to_Paris_15T_agg_15_week_agg_to_nan_X.csv")
ts_A.to_csv("St_Denis_load_to_Paris_15T_agg_15_week_agg_to_nan_A.csv")
    
