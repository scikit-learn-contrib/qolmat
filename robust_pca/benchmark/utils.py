from skopt.space import Categorical, Real, Integer
import pandas as pd
import numpy as np
from sklearn.utils import resample
from math import floor
import sys



def get_search_space(tested_model, search_params):
    search_space = None
    search_name = None
    if str(type(tested_model).__name__) in search_params.keys():
        search_space = []
        search_name = []
        for name_param, vals_params in search_params[str(type(tested_model).__name__)].items():
            search_name.append(name_param)
            if vals_params["type"] == "Integer":
                search_space.append(Integer(low=vals_params["min"], high=vals_params["max"], name=name_param))
            elif vals_params["type"] == "Real":
                search_space.append(Real(low=vals_params["min"], high=vals_params["max"], name=name_param))
            elif vals_params["type"] == "Categorical":
                search_space.append(Categorical(categories=vals_params["categories"], name=name_param))

    return search_space, search_name


def custom_groupby(df, groups):
    print(groups)
    if len(groups) > 0:
        groupby = []
        for g in groups:
            groupby.append(eval("df." + g))
        for key, df in df.groupby(groupby):
            print(key)
            print(df.head(20))
        print(groupby)
        return df.groupby(groupby)
    else:
        return df


def choice_with_mask(df, mask, ratio, random_state=None):
    indices = np.argwhere(mask.to_numpy().flatten())
    indices = resample(
        indices,
        replace=False,
        n_samples=floor(len(indices) * ratio),
        random_state=random_state,
        stratify=None,
    )
    choosed = np.zeros(df.size)
    choosed[indices] = 1
    return pd.DataFrame(
        choosed.reshape(df.shape), index=df.index, columns=df.columns, dtype=bool
    )



def mean_squared_error(df1, df2, squared=True):
    """
    We provide an implementation robust to nans.
    """
    squared_errors = ((df1 - df2) ** 2).sum().sum()
    if squared:
        return squared_errors
    else:
        return np.sqrt(squared_errors)


def mean_absolute_error(df1, df2):
    return (df1 - df2).abs().sum().sum()


def weighted_mean_absolute_percentage_error(df_true, df_pred):
    return np.nanmean(np.abs(df_true.values - df_pred.values)) / np.nanmean(np.abs(df_true.values))


def aggregate_time_data(
    data: pd.DataFrame, 
    agg_time: str,
    na_handling="ignore"
):
    columns = data.columns
    df_ref = data.copy()
    df_ref = df_ref.reset_index()
    df_ref.loc[:, "datetime"] = df_ref.datetime.dt.tz_localize(tz=None)
    df = df_ref.copy()
    df_agg = df_ref.set_index("datetime")
    df_agg_nan = df_agg.copy()
    
    if na_handling == "ignore":
        df_agg = df_agg.resample(agg_time, axis=0, closed="right")[columns].sum()
    elif na_handling == "agg_to_nan":
        agg_sum = lambda x: x.sum(skipna=False, min_count=0)
        df_agg = df_agg.resample(agg_time, axis=0, closed="right")[
            columns
        ].apply(agg_sum)
    elif na_handling == "fillna":
        df_agg["mean_agg"] = df_agg.resample(agg_time, axis=0, closed="right")[
            columns
        ].transform("mean")
        df_agg.loc[:, columns] = np.where(
            np.isnan(df_agg.loc[:, columns]),
            df_agg.loc[:, "mean_agg"],
            df_agg.loc[:, columns],
        )
        df_agg = df_agg.resample(agg_time, axis=0, closed="right")[columns].sum()
    else:
        raise ValueError("'na_handling' should be in ['igore', 'agg_to_nan', 'fillna']")

    custom_sum = lambda x: x.sum(skipna=False, min_count=1)
    df_agg_nan = df_agg_nan.resample(agg_time, axis=0, closed="right")[
        columns
    ].apply(custom_sum)

    indices_to_nan = np.where(df_agg_nan.values > 0.0)
    return df_ref, df_agg, df_agg_nan, indices_to_nan


def disaggregate_time_data(
    df: pd.DataFrame, 
    df_agg: pd.DataFrame, 
    df_impute: pd.DataFrame,
    agg_time: str
):
    init_index = df.index.names
    init_cols = df.columns

    df_impute.index = df_impute.index.shift(1, freq=agg_time)
    df_impute_agg = df_impute.copy()
    df_impute_agg = df_impute_agg.rename(
        columns={c: f"impute_{c}" for c in init_cols}
    )
    cols = df_impute_agg.columns
    # faire par colonne !!!!
    for col in cols:
        df_impute_agg.loc[:,col] = np.where(
            df_impute_agg.loc[:, col] >= 0,
            df_impute_agg.loc[:, col],
            0,
        )
    df_impute_agg = df_impute_agg.reset_index()
    df_impute_agg = df_impute_agg.sort_values(by="datetime")
    

    df_impute_agg[[f"{c}_cum_sum_impute" for c in init_cols]] = df_impute_agg.groupby(
        df_impute_agg.datetime.dt.date
    )[cols].transform(lambda x: x.cumsum())
        
    df = df.reset_index()
    df_res = df.merge(df_impute_agg, on="datetime", how="outer")
    df_res = df_res.sort_values(by="datetime")
    df_res = df_res.set_index("datetime")
    
    df_res[[f"{c}_piecewise_lin" for c in init_cols]] = df_res.groupby(df_res.index.date)[
        [f"{c}_cum_sum_impute" for c in init_cols]
    ].transform(lambda x: x.interpolate(method="time", limit_direction="forward"))
    
    df_res.loc[:, [f"{c}_piecewise_lin" for c in init_cols]] = np.where(
        df_res.loc[:, [f"{c}_piecewise_lin" for c in init_cols]] >= 0, 
        df_res.loc[:, [f"{c}_piecewise_lin" for c in init_cols]], 
        0
    )
    df_res = df_res.sort_index()
    df_res = df_res.loc[df.datetime, :]
    df_res = df_res.reset_index().sort_values(by="datetime")
    
    df_res[[f"{c}_piecewise_lin_shift" for c in init_cols]] = df_res[
        [f"{c}_piecewise_lin" for c in init_cols]
    ].shift(1)
    df_res[[f"{c}_result" for c in init_cols]] = (
        df_res.loc[:, [f"{c}_piecewise_lin" for c in init_cols]].values
        - df_res.loc[:, [f"{c}_piecewise_lin_shift" for c in init_cols]].values
    )
    df_res = df_res.sort_values(by="datetime")
    df_res = df_res.set_index(init_index)
    df_res = df_res[[f"{c}_result" for c in init_cols]]
    df_res = df_res.rename(columns={k:v for k,v in zip(df_res.columns, init_cols)})
    
    return df_res
