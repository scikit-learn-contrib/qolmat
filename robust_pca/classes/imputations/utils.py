from skopt.space import Categorical, Real, Integer
import pandas as pd
import numpy as np
from sklearn.utils import resample
from math import floor

def get_search_space(tested_model, search_params):
    search_space = None
    search_name = None
    if str(type(tested_model).__name__) in search_params.keys():
        search_space = []
        search_name = []
        for name_param, vals_params in search_params[
            str(type(tested_model).__name__)
        ].items():
            search_name.append(name_param)
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

    return search_space, search_name


def custom_groupby(df, groups):
    if len(groups) > 0:
        groupby = []
        for g in groups:
            groupby.append(eval("df." + g))
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
    return pd.DataFrame(choosed.reshape(df.shape), index=df.index, columns=df.columns, dtype=bool)

def mean_squared_error(df1, df2, squared=True):
    """
    We provide an implementation robust to nans.
    """
    squared_errors = ((df1 - df2)**2).sum().sum()
    if squared:
        return squared_errors
    else:
        return np.sqrt(squared_errors)

def mean_absolute_error(df1, df2):
    return (df1 - df2).abs().sum().sum()
