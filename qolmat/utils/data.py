import os
import urllib
import zipfile
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from qolmat.benchmark import missing_patterns


def get_data(datapath: str = "data/", download: Optional[bool] = True):
    """Download or generate data

    Parameters
    ----------
    datapath : str, optional
        data path, by default "data/"
    download : bool, optional
        if True: download a public dataset, if False: generate random univariate time series, by
        default True

    Returns
    -------
    pd.DataFrame
        requested data
    """
    if download:
        urllink = "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/"
        zipname = "PRSA2017_Data_20130301-20170228"
        path_zip = os.path.join(datapath, zipname)

        if not os.path.exists(path_zip + ".zip"):
            urllib.request.urlretrieve(urllink + zipname + ".zip", path_zip + ".zip")
        with zipfile.ZipFile(path_zip + ".zip", "r") as zip_ref:
            zip_ref.extractall(path_zip)
        data_folder = os.listdir(path_zip)
        subfolder = os.path.join(path_zip, data_folder[0])
        data_files = os.listdir(subfolder)
        list_df = [pd.read_csv(os.path.join(subfolder, file)) for file in data_files]
        list_df = [preprocess_data(df) for df in list_df]
        df = pd.concat(list_df)
        return df
    else:
        city = "Wonderland"
        x = np.linspace(0, 4 * np.pi, 200)
        y = 3 + np.sin(x) + np.random.random(len(x)) * 0.2
        datelist = pd.date_range(datetime(2013, 3, 1), periods=len(y)).tolist()
        dataset = pd.DataFrame({"var": y, "datetime": datelist[: len(y)], "station": city})
        dataset.set_index(["station", "datetime"], inplace=True)
        return dataset


def preprocess_data(df: pd.DataFrame):
    """Put data into dataframe

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with some specific column names

    Returns
    -------
    pd.DataFrame
        preprocessed dataframe
    """
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df.set_index(["station", "datetime"], inplace=True)
    df.drop(columns=["year", "month", "day", "hour", "wd", "No"], inplace=True)
    df.sort_index(inplace=True)
    dict_agg = {key: np.mean for key in df.columns}
    dict_agg["RAIN"] = np.mean
    df = df.groupby(["station", df.index.get_level_values("datetime").floor("d")]).agg(dict_agg)
    return df


def add_holes(
    X: pd.DataFrame,
    ratio_masked: float,
    mean_size: int,
    groups: List[str] = [],
):
    """
    Creates holes in a dataset with no missing value. Only used in the documentation to design
    examples.

    Parameters
    ----------
    X : pd.DataFrame
        dataframe no missing values

    mean_size : int
        Targeted mean size of the holes to add

    ratio_masked : float
        Targeted global proportion of nans added in the returned dataset

    groups: list of strings
        List of the column names used as groups

    Returns
    -------
    pd.DataFrame
        dataframe with missing values
    """
    generator = missing_patterns.GeometricHoleGenerator(
        1, ratio_masked=ratio_masked, subset=X.columns, groups=groups
    )

    generator.dict_probas_out = {column: 1 / mean_size for column in X.columns}
    generator.dict_ratios = {column: 1 / len(X.columns) for column in X.columns}
    if generator.groups == []:
        mask = generator.generate_mask(X)
    else:
        mask = X.groupby(groups).apply(generator.generate_mask)
    X_with_nans = X.copy()
    X_with_nans[mask] = np.nan
    return X_with_nans


def get_data_corrupted(
    datapath: str = "data/",
    download: Optional[bool] = True,
    mean_size: int = 90,
    ratio_masked: float = 0.2,
    groups: List[str] = [],
):
    df = get_data(datapath, download)
    df = add_holes(df, mean_size=mean_size, ratio_masked=ratio_masked, groups=groups)
    return df
