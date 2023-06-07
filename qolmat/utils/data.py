import os
import zipfile
from math import pi
from typing import List, Optional
from urllib import request

import numpy as np
import pandas as pd

from qolmat.benchmark import missing_patterns


def download_data(zipname: str, urllink: str, datapath: str = "data/") -> List[pd.DataFrame]:
    path_zip = os.path.join(datapath)
    if not os.path.exists(path_zip + ".zip"):
        if not os.path.exists(datapath):
            os.mkdir(datapath)
        request.urlretrieve(urllink + zipname + ".zip", path_zip + ".zip")

    with zipfile.ZipFile(path_zip + ".zip", "r") as zip_ref:
        zip_ref.extractall(path_zip)
    data_folder = os.listdir(path_zip)
    subfolder = os.path.join(path_zip, data_folder[0])
    data_files = os.listdir(subfolder)
    list_df = [pd.read_csv(os.path.join(subfolder, file)) for file in data_files]
    return list_df


def get_data(
    name_data: str = "Beijing", datapath: str = "data/", download: Optional[bool] = True
) -> pd.DataFrame:
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
    if name_data == "Beijing":
        urllink = "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/"
        zipname = "PRSA2017_Data_20130301-20170228"
        list_df = download_data(zipname, urllink, datapath=datapath)
        list_df = [preprocess_data(df) for df in list_df]
        df = pd.concat(list_df)
        return df
    elif name_data == "Artificial":
        city = "Wonderland"
        n_samples = 1000
        p1 = 100
        p2 = 20
        amplitude_A = 0.5
        freq_A = 0.05
        amplitude_E = 0.1

        mesh = np.arange(n_samples)

        X_true = 1 + np.sin(2 * pi * mesh / p1) + np.sin(2 * pi * mesh / p2)

        noise = np.random.uniform(size=n_samples)
        A_true = (
            amplitude_A
            * np.where(noise < freq_A, -np.log(noise), 0)
            * (2 * (np.random.uniform(size=n_samples) > 0.5) - 1)
        )

        E_true = amplitude_E * np.random.normal(size=n_samples)

        signal = X_true + E_true
        signal[A_true != 0] = A_true[A_true != 0]

        df = pd.DataFrame({"signal": signal, "index": range(n_samples), "station": city})
        df.set_index(["station", "index"], inplace=True)

        df["X"] = X_true
        df["A"] = A_true
        df["E"] = E_true
        return df
    else:
        raise ValueError(f"Data name {name_data} is unknown!")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data from the "Beijing" datset

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
    df = df.groupby(
        ["station", df.index.get_level_values("datetime").floor("d")], group_keys=False
    ).mean()
    return df


def add_holes(df: pd.DataFrame, ratio_masked: float, mean_size: int) -> pd.DataFrame:
    """
    Creates holes in a dataset with no missing value, starting from `df`. Only used in the
    documentation to design examples.

    Parameters
    ----------
    df : pd.DataFrame
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
    groups = df.index.names.difference(["datetime", "date", "index"])
    generator = missing_patterns.GeometricHoleGenerator(
        1, ratio_masked=ratio_masked, subset=df.columns, groups=groups
    )

    generator.dict_probas_out = {column: 1 / mean_size for column in df.columns}
    generator.dict_ratios = {column: 1 / len(df.columns) for column in df.columns}
    if generator.groups == []:
        mask = generator.generate_mask(df)
    else:
        mask = df.groupby(groups, group_keys=False).apply(generator.generate_mask)

    X_with_nans = df.copy()
    X_with_nans[mask] = np.nan
    return X_with_nans


def get_data_corrupted(
    name_data: str = "Beijing",
    mean_size: int = 90,
    ratio_masked: float = 0.2,
) -> pd.DataFrame:
    """
    Returns a dataframe with controled corruption optained from the source `name_data`

    Parameters
    ----------
    name_data : str
        Name of the data source, can be "Beijing" or "Artificial"
    mean_size: int
        Mean size of the holes to be generated using a geometric law
    ratio_masked: float
        Percent of missing data in each column in the output dataframe
    Returns
    -------
    pd.DataFrame
        Dataframe with missing values
    """
    df = get_data(name_data)
    df = add_holes(df, mean_size=mean_size, ratio_masked=ratio_masked)
    return df


def add_station_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a station feature in the dataset

    Parameters
    ----------
    df : pd.DataFrame
        dataframe no missing values

    Returns
    -------
    pd.DataFrame
        dataframe with missing values
    """
    df = df.copy()
    stations = df.index.get_level_values("station")
    for station in stations.unique():
        df[f"station={station}"] = (stations == station).astype(float)
    return df


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a seasonal feature in the dataset with a cosine function

    Parameters
    ----------
    df : pd.DataFrame
        dataframe no missing values

    Returns
    -------
    pd.DataFrame
        dataframe with missing values
    """
    df = df.copy()
    time = df.index.get_level_values("datetime").to_series()
    days_in_year = time.dt.year.apply(
        lambda x: 366 if ((x % 4 == 0) and (x % 100 != 0)) or (x % 400 == 0) else 365
    )
    time_cos = np.cos(2 * np.pi * time.dt.dayofyear / days_in_year)
    df["time_cos"] = np.array(time_cos)
    return df
