import os
import sys
import zipfile
from math import pi
from typing import List, Optional
from urllib import request

import numpy as np
import pandas as pd

from qolmat.benchmark import missing_patterns


def download_data(zipname: str, urllink: str, datapath: str = "data/") -> List[pd.DataFrame]:
    path_zip = os.path.join(datapath, zipname)
    path_zip_ext = path_zip + ".zip"
    url = os.path.join(urllink, zipname) + ".zip"
    os.makedirs(datapath, exist_ok=True)
    if not os.path.exists(path_zip_ext) and not os.path.exists(path_zip):
        request.urlretrieve(url, path_zip_ext)
    if not os.path.exists(path_zip):
        with zipfile.ZipFile(path_zip_ext, "r") as zip_ref:
            zip_ref.extractall(path_zip)
    list_df = []
    for folder, _, files in os.walk(path_zip):
        for file in files:
            if ".csv" in file:
                list_df.append(pd.read_csv(os.path.join(folder, file)))
    return list_df


def generate_artificial_ts(n_samples, periods, amp_anomalies, ratio_anomalies, amp_noise):
    mesh = np.arange(n_samples)
    X = np.ones(n_samples)
    for p in periods:
        X += np.sin(2 * pi * mesh / p)

    n_anomalies = int(n_samples * ratio_anomalies)
    anomalies = np.random.standard_exponential(size=n_anomalies)
    anomalies *= amp_anomalies * np.random.choice([-1, 1], size=n_anomalies)
    ind_anomalies = np.random.choice(range(n_samples), size=n_anomalies, replace=False)
    A = np.zeros(n_samples)
    A[ind_anomalies] = anomalies

    E = amp_noise * np.random.normal(size=n_samples)
    return X, A, E


def get_data(
    name_data: str = "Beijing",
    datapath: str = "data/",
    n_groups_max: int = sys.maxsize,
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
        urllink = "https://archive.ics.uci.edu/static/public/381/"
        zipname = "beijing+pm2+5+data"

        list_df = download_data(zipname, urllink, datapath=datapath)
        list_df = [preprocess_data_beijing(df) for df in list_df]
        df = pd.concat(list_df)
        return df
    elif name_data == "Beijing_offline":
        urllink = "https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data"
        zipname = "PRSA2017_Data_20130301-20170228"

        list_df = download_data(zipname, urllink, datapath=datapath)
        list_df = [preprocess_data_beijing_offline(df) for df in list_df]
        df = pd.concat(list_df)
        return df
    elif name_data == "Artificial":
        city = "Wonderland"
        n_samples = 1000
        periods = [100, 20]
        amp_anomalies = 0.5
        ratio_anomalies = 0.05
        amp_noise = 0.1

        X, A, E = generate_artificial_ts(
            n_samples, periods, amp_anomalies, ratio_anomalies, amp_noise
        )
        signal = X + A + E
        df = pd.DataFrame({"signal": signal, "index": range(n_samples), "station": city})
        df.set_index(["station", "index"], inplace=True)

        df["X"] = X
        df["A"] = A
        df["E"] = E
        return df
    elif name_data == "SNCF":
        path_file = os.path.join(datapath, "validations_idfm_std.parq")
        df = pd.read_parquet(path_file)
        sizes_stations = df.groupby("station")["val_in"].mean().sort_values()
        n_groups_max = min(len(sizes_stations), n_groups_max)
        stations = sizes_stations.index.get_level_values("station").unique()[-n_groups_max:]
        df = df.loc[stations]
        return df
    else:
        raise ValueError(f"Data name {name_data} is unknown!")


def preprocess_data_beijing(df: pd.DataFrame) -> pd.DataFrame:
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
    df["station"] = "Beijing"
    df.set_index(["station", "datetime"], inplace=True)
    df.drop(
        columns=["year", "month", "day", "hour", "No", "cbwd", "Iws", "Is", "Ir"], inplace=True
    )
    df.sort_index(inplace=True)
    df = df.groupby(
        ["station", df.index.get_level_values("datetime").floor("d")], group_keys=False
    ).mean()
    return df


def preprocess_data_beijing_offline(df: pd.DataFrame) -> pd.DataFrame:
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
