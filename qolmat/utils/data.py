import os
import sys
import zipfile
from datetime import datetime
from math import pi
from typing import List
from urllib import request

import numpy as np
import pandas as pd

from qolmat.benchmark import missing_patterns

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.join(CURRENT_DIR, "..")


def read_csv_local(data_file_name: str) -> pd.DataFrame:
    """Load csv files

    Parameters
    ----------
    data_file_name : str
        Filename. Has to be "beijing" or "conductors"

    Returns
    -------
    df : pd.DataFrame
        dataframe
    """
    df = pd.read_csv(os.path.join(ROOT_DIR, "data", f"{data_file_name}.csv"))
    return df


def download_data_from_zip(
    zipname: str, urllink: str, datapath: str = "data/"
) -> List[pd.DataFrame]:
    path_zip = os.path.join(datapath, zipname)
    path_zip_ext = path_zip + ".zip"
    url = os.path.join(urllink, zipname) + ".zip"
    os.makedirs(datapath, exist_ok=True)
    if not os.path.exists(path_zip_ext) and not os.path.exists(path_zip):
        request.urlretrieve(url, path_zip_ext)
    if not os.path.exists(path_zip):
        with zipfile.ZipFile(path_zip_ext, "r") as zip_ref:
            zip_ref.extractall(path_zip)
    list_df = get_dataframes_in_folder(path_zip, ".csv")
    return list_df


def get_dataframes_in_folder(path: str, extension: str) -> List[pd.DataFrame]:
    list_df = []
    for folder, _, files in os.walk(path):
        for file in files:
            if extension in file:
                list_df.append(pd.read_csv(os.path.join(folder, file)))
            if ".tsf" in file:
                loaded_data = convert_tsf_to_dataframe(os.path.join(folder, file))
                return [loaded_data]
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
    url_zenodo = "https://zenodo.org/record/"
    if name_data == "Beijing":
        df = read_csv_local("beijing")
        df = df.set_index(["station", "date"])
        return df
    if name_data == "Superconductor":
        df = read_csv_local("conductors")
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
    elif name_data == "Beijing_online":
        # urllink = "https://archive.ics.uci.edu/static/public/381/"
        # zipname = "beijing+pm2+5+data"
        urllink = "https://archive.ics.uci.edu/static/public/501/"
        zipname = "beijing+multi+site+air+quality+data"

        list_df = download_data_from_zip(zipname, urllink, datapath=datapath)
        list_df = [preprocess_data_beijing(df) for df in list_df]
        df = pd.concat(list_df)
        return df
    elif name_data == "Superconductor_online":
        csv_url = (
            "https://huggingface.co/datasets/polinaeterna/"
            "tabular-benchmark/resolve/main/reg_num/superconduct.csv"
        )
        df = pd.read_csv(csv_url, index_col=0)
        return df
    elif name_data == "Monach_weather":
        urllink = os.path.join(url_zenodo, "4654822/files/weather_dataset.zip?download=1")
        zipname = "weather_dataset"
        list_loaded_data = download_data_from_zip(zipname, urllink, datapath=datapath)
        loaded_data = list_loaded_data[0]
        df_list: List[pd.DataFrame] = []
        for k in range(len(loaded_data)):
            values = list(loaded_data["series_value"][k])
            freq = "1D"
            time_index = pd.date_range(
                start=pd.Timestamp("01/01/2010"), periods=len(values), freq=freq
            )
            df_list = df_list + [
                pd.DataFrame(
                    {loaded_data.series_name[k] + " " + loaded_data.series_type[k]: values},
                    index=time_index,
                )
            ]
        minimum = min([len(df) for df in df_list])
        df = pd.concat(df_list, axis=1)
        df = df[:minimum]
        return df
    elif name_data == "Monach_electricity_australia":
        urllink = os.path.join(
            url_zenodo, "4659727/files/australian_electricity_demand_dataset.zip?download=1"
        )
        zipname = "australian_electricity_demand_dataset"
        list_loaded_data = download_data_from_zip(zipname, urllink, datapath=datapath)
        loaded_data = list_loaded_data[0]
        df_list = []
        for k in range(len(loaded_data)):
            values = list(loaded_data["series_value"][k])
            freq = "30min"
            time_index = pd.date_range(
                start=loaded_data.start_timestamp[k], periods=len(values), freq=freq
            )
            df_list = df_list + [
                pd.DataFrame(
                    {loaded_data.series_name[k] + " " + loaded_data.state[k]: values},
                    index=time_index,
                )
            ]
        minimum = min([len(df) for df in df_list])
        df = pd.concat(df_list, axis=1)
        df = df[:minimum]
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
    try:
        groups = df.index.names.difference(["datetime", "date", "index"])
        generator = missing_patterns.GeometricHoleGenerator(
            1, ratio_masked=ratio_masked, subset=df.columns, groups=groups
        )
    except ValueError:
        print("No group")
    else:
        generator = missing_patterns.GeometricHoleGenerator(
            1, ratio_masked=ratio_masked, subset=df.columns
        )

    generator.dict_probas_out = {column: 1 / mean_size for column in df.columns}
    generator.dict_ratios = {column: 1 / len(df.columns) for column in df.columns}
    if generator.groups:
        mask = df.groupby(groups, group_keys=False).apply(generator.generate_mask)
    else:
        mask = generator.generate_mask(df)

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


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    # frequency = None
    # forecast_horizon = None
    # contain_missing_values = None
    # contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if len(line_content) != 3:  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if len(line_content) != 2:  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            # if line.startswith("@frequency"):
                            #     frequency = line_content[1]
                            # elif line.startswith("@horizon"):
                            #     forecast_horizon = int(line_content[1])
                            # elif line.startswith("@missing"):
                            #     contain_missing_values = bool(strtobool(line_content[1]))
                            # elif line.startswith("@equallength"):
                            #     contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception("Attribute section must come before data.")

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(" Attribute section must come before data.")
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(" Missing values should be indicated with ? symbol")

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(numeric_series):
                            raise Exception(
                                "At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(full_info[i], "%Y-%m-%d %H-%M-%S")
                            else:
                                raise Exception("Invalid attribute type.")

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return loaded_data
