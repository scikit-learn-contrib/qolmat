import urllib
import os
import sys
import pandas as pd
import numpy as np
import zipfile

def get_data(datapath="data/"):
    urllink = "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/"
    # datapath = "../data/"
    zipname = "PRSA2017_Data_20130301-20170228"
    path_zip = os.path.join(datapath, zipname)
    
    if not os.path.exists(path_zip + ".zip"):
        urllib.request.urlretrieve(urllink+zipname+".zip", path_zip + ".zip")
    with zipfile.ZipFile(path_zip + ".zip", "r") as zip_ref:
        zip_ref.extractall(path_zip)
    data_folder = os.listdir(path_zip)
    subfolder = os.path.join(path_zip, data_folder[0])
    data_files = os.listdir(subfolder)
    # cities = sorted([file.split("_")[2] for file in data_files])
    # dfs_init = {city: pd.read_csv(os.path.join(subfolder, file)) for city, file in zip(cities, data_files)}
    # dfs_init = {city: preprocess_data(df) for city, df in dfs_init.items()}
    # df = pd.concat(dfs_init.values(), keys=dfs_init.keys())
    list_df = [pd.read_csv(os.path.join(subfolder, file)) for file in data_files]
    list_df = [preprocess_data(df) for df in list_df]
    df = pd.concat(list_df)
    return df
    # for city, df in dfs_init.items():
    #     df.index = pd.to_datetime(df[["year", "month", "day", "hour"]])
    #     df = df.interpolate().ffill().bfill()
    # df_init = pd.concat(dfs_init)
    # cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN"]
    # cols_sum = ["RAIN"]
    # cols_mean = list(set(cols) - set(cols_sum))
    # cols_imp = ["TEMP", "PRES", "DEWP"]
    # cols_time = ["year", "month", "day", "hour"]
    # df_init_day = pd.concat([
    #     df_init.groupby(["station", "year", "month", "day"])[cols_mean].mean(),
    #     df_init.groupby(["station", "year", "month", "day"])[cols_sum].sum(),
    # ])
    # df_init_day["date"] = pd.to_datetime(df_init_day.reset_index()[["year", "month", "day"]]).values
    # # df_init_day["station"] = df_init_day.index.get_level_values(0)
    # df_init_day = df_init_day.reset_index().set_index(["station", "date"], drop=False)
    # return dfs_init


def preprocess_data(df):
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df.set_index(["station", "datetime"], inplace=True)
    df.drop(columns=["year", "month", "day", "hour", "wd", "No"], inplace=True)
    df.sort_index(inplace=True)
    # cols_sum = ["RAIN"]
    # cols_mean = list(set(df.columns) - set(cols_sum))
    # df[cols_mean] = df.groupby(["station", df.index.get_level_values("datetime").date]).apply(np.mean)
    # df[cols_sum] = df.groupby(["station", df.index.get_level_values("datetime").date]).apply(np.sum)
    dict_agg = {key: np.mean for key in df.columns}
    dict_agg["RAIN"] = np.mean
    df = df.groupby(["station", df.index.get_level_values("datetime").floor("d")]).agg(dict_agg)
    return df
