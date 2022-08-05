import urllib
import os
import sys
import pandas as pd
import numpy as np
import zipfile
import matplotlib as mpl
from datetime import datetime


def get_data(datapath: str = "data/", download: bool = True):
    """Download or generate data

    Parameters
    ----------
    datapath : str, optional
        data path, by default "data/"
    download : bool, optional
        if True: download a public dataset, if False: generate random univariate time series, by default True

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
        dataset = pd.DataFrame(
            {"var": y, "datetime": datelist[: len(y)], "station": city}
        )
        dataset.set_index(["station", "datetime"], inplace=True)
        return dataset


def preprocess_data(df: pd.DataFrame):
    """_summary_

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
    # cols_sum = ["RAIN"]
    # cols_mean = list(set(df.columns) - set(cols_sum))
    # df[cols_mean] = df.groupby(["station", df.index.get_level_values("datetime").date]).apply(np.mean)
    # df[cols_sum] = df.groupby(["station", df.index.get_level_values("datetime").date]).apply(np.sum)
    dict_agg = {key: np.mean for key in df.columns}
    dict_agg["RAIN"] = np.mean
    df = df.groupby(["station", df.index.get_level_values("datetime").floor("d")]).agg(
        dict_agg
    )
    return df


def make_ellipses(X, ax, color):
    covariances = X.cov()  # gmm.covariances_[0] # [n][:2, :2]
    v, w = np.linalg.eigh(covariances)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    center = X.mean()  # .means_[0]
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    ell = mpl.patches.Ellipse(center, v[0], v[1], 180 + angle, color=color)
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.5)
    ax.add_artist(ell)
    ax.set_aspect("equal", "datalim")


def compare_covariances(df1, df2, var_x, var_y, ax):
    """
    Covariance plot: scatter plot with ellipses

    Parameters
    ----------
    df1 : pd.DataFrame
        dataframe with raw data
    df2 : pd.DataFrame
        dataframe with imputations
    var_x : str
        variable x, column's name of dataframe df1 to compare with
    var_y : str
        variable y, column's name of dataframe df2 to compare with
    ax : matplotlib.axes._subplots.AxesSubplot
        axes
    """
    ax.scatter(df1[var_x], df1[var_y], marker=".", color="C3")
    ax.scatter(df2[var_x], df2[var_y], marker=".", color="C0")
    make_ellipses(df1[[var_x, var_y]], ax, "turquoise")
    make_ellipses(df2[[var_x, var_y]], ax, "crimson")
    ax.set_xlabel(var_x)
    ax.set_ylabel(var_y)
    ax.legend(["Raw data", "After imputation"])


def KL(P: pd.Series, Q: pd.Series) -> float:
    """
    Compute the Kullback-Leibler divergence between distributions P and Q
    Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0.

    Parameters
    ----------
    P : pd.Series
        "true" distribution
    Q : pd.Series
        distribution

    Return
    ------
    float
        KL(P,Q)
    """
    epsilon = 0.00001

    P = (P - P.min()) / (P.max() - P.min())
    Q = (Q - Q.min()) / (Q.max() - Q.min())
    P = P / P.sum()
    Q = Q / Q.sum()

    P = P.copy() + epsilon
    Q = Q.copy() + epsilon

    divergence = np.sum(P * np.log(P / Q))
    return divergence
