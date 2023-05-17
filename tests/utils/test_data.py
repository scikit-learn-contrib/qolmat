import pandas as pd
import pytest
import numpy as np
import datetime

from qolmat.utils import data

columns = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
index = pd.MultiIndex.from_tuples(
    [
        ("Gucheng", datetime.datetime(2013, 3, 1)),
        ("Gucheng", datetime.datetime(2014, 3, 1)),
        ("Gucheng", datetime.datetime(2015, 3, 1)),
        ("Gucheng", datetime.datetime(2016, 3, 1)),
    ],
    names=["station", "datetime"],
)
df = pd.DataFrame(
    [
        [6.0, 18.0, 5.0, np.nan, 800.0, 88.0, 0.1, 1021.1, -18.6, 0.0, 4.4],
        [6.0, 18.0, 5.0, np.nan, 800.0, 88.0, 0.1, 1021.1, -18.6, 0.0, 4.4],
        [6.0, 18.0, 5.0, 0.1, 800.0, 88.0, 0.1, 1021.1, -18.6, 0.0, 4.4],
        [6.0, 18.0, 5.0, 0.1, 800.0, 88.0, 0.1, 1021.1, -18.6, 0.0, 4.4],
    ],
    columns=columns,
    index=index,
)


def test_preprocess_data() -> None:

    columns_raw = [
        "No",
        "year",
        "month",
        "day",
        "hour",
        "PM2.5",
        "PM10",
        "SO2",
        "NO2",
        "CO",
        "O3",
        "TEMP",
        "PRES",
        "DEWP",
        "RAIN",
        "wd",
        "WSPM",
        "station",
    ]
    df_raw = pd.DataFrame(
        [
            [
                1,
                2013,
                3,
                1,
                0,
                6.0,
                18.0,
                5.0,
                np.nan,
                800.0,
                88.0,
                0.1,
                1021.1,
                -18.6,
                0.0,
                "NW",
                4.4,
                "Gucheng",
            ],
            [
                2,
                2014,
                3,
                1,
                0,
                6.0,
                18.0,
                5.0,
                np.nan,
                800.0,
                88.0,
                0.1,
                1021.1,
                -18.6,
                0.0,
                "NW",
                4.4,
                "Gucheng",
            ],
            [
                3,
                2015,
                3,
                1,
                0,
                6.0,
                18.0,
                5.0,
                0.1,
                800.0,
                88.0,
                0.1,
                1021.1,
                -18.6,
                0.0,
                "NW",
                4.4,
                "Gucheng",
            ],
            [
                4,
                2016,
                3,
                1,
                0,
                6.0,
                18.0,
                5.0,
                0.1,
                800.0,
                88.0,
                0.1,
                1021.1,
                -18.6,
                0.0,
                "NW",
                4.4,
                "Gucheng",
            ],
        ],
        columns=columns_raw,
    )

    assert data.preprocess_data(df_raw).equals(df)


def test_add_holes() -> None:
    assert data.add_holes(df, 0.0, 1).isna().sum().sum() == 2
    assert data.add_holes(df, 1.0, 1).isna().sum().sum() > 2


def test_add_station_features() -> None:
    columns_out = columns + ["station=Gucheng"]
    df_out = pd.DataFrame(
        [
            [6.0, 18.0, 5.0, np.nan, 800.0, 88.0, 0.1, 1021.1, -18.6, 0.0, 4.4, 1.0],
            [6.0, 18.0, 5.0, np.nan, 800.0, 88.0, 0.1, 1021.1, -18.6, 0.0, 4.4, 1.0],
            [6.0, 18.0, 5.0, 0.1, 800.0, 88.0, 0.1, 1021.1, -18.6, 0.0, 4.4, 1.0],
            [6.0, 18.0, 5.0, 0.1, 800.0, 88.0, 0.1, 1021.1, -18.6, 0.0, 4.4, 1.0],
        ],
        columns=columns_out,
        index=index,
    )

    assert data.add_station_features(df).equals(df_out)


def test_add_datetime_features() -> None:
    columns_out = columns + ["time_cos"]
    df_out = pd.DataFrame(
        [
            [6.0, 18.0, 5.0, np.nan, 800.0, 88.0, 0.1, 1021.1, -18.6, 0.0, 4.4, 0.51237141],
            [6.0, 18.0, 5.0, np.nan, 800.0, 88.0, 0.1, 1021.1, -18.6, 0.0, 4.4, 0.51237141],
            [6.0, 18.0, 5.0, 0.1, 800.0, 88.0, 0.1, 1021.1, -18.6, 0.0, 4.4, 0.51237141],
            [6.0, 18.0, 5.0, 0.1, 800.0, 88.0, 0.1, 1021.1, -18.6, 0.0, 4.4, 0.5],
        ],
        columns=columns_out,
        index=index,
    )

    np.testing.assert_allclose(data.add_datetime_features(df), df_out, atol=1.0e-5)
