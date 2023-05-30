import datetime

import numpy as np
import pandas as pd
import pytest

from qolmat.utils import data

columns = ["a", "b"]
index = pd.MultiIndex.from_tuples(
    [
        ("Gucheng", datetime.datetime(2013, 3, 1)),
        ("Gucheng", datetime.datetime(2014, 3, 1)),
        ("Gucheng", datetime.datetime(2015, 3, 1)),
    ],
    names=["station", "datetime"],
)
df = pd.DataFrame(
    [
        [1, 2],
        [3, np.nan],
        [np.nan, 6],
    ],
    columns=columns,
    index=index,
)


def test_preprocess_data():
    columns_raw = [
        "No",
        "year",
        "month",
        "day",
        "hour",
        "a",
        "b",
        "wd",
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
                1,
                2,
                "NW",
                "Gucheng",
            ],
            [
                2,
                2014,
                3,
                1,
                0,
                3,
                np.nan,
                "NW",
                "Gucheng",
            ],
            [
                3,
                2015,
                3,
                1,
                0,
                np.nan,
                6,
                "NW",
                "Gucheng",
            ],
        ],
        columns=columns_raw,
    )
    result = data.preprocess_data(df_raw)
    # assert result.equals(df)
    pd.testing.assert_frame_equal(result, df, atol=1e-3)


def test_add_holes() -> None:
    df_out = data.add_holes(df, 0.0, 1)
    assert df_out.isna().sum().sum() == 2
    df_out = data.add_holes(df, 1.0, 1)
    assert df_out.isna().sum().sum() > 2


def test_add_station_features() -> None:
    columns_out = columns + ["station=Gucheng"]
    expected = pd.DataFrame(
        [
            [1, 2, 1.0],
            [3, np.nan, 1.0],
            [np.nan, 6, 1.0],
        ],
        columns=columns_out,
        index=index,
    )
    result = data.add_station_features(df)
    pd.testing.assert_frame_equal(result, expected, atol=1e-3)


def test_add_datetime_features() -> None:
    columns_out = columns + ["time_cos"]
    expected = pd.DataFrame(
        [
            [1, 2, 0.512],
            [3, np.nan, 0.512],
            [np.nan, 6, 0.512],
        ],
        columns=columns_out,
        index=index,
    )
    result = data.add_datetime_features(df)
    pd.testing.assert_frame_equal(result, expected, atol=1e-3)
