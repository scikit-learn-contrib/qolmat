import datetime

import numpy as np
import pandas as pd
import pytest
from pytest_mock.plugin import MockerFixture

from qolmat.utils import data

columns = ["No", "year", "month", "day", "hour", "a", "b", "wd"]
df_beijing = pd.DataFrame(
    [
        [1, 2013, 3, 1, 0, 1, 2, "NW"],
        [2, 2014, 3, 1, 0, 3, np.nan, "NW"],
        [3, 2015, 3, 1, 0, np.nan, 6, "NW"],
    ],
    columns=columns,
)

index_beijing_preprocess = pd.MultiIndex.from_tuples(
    [
        ("Gucheng", datetime.datetime(2013, 3, 1)),
        ("Gucheng", datetime.datetime(2014, 3, 1)),
        ("Gucheng", datetime.datetime(2015, 3, 1)),
    ],
    names=["station", "datetime"],
)

df_beijing_preprocess = pd.DataFrame(
    [[1, 2], [3, np.nan], [np.nan, 6]], columns=["a", "b"], index=index_beijing_preprocess
)

df_monach_weather = pd.DataFrame(
    {
        "series_name": ["T1", "T2", "T3", "T4", "T5"],
        "series_type": ["rain", "preasure", "temperature", "humidity", "sun"],
        "series_value": [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [5.0, 4.0, 3.0],
            [2.0, 1.0, 4.0],
            [1.0, 4.0, 6.0],
        ],
    }
)

df_monach_weather_preprocess = pd.DataFrame(
    [
        [1.0, 4.0, 5.0, 2.0, 1.0],
        [2.0, 5.0, 4.0, 1.0, 4.0],
        [3.0, 6.0, 3.0, 4.0, 6.0],
    ],
    columns=["T1 rain", "T2 preasure", "T3 temperature", "T4 humidity", "T5 sun"],
    index=pd.date_range(start="2010-01-01", periods=3, freq="1D"),
)


df_monach_elec = pd.DataFrame(
    {
        "series_name": ["T1", "T2", "T3", "T4", "T5"],
        "state": ["NSW", "VIC", "QUN", "SA", "TAS"],
        "start_timestamp": [
            pd.Timestamp("2002-01-01"),
            pd.Timestamp("2002-01-01"),
            pd.Timestamp("2002-01-01"),
            pd.Timestamp("2002-01-01"),
            pd.Timestamp("2002-01-01"),
        ],
        "series_value": [
            [5714.0, 5360.0, 5014.0],
            [3535.0, 3383.0, 3655.0],
            [3382.0, 3288.0, 3172.0],
            [1191.0, 1219.0, 1119.0],
            [315.0, 306.0, 305.0],
        ],
    }
)

df_monach_elec_preprocess = pd.DataFrame(
    [
        [5714.0, 3535.0, 3382.0, 1191.0, 315.0],
        [5360.0, 3383.0, 3288.0, 1219.0, 306.0],
        [5014.0, 3655.0, 3172.0, 1119.0, 305.0],
    ],
    columns=["T1 NSW", "T2 VIC", "T3 QUN", "T4 SA", "T5 TAS"],
    index=pd.date_range(start="2002-01-01", periods=3, freq="30T"),
)


urllink = "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/"
zipname = "PRSA2017_Data_20130301-20170228"


# @pytest.mark.parametrize("zipname, urllink", [(zipname, urllink)])
# def test_utils_data_download_data(zipname: str, urllink: str, mocker: MockerFixture) -> None:
#     mocker.patch("urllib.request.urlretrieve")
#     mocker.patch("zipfile.ZipFile")
#     list_df_result = data.download_data(zipname, urllink)


@pytest.mark.parametrize(
    "name_data, df",
    [
        ("Beijing", df_beijing),
        ("Monach_weather", df_monach_weather),
        ("Monach_electricity_australia", df_monach_elec),
        ("Artificial", None),
        ("Bug", None),
    ],
)
def test_utils_data_get_data(name_data: str, df: pd.DataFrame, mocker: MockerFixture) -> None:
    mock_download = mocker.patch("qolmat.utils.data.download_data", return_value=[df])
    mocker.patch("qolmat.utils.data.preprocess_data_beijing", return_value=df_beijing_preprocess)
    mock_get = mocker.patch("qolmat.utils.data.get_dataframes_in_folder", return_value=[df])
    try:
        df_result = data.get_data(name_data=name_data)
    except ValueError:
        assert name_data not in [
            "Beijing",
            "Monach_weather",
            "Monach_electricity_australia",
            "Artificial",
        ]
        np.testing.assert_raises(ValueError, data.get_data, name_data)
        return

    if name_data == "Beijing":
        assert mock_download.call_count == 0
        assert mock_get.call_count == 1
        pd.testing.assert_frame_equal(df_result, df_beijing_preprocess)
    elif name_data == "Artificial":
        expected_columns = ["signal", "X", "A", "E"]
        assert isinstance(df_result, pd.DataFrame)
        assert df_result.columns.tolist() == expected_columns
    elif name_data == "Monach_weather":
        assert mock_download.call_count == 1
        print(df_result)
        pd.testing.assert_frame_equal(df_result, df_monach_weather_preprocess)
    elif name_data == "Monach_electricity_australia":
        assert mock_download.call_count == 1
        print(df_result)
        pd.testing.assert_frame_equal(df_result, df_monach_elec_preprocess)
    else:
        assert False


@pytest.mark.parametrize("df", [df_beijing_preprocess])
def test_utils_data_add_holes(df: pd.DataFrame) -> None:
    df_out = data.add_holes(df, 0.0, 1)
    assert df_out.isna().sum().sum() == 2
    df_out = data.add_holes(df, 1.0, 1)
    assert df_out.isna().sum().sum() > 2


@pytest.mark.parametrize("name_data", ["Beijing"])
def test_utils_data_get_data_corrupted(name_data: str, mocker: MockerFixture) -> None:
    mocker.patch("qolmat.utils.data.get_data", return_value=df_beijing_preprocess)
    df_out = data.get_data_corrupted()
    df_result = pd.DataFrame(
        [[1, 2], [np.nan, np.nan], [np.nan, 6]], columns=["a", "b"], index=index_beijing_preprocess
    )
    pd.testing.assert_frame_equal(df_result, df_out)


@pytest.mark.parametrize("df", [df_beijing_preprocess])
def test_utils_data_add_station_features(df: pd.DataFrame) -> None:
    columns_out = ["a", "b"] + ["station=Gucheng"]
    expected = pd.DataFrame(
        [
            [1, 2, 1.0],
            [3, np.nan, 1.0],
            [np.nan, 6, 1.0],
        ],
        columns=columns_out,
        index=index_beijing_preprocess,
    )
    result = data.add_station_features(df)
    pd.testing.assert_frame_equal(result, expected, atol=1e-3)


@pytest.mark.parametrize("df", [df_beijing_preprocess])
def test_utils_data_add_datetime_features(df: pd.DataFrame) -> None:
    columns_out = ["a", "b"] + ["time_cos"]
    expected = pd.DataFrame(
        [
            [1, 2, 0.512],
            [3, np.nan, 0.512],
            [np.nan, 6, 0.512],
        ],
        columns=columns_out,
        index=index_beijing_preprocess,
    )
    result = data.add_datetime_features(df)
    pd.testing.assert_frame_equal(result, expected, atol=1e-3)
