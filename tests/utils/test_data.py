import datetime

import numpy as np
import pandas as pd
import pytest
from pytest_mock.plugin import MockerFixture

from qolmat.utils import data

columns = ["station", "date", "year", "month", "day", "hour", "a", "b", "wd"]
df_beijing_raw = pd.DataFrame(
    [
        ["Beijing", datetime.datetime(2013, 3, 1), 2013, 3, 1, 0, 1, 2, "NW"],
        ["Beijing", datetime.datetime(2013, 3, 1), 2014, 3, 1, 0, 3, np.nan, "NW"],
        ["Beijing", datetime.datetime(2013, 3, 1), 2015, 3, 1, 0, np.nan, 6, "NW"],
    ],
    columns=columns,
)
df_beijing = df_beijing_raw.set_index(["station", "date"])

columns = ["No", "year", "month", "day", "hour", "a", "b", "wd"]
df_beijing_online = pd.DataFrame(
    [
        [1, 2013, 3, 1, 0, 1, 2, "NW"],
        [2, 2014, 3, 1, 0, 3, np.nan, "NW"],
        [3, 2015, 3, 1, 0, np.nan, 6, "NW"],
    ],
    columns=columns,
)
index_preprocess_beijing = pd.MultiIndex.from_tuples(
    [
        ("Beijing", datetime.datetime(2013, 3, 1)),
        ("Beijing", datetime.datetime(2014, 3, 1)),
        ("Beijing", datetime.datetime(2015, 3, 1)),
    ],
    names=["station", "datetime"],
)
df_preprocess_beijing = pd.DataFrame(
    [[1, 2], [3, np.nan], [np.nan, 6]], columns=["a", "b"], index=index_preprocess_beijing
)

columns = ["mean_atomic_mass", "wtd_mean_atomic_mass"]
df_conductor = pd.DataFrame(
    [
        [1, 2],
        [3, 4],
    ],
    columns=columns,
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

index_preprocess_offline = pd.MultiIndex.from_tuples(
    [
        ("Gucheng", datetime.datetime(2013, 3, 1)),
        ("Gucheng", datetime.datetime(2014, 3, 1)),
        ("Gucheng", datetime.datetime(2015, 3, 1)),
    ],
    names=["station", "datetime"],
)
df_preprocess_offline = pd.DataFrame(
    [[1, 2], [3, np.nan], [np.nan, 6]], columns=["a", "b"], index=index_preprocess_offline
)


urllink = "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/"
zipname = "PRSA2017_Data_20130301-20170228"


# @pytest.mark.parametrize("zipname, urllink", [(zipname, urllink)])
# def test_utils_data_download_data(zipname: str, urllink: str, mocker: MockerFixture) -> None:
#     mocker.patch("urllib.request.urlretrieve")
#     mocker.patch("zipfile.ZipFile")
#     list_df_result = data.download_data_from_zip(zipname, urllink)


@pytest.mark.parametrize(
    "name_data, df",
    [
        ("Beijing", df_beijing_raw),
        ("Superconductor", df_conductor),
        ("Beijing_online", df_beijing_online),
        ("Superconductor_online", df_conductor),
        ("Monach_weather", df_monach_weather),
        ("Monach_electricity_australia", df_monach_elec),
        ("Artificial", None),
        ("Bug", None),
    ],
)
def test_utils_data_get_data(name_data: str, df: pd.DataFrame, mocker: MockerFixture) -> None:
    mock_download = mocker.patch("qolmat.utils.data.download_data_from_zip", return_value=[df])
    mock_read = mocker.patch("qolmat.utils.data.read_csv_local", return_value=df)
    mock_read_dl = mocker.patch("pandas.read_csv", return_value=df)
    mocker.patch("qolmat.utils.data.preprocess_data_beijing", return_value=df_preprocess_beijing)

    try:
        df_result = data.get_data(name_data=name_data)
    except ValueError:
        assert name_data not in [
            "Beijing",
            "Superconductor",
            "Artificial",
            "SNCF",
            "Beijing_online",
            "Superconductor_online",
            "Monach_weather",
            "Monach_weather",
            "Monach_electricity_australia",
        ]
        np.testing.assert_raises(ValueError, data.get_data, name_data)
        return

    if name_data == "Beijing":
        assert mock_download.call_count == 0
        assert mock_read.call_count == 1
        pd.testing.assert_frame_equal(df_result, df.set_index(["station", "date"]))
    elif name_data == "Superconductor":
        assert mock_download.call_count == 0
        assert mock_read.call_count == 1
        pd.testing.assert_frame_equal(df_result, df)
    elif name_data == "Beijing_online":
        assert mock_download.call_count == 1
        assert mock_read.call_count == 0
        pd.testing.assert_frame_equal(df_result, df_preprocess_beijing)
    elif name_data == "Superconductor_online":
        assert mock_read_dl.call_count == 1
        assert mock_read.call_count == 0
        pd.testing.assert_frame_equal(df_result, df)
    elif name_data == "Artificial":
        expected_columns = ["signal", "X", "A", "E"]
        assert isinstance(df_result, pd.DataFrame)
        assert df_result.columns.tolist() == expected_columns
    elif name_data == "Monach_weather":
        assert mock_download.call_count == 1
        pd.testing.assert_frame_equal(df_result, df_monach_weather_preprocess)
    elif name_data == "Monach_electricity_australia":
        assert mock_download.call_count == 1
        pd.testing.assert_frame_equal(df_result, df_monach_elec_preprocess)
    else:
        assert False


@pytest.mark.parametrize("df", [df_preprocess_offline])
def test_utils_data_add_holes(df: pd.DataFrame) -> None:
    df_out = data.add_holes(df, 0.0, 1)
    assert df_out.isna().sum().sum() == 2
    df_out = data.add_holes(df, 1.0, 1)
    assert df_out.isna().sum().sum() > 2


@pytest.mark.parametrize(
    "name_data, df",
    [
        ("Beijing", df_beijing),
    ],
)
def test_utils_data_get_data_corrupted(
    name_data: str, df: pd.DataFrame, mocker: MockerFixture
) -> None:
    mock_get = mocker.patch("qolmat.utils.data.get_data", return_value=df)
    df_out = data.get_data_corrupted(name_data)
    print(df_out)
    print(df)
    assert mock_get.call_count == 1
    assert df_out.shape == df.shape
    pd.testing.assert_index_equal(df_out.index, df.index)
    pd.testing.assert_index_equal(df_out.columns, df.columns)
    assert df_out.isna().sum().sum() > df.isna().sum().sum()


@pytest.mark.parametrize("df", [df_preprocess_beijing])
def test_utils_data_add_station_features(df: pd.DataFrame) -> None:
    columns_out = ["a", "b"] + ["station=Beijing"]
    expected = pd.DataFrame(
        [
            [1, 2, 1.0],
            [3, np.nan, 1.0],
            [np.nan, 6, 1.0],
        ],
        columns=columns_out,
        index=index_preprocess_beijing,
    )
    result = data.add_station_features(df)
    pd.testing.assert_frame_equal(result, expected, atol=1e-3)


@pytest.mark.parametrize("df", [df_preprocess_beijing])
def test_utils_data_add_datetime_features(df: pd.DataFrame) -> None:
    columns_out = ["a", "b"] + ["time_cos"]
    expected = pd.DataFrame(
        [
            [1, 2, 0.512],
            [3, np.nan, 0.512],
            [np.nan, 6, 0.512],
        ],
        columns=columns_out,
        index=index_preprocess_beijing,
    )
    result = data.add_datetime_features(df)
    pd.testing.assert_frame_equal(result, expected, atol=1e-3)
