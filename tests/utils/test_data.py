import datetime
import os

import numpy as np
import pandas as pd
import pytest
from pytest_mock.plugin import MockerFixture
from unittest.mock import MagicMock, patch
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
    [[1, 2], [3, np.nan], [np.nan, 6]],
    columns=["a", "b"],
    index=index_preprocess_beijing,
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
    [[1, 2], [3, np.nan], [np.nan, 6]],
    columns=["a", "b"],
    index=index_preprocess_offline,
)

df_sncf = pd.DataFrame(
    {
        "station": [
            "Gare du Nord",
            "Gare du Nord",
            "Gare de Lyon",
            "Gare de Lyon",
            "Gare Montparnasse",
            "Gare Montparnasse",
        ],
        "val_in": [120, np.nan, 180, np.nan, 140, 130],
    }
)
df_sncf.set_index("station", inplace=True)

df_titanic = pd.DataFrame(
    {
        "Survived": [0, 1, 1],
        "Sex": ["Male", "Female", "Male"],
        "Age": ["22", "unknown", "33"],
        "SibSp": [0, 0, 2],
        "Parch": [2, 2, 1],
        "Fare": ["210.5", "15.5", "7.25"],
        "Embarked": ["Cherbourg", "Liverpool", "Liverpool"],
        "Pclass": [1, 2, 3],
    }
)

df_beijing_without_preprocess = pd.DataFrame(
    {
        "year": [2020, 2020, 2020],
        "month": [1, 1, 1],
        "day": [1, 1, 1],
        "hour": [0, 1, 2],
        "No": [1, 2, 3],
        "cbwd": ["NW", "NW", "NW"],
        "Iws": [23.5, 24.6, 25.7],
        "Is": [0, 0, 0],
        "Ir": [0, 0, 0],
        "pm2.5": [200, 180, 150],
    }
)


urllink = "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/"
zipname = "PRSA2017_Data_20130301-20170228"


@patch("pandas.read_csv", return_value=df_beijing)
def test_read_csv_local(mock_read_csv):
    result_df = data.read_csv_local("beijing")
    pd.testing.assert_frame_equal(result_df, df_beijing)
    mock_read_csv.assert_called_once()


@patch("os.makedirs")
@patch("os.path.exists")
@patch("urllib.request.urlretrieve")
@patch("zipfile.ZipFile")
@patch("qolmat.utils.data.get_dataframes_in_folder")
def test_download_data_from_zip_all_cases(
    mock_get_dataframes_in_folder,
    mock_zipfile,
    mock_urlretrieve,
    mock_exists,
    mock_makedirs,
):
    mock_exists.side_effect = [False, False, False, True]
    mock_zipfile.return_value.__enter__.return_value = MagicMock()

    expected_dfs = [pd.DataFrame([1]), pd.DataFrame([2])]
    mock_get_dataframes_in_folder.return_value = expected_dfs

    result_dfs = data.download_data_from_zip("zipname", "http://example.com/")

    assert result_dfs == expected_dfs
    mock_urlretrieve.assert_called_once()
    mock_zipfile.assert_called_once()
    mock_makedirs.assert_called_once_with("data/", exist_ok=True)
    mock_get_dataframes_in_folder.assert_called_once()

    mock_urlretrieve.reset_mock()
    mock_zipfile.reset_mock()
    mock_makedirs.reset_mock()
    mock_exists.side_effect = [True, True]

    result_dfs = data.download_data_from_zip("zipname", "http://example.com/")
    assert result_dfs == expected_dfs
    mock_urlretrieve.assert_not_called()
    mock_zipfile.assert_not_called()
    mock_makedirs.assert_called_once_with("data/", exist_ok=True)
    mock_get_dataframes_in_folder.assert_called()


@patch("os.walk")
@patch("pandas.read_csv", return_value=df_conductor)
@patch("qolmat.utils.data.convert_tsf_to_dataframe", return_value=df_beijing)
def test_get_dataframes_in_folder(mock_convert_tsf, mock_read_csv, mock_walk):
    mock_walk.return_value = [("/fakepath", ("subfolder",), ("file.csv",))]
    result_csv = data.get_dataframes_in_folder("/fakepath", ".csv")
    assert len(result_csv) == 1
    mock_read_csv.assert_called_once_with(os.path.join("/fakepath", "file.csv"))
    pd.testing.assert_frame_equal(result_csv[0], df_conductor)

    mock_read_csv.reset_mock()
    mock_convert_tsf.reset_mock()
    mock_walk.return_value = [("/fakepath", ("subfolder",), ("file.tsf",))]
    result_tsf = data.get_dataframes_in_folder("/fakepath", ".tsf")
    assert len(result_tsf) == 1
    mock_convert_tsf.assert_called_once_with(os.path.join("/fakepath", "file.tsf"))
    pd.testing.assert_frame_equal(result_tsf[0], df_beijing)
    mock_read_csv.assert_called()


@patch("numpy.random.normal")
@patch("numpy.random.choice")
@patch("numpy.random.standard_exponential")
def test_generate_artificial_ts(mock_standard_exponential, mock_choice, mock_normal):
    n_samples = 100
    periods = [10, 20]
    amp_anomalies = 1.0
    ratio_anomalies = 0.1
    amp_noise = 0.1

    mock_standard_exponential.return_value = np.ones(int(n_samples * ratio_anomalies))
    mock_choice.return_value = np.arange(int(n_samples * ratio_anomalies))
    mock_normal.return_value = np.zeros(n_samples)

    X, A, E = data.generate_artificial_ts(
        n_samples, periods, amp_anomalies, ratio_anomalies, amp_noise
    )

    assert len(X) == n_samples
    assert len(A) == n_samples
    assert len(E) == n_samples
    assert np.all(E == 0)


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
        ("Titanic", df_titanic),
        ("SNCF", df_sncf),
        ("Bug", None),
    ],
)
def test_data_get_data(name_data: str, df: pd.DataFrame, mocker: MockerFixture) -> None:
    mock_download = mocker.patch("qolmat.utils.data.download_data_from_zip", return_value=[df])
    mock_read = mocker.patch("qolmat.utils.data.read_csv_local", return_value=df)
    mock_read_dl = mocker.patch("pandas.read_csv", return_value=df)
    mocker.patch("qolmat.utils.data.preprocess_data_beijing", return_value=df_preprocess_beijing)
    mocker.patch("pandas.read_parquet", return_value=df_sncf)

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
            "Titanic",
            "SNCF",
        ]
        np.testing.assert_raises(ValueError, data.get_data, name_data)
        return

    if name_data == "Beijing":
        assert mock_download.call_count == 0
        assert mock_read.call_count == 1
        assert df_result.index.names == ["station", "date"]
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
    elif name_data == "Titanic":
        assert mock_read_dl.call_count == 1
        assert np.shape(df_result) == (3, 7)
    elif name_data == "SNCF":
        assert not df_result.empty
        assert df_result.index.name == "station"
        assert df_result["val_in"].sum() == df["val_in"].sum()
    else:
        assert False


@pytest.mark.parametrize("df", [df_beijing_without_preprocess])
def test_preprocess_data_beijing(df: pd.DataFrame) -> None:
    result_df = data.preprocess_data_beijing(df)

    assert "year" not in result_df.columns
    assert "pm2.5" in result_df.columns
    assert result_df.index.names == ["station", "datetime"]
    assert all(result_df.index.get_level_values("station") == "Beijing")
    assert len(result_df) == 1
    assert np.isclose(result_df.loc[(("Beijing"),), "pm2.5"], 176.66666666666666)


@pytest.mark.parametrize("df", [df_preprocess_offline])
def test_data_add_holes(df: pd.DataFrame) -> None:
    df_out = data.add_holes(df, 0.0, 1)
    assert df_out.isna().sum().sum() == 2
    df_out = data.add_holes(df.loc[("Gucheng",)], 1.0, 1)
    assert df_out.isna().sum().sum() > 2


@pytest.mark.parametrize(
    "name_data, df",
    [
        ("Beijing", df_beijing),
    ],
)
def test_data_get_data_corrupted(name_data: str, df: pd.DataFrame, mocker: MockerFixture) -> None:
    mock_get = mocker.patch("qolmat.utils.data.get_data", return_value=df)
    df_out = data.get_data_corrupted(name_data)
    assert mock_get.call_count == 1
    assert df_out.shape == df.shape
    pd.testing.assert_index_equal(df_out.index, df.index)
    pd.testing.assert_index_equal(df_out.columns, df.columns)
    assert df_out.isna().sum().sum() > df.isna().sum().sum()


@pytest.mark.parametrize("df", [df_preprocess_beijing])
def test_data_add_station_features(df: pd.DataFrame) -> None:
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
def test_data_add_datetime_features(df: pd.DataFrame) -> None:
    columns_out = ["a", "b"] + ["time_cos", "time_sin"]
    result = data.add_datetime_features(df)
    pd.testing.assert_index_equal(result.index, df.index)
    assert result.columns.tolist() == columns_out
    pd.testing.assert_frame_equal(result.drop(columns=["time_cos", "time_sin"]), df)
    assert (result["time_cos"] ** 2 + result["time_sin"] ** 2 == 1).all()
