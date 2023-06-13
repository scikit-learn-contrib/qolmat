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

columns = ["No", "year", "month", "day", "hour", "a", "b", "wd", "station"]
df_offline = pd.DataFrame(
    [
        [1, 2013, 3, 1, 0, 1, 2, "NW", "Gucheng"],
        [2, 2014, 3, 1, 0, 3, np.nan, "NW", "Gucheng"],
        [3, 2015, 3, 1, 0, np.nan, 6, "NW", "Gucheng"],
    ],
    columns=columns,
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
#     list_df_result = data.download_data(zipname, urllink)


@pytest.mark.parametrize(
    "name_data, df",
    [
        ("Beijing", df_beijing),
        ("Beijing_offline", df_offline),
        ("Artificial", None),
        ("Bug", None),
    ],
)
def test_utils_data_get_data(name_data: str, df: pd.DataFrame, mocker: MockerFixture) -> None:
    mock_download = mocker.patch("qolmat.utils.data.download_data", return_value=[df])
    mocker.patch(
        "qolmat.utils.data.preprocess_data_beijing_offline", return_value=df_preprocess_offline
    )
    mocker.patch("qolmat.utils.data.preprocess_data_beijing", return_value=df_preprocess_beijing)
    try:
        df_result = data.get_data(name_data=name_data)
    except ValueError:
        assert name_data not in ["Beijing", "Beijing_offline", "Artificial"]
        np.testing.assert_raises(ValueError, data.get_data, name_data)
        return

    if name_data == "Beijing":
        assert mock_download.call_count == 1
        pd.testing.assert_frame_equal(df_result, df_preprocess_beijing)
    elif name_data == "Beijing_offline":
        assert mock_download.call_count == 1
        pd.testing.assert_frame_equal(df_result, df_preprocess_offline)
    elif name_data == "Artificial":
        expected_columns = ["signal", "X", "A", "E"]
        assert isinstance(df_result, pd.DataFrame)
        assert df_result.columns.tolist() == expected_columns
    else:
        assert False


@pytest.mark.parametrize("df", [df_offline])
def test_utils_data_preprocess_data_beijing_offline(df: pd.DataFrame) -> None:
    result = data.preprocess_data_beijing_offline(df)
    print(result)
    print(df_preprocess_offline)
    print(result.dtypes)
    print(df_preprocess_offline.dtypes)
    pd.testing.assert_frame_equal(result, df_preprocess_offline, atol=1e-3)


@pytest.mark.parametrize("df", [df_preprocess_offline])
def test_utils_data_add_holes(df: pd.DataFrame) -> None:
    df_out = data.add_holes(df, 0.0, 1)
    assert df_out.isna().sum().sum() == 2
    df_out = data.add_holes(df, 1.0, 1)
    assert df_out.isna().sum().sum() > 2


@pytest.mark.parametrize("name_data", ["Beijing"])
def test_utils_data_get_data_corrupted(name_data: str, mocker: MockerFixture) -> None:
    mock_download = mocker.patch("qolmat.utils.data.download_data", return_value=[df_beijing])
    mocker.patch("qolmat.utils.data.preprocess_data_beijing", return_value=df_preprocess_beijing)
    df_out = data.get_data_corrupted()
    df_result = pd.DataFrame(
        [[1, 2], [np.nan, np.nan], [np.nan, 6]], columns=["a", "b"], index=index_preprocess_beijing
    )
    assert mock_download.call_count == 1
    pd.testing.assert_frame_equal(df_result, df_out)


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
