import datetime

import numpy as np
import pandas as pd
import pytest

from qolmat.utils import data
from pytest_mock.plugin import MockerFixture

columns = ["No", "year", "month", "day", "hour", "a", "b", "wd", "station"]
df = pd.DataFrame(
    [
        [1, 2013, 3, 1, 0, 1, 2, "NW", "Gucheng"],
        [2, 2014, 3, 1, 0, 3, np.nan, "NW", "Gucheng"],
        [3, 2015, 3, 1, 0, np.nan, 6, "NW", "Gucheng"],
    ],
    columns=columns,
)

index_preprocess = pd.MultiIndex.from_tuples(
    [
        ("Gucheng", datetime.datetime(2013, 3, 1)),
        ("Gucheng", datetime.datetime(2014, 3, 1)),
        ("Gucheng", datetime.datetime(2015, 3, 1)),
    ],
    names=["station", "datetime"],
)
df_preprocess = pd.DataFrame(
    [[1, 2], [3, np.nan], [np.nan, 6]], columns=["a", "b"], index=index_preprocess
)

urllink = "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/"
zipname = "PRSA2017_Data_20130301-20170228"


# @pytest.mark.parametrize("zipname, urllink", [(zipname, urllink)])
# def test_utils_data_download_data(zipname: str, urllink: str, mocker: MockerFixture) -> None:
#     mocker.patch("urllib.request.urlretrieve")
#     mocker.patch("zipfile.ZipFile")
#     list_df_result = data.download_data(zipname, urllink)


@pytest.mark.parametrize("name_data", ["Beijing", "Artificial", "Bug"])
def test_utils_data_get_data(name_data: str, mocker: MockerFixture) -> None:
    mock_download = mocker.patch("qolmat.utils.data.download_data", return_value=[df])
    mocker.patch("qolmat.utils.data.preprocess_data", return_value=df_preprocess)
    try:
        df_result = data.get_data(name_data=name_data)
    except ValueError:
        assert name_data not in ["Beijing", "Artificial"]
        np.testing.assert_raises(ValueError, data.get_data, name_data)
        return

    if name_data == "Beijing":
        assert mock_download.call_count == 1
        pd.testing.assert_frame_equal(df_result, df_preprocess)
    elif name_data == "Artificial":
        expected_columns = ["signal", "X", "A", "E"]
        assert isinstance(df_result, pd.DataFrame)
        assert df_result.columns.tolist() == expected_columns
    else:
        assert False


@pytest.mark.parametrize("df", [df])
def test_utils_data_preprocess_data(df: pd.DataFrame) -> None:
    result = data.preprocess_data(df)
    pd.testing.assert_frame_equal(result, df_preprocess, atol=1e-3)


@pytest.mark.parametrize("df", [df_preprocess])
def test_utils_data_add_holes(df: pd.DataFrame) -> None:
    df_out = data.add_holes(df, 0.0, 1)
    assert df_out.isna().sum().sum() == 2
    df_out = data.add_holes(df, 1.0, 1)
    assert df_out.isna().sum().sum() > 2


@pytest.mark.parametrize("name_data", ["Beijing"])
def test_utils_data_get_data_corrupted(name_data: str, mocker: MockerFixture) -> None:
    mock_download = mocker.patch("qolmat.utils.data.download_data", return_value=[df])
    mocker.patch("qolmat.utils.data.preprocess_data", return_value=df_preprocess)
    df_out = data.get_data_corrupted()
    df_result = pd.DataFrame(
        [[1, 2], [np.nan, np.nan], [np.nan, 6]], columns=["a", "b"], index=index_preprocess
    )
    assert mock_download.call_count == 1
    pd.testing.assert_frame_equal(df_result, df_out)


@pytest.mark.parametrize("df", [df_preprocess])
def test_utils_data_add_station_features(df: pd.DataFrame) -> None:
    columns_out = ["a", "b"] + ["station=Gucheng"]
    expected = pd.DataFrame(
        [
            [1, 2, 1.0],
            [3, np.nan, 1.0],
            [np.nan, 6, 1.0],
        ],
        columns=columns_out,
        index=index_preprocess,
    )
    result = data.add_station_features(df)
    pd.testing.assert_frame_equal(result, expected, atol=1e-3)


@pytest.mark.parametrize("df", [df_preprocess])
def test_utils_data_add_datetime_features(df: pd.DataFrame) -> None:
    columns_out = ["a", "b"] + ["time_cos"]
    expected = pd.DataFrame(
        [
            [1, 2, 0.512],
            [3, np.nan, 0.512],
            [np.nan, 6, 0.512],
        ],
        columns=columns_out,
        index=index_preprocess,
    )
    result = data.add_datetime_features(df)
    pd.testing.assert_frame_equal(result, expected, atol=1e-3)
