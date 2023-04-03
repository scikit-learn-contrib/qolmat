import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from qolmat.benchmark import utils

df0 = np.array([[1, 2, 3], [4, 5, 7], [10, 14, 22]])

df1 = df0 + 1
df2 = df1 + 2


def test_check_dfs_columns():
    df0_c0 = pd.DataFrame(df0, columns=["a", "b", "c"])
    df0_c1 = pd.DataFrame(df0, columns=["b", "a", "c"])
    df0_c2 = pd.DataFrame(df0, columns=["b", "d", "c"])

    utils._check_dfs_columns(df0_c0, df0_c1)

    with pytest.raises(ValueError, match=r".*The columns of the two dataframes do not match*"):
        utils._check_dfs_columns(df0_c0, df0_c2)


def test_mean_squared_error():
    assert utils.mean_squared_error(df0, df1) == 1.0
    assert utils.mean_squared_error(df0, df2) == 4.0


def test_root_mean_squared_error():
    assert utils.root_mean_squared_error(df0, df1) == 1.0
    assert utils.root_mean_squared_error(df0, df2) == 2.0


def test_mean_absolute_error():
    assert utils.mean_absolute_error(df0, df1) == 1.0
    assert utils.mean_absolute_error(df0, df2) == 2.0


def test_weighted_mean_absolute_percentage_error():
    assert round(utils.weighted_mean_absolute_percentage_error(df0, df1), 2) == 0.13
    assert round(utils.weighted_mean_absolute_percentage_error(df0, df2), 2) == 0.26


def test_wasser_distance():
    pd_df0 = pd.DataFrame(df0, columns=["a", "b", "c"])
    pd_df1 = pd.DataFrame(df1, columns=["a", "b", "c"])
    pd_df2 = pd.DataFrame(df2, columns=["a", "b", "c"])

    wr_distances_0 = utils.wasser_distance(pd_df0, pd_df0)
    wr_distances_1 = utils.wasser_distance(pd_df0, pd_df1)
    wr_distances_2 = utils.wasser_distance(pd_df0, pd_df2)

    assert wr_distances_0.equals(pd.Series([0, 0, 0], index=["a", "b", "c"]))
    assert wr_distances_1.equals(pd.Series([1, 1, 1], index=["a", "b", "c"]))
    assert wr_distances_2.equals(pd.Series([2, 2, 2], index=["a", "b", "c"]))


def test_wasser_distance_error():
    pd_df0 = pd.DataFrame(df0, columns=["a", "b", "c"])
    pd_df1_nan = pd.DataFrame(df1, columns=["a", "b", "c"])
    pd_df1_nan.iloc[1, 1] = np.nan

    with pytest.raises(ValueError, match=r".*The column a contains nan in one of the dataframe*"):
        utils.wasser_distance(pd_df0, pd_df1_nan)
