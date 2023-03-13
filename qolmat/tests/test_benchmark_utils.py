from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from skopt.space import Integer

from qolmat.benchmark import utils
from qolmat.imputations import imputers


def test_get_search_space() -> None:
    model = imputers.ImputeKNN()
    search_params = {"k": {"min": 2, "max": 10, "type": "Integer"}}
    print(utils.get_search_space(model, {}))
    assert utils.get_search_space(model, {}) == []
    assert utils.get_search_space(model, search_params) == [
        Integer(low=2, high=10, prior='uniform', transform='identity')
    ]


def test_custom_groupby() -> None:
    df = pd.DataFrame(
        [
            ('bird', 'Falconiformes', 389.0),
            ('bird', 'Psittaciformes', 24.0),
            ('mammal', 'Carnivora', 80.2),
            ('mammal', 'Primates', 34.12),
            ('mammal', 'Carnivora', 58),
        ],
        index=['falcon', 'parrot', 'lion', 'monkey', 'leopard'],
        columns=('class', 'order', 'max_speed'),
    )

    assert_frame_equal(utils.custom_groupby(df, groups=[]), df)
    assert (
        type(utils.custom_groupby(df, groups=['class', 'order']))
        == pd.core.groupby.generic.DataFrameGroupBy
    )

######################
# Evaluation metrics #
######################


def test_mean_squared_error() -> None:
    df1 = pd.DataFrame(
        data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        columns=['var1', 'var2', 'var3'],
    )

    df2 = pd.DataFrame(
        data=[[1, 2, 3], [1, 2, 3], [1, 8, 9], [3, 4, 8]],
        columns=['var1', 'var2', 'var3'],
    )
    assert_series_equal(
        utils.mean_squared_error(df1, df2),
        pd.Series([94, 58, 25], index=["var1", "var2", "var3"]),
    )


def test_mean_absolute_error() -> None:
    df1 = pd.DataFrame(
        data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        columns=['var1', 'var2', 'var3'],
    )

    df2 = pd.DataFrame(
        data=[[1, 2, 3], [1, 2, 3], [1, 8, 9], [3, 4, 8]],
        columns=['var1', 'var2', 'var3'],
    )
    assert utils.mean_absolute_error(df1, df2, columnwise_evaluation=False) == 33


def test_weighted_mean_absolute_percentage_error() -> None:
    df1 = pd.DataFrame(
        data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        columns=['var1', 'var2', 'var3'],
    )

    df2 = pd.DataFrame(
        data=[[1, 2, 3], [1, 2, 3], [1, 8, 9], [3, 4, 8]],
        columns=['var1', 'var2', 'var3'],
    )

    assert (
        round(
            utils.weighted_mean_absolute_percentage_error(
                df1, df2, columnwise_evaluation=False
            ),
            4,
        )
        == 0.4484
    )


def test_wasser_distance() -> None:
    df1 = pd.DataFrame(
        data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        columns=['var1', 'var2', 'var3'],
    )

    df2 = pd.DataFrame(
        data=[[1, 2, 3], [1, 2, 3], [1, 8, 9], [3, 4, 8]],
        columns=['var1', 'var2', 'var3'],
    )
    assert_series_equal(
        utils.wasser_distance(df1, df2),
        pd.Series([4, 2.5, 1.75], index=['var1', 'var2', 'var3']),
    )


def test_kl_divergence() -> None:
    df1 = pd.DataFrame(
        data=[[1, 2, 3], [6, 4, 2], [7, 8, 9], [10, 10, 12]],
        columns=['var1', 'var2', 'var3'],
    )

    df2 = pd.DataFrame(
        data=[[1, 2, 3], [5, 2, 3], [1, 8, 9], [3, 4, 6]],
        columns=['var1', 'var2', 'var3'],
    )
    assert_series_equal(
        utils.kl_divergence(df1, df2, columnwise_evaluation=True),
        pd.Series(
            [17.960112, 17.757379, 17.757379], index=['var1', 'var2', 'var3']
        ),
    )
    assert (
        round(utils.kl_divergence(df1, df2, columnwise_evaluation=False), 4)
        == 14.0112
    )


def test_frechet_distance() -> None:
    df1 = pd.DataFrame(
        data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        columns=['var1', 'var2', 'var3'],
    )

    df2 = pd.DataFrame(
        data=[[1, 2, 3], [1, 2, 3], [1, 8, 9], [3, 4, 8]],
        columns=['var1', 'var2', 'var3'],
    )
    assert (
        round(utils.frechet_distance(df1, df2, normalized=False), 4) == 41.6563
    )
    assert (
        round(utils.frechet_distance(df1, df2, normalized=True), 4) == 1.9782
    )
