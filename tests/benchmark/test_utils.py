from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from skopt.space import Integer, Real

from qolmat.benchmark import utils
from qolmat.imputations import imputers

# def test_get_search_space() -> None:
#     model = imputers.ImputerKNN()
#     search_params = {"k": {"min": 2, "max": 10, "type": "Integer"}}
#     print(utils.get_search_space(model, {}))
#     assert utils.get_search_space(model, {}) == []
#     assert utils.get_search_space(model, search_params) == [
#         Integer(low=2, high=10, prior="uniform", transform="identity")
#     ]


def test_get_search_space():
    # Test the function with one hyperparameter, type='float'
    search_params = {
        "param1": {
            "col1": {"min": 0.5, "max": 5, "type": "Real"},
            "col2": {"min": 0.6, "max": 6, "type": "Real"},
        },
        "param2": {"min": 0.1, "max": 1, "type": "Integer"},
    }
    expected = [
        Real(low=0.5, high=5, name="param1/col1"),
        Real(low=0.6, high=6, name="param1/col2"),
        Integer(low=0.1, high=1, name="param2"),
    ]
    assert utils.get_search_space(search_params) == expected

    # Test the function with multiple hyperparameters and columns
    search_params = {
        "param1": {"min": 0.2, "max": 2, "type": "Real"},
        "param2": {"min": 0.3, "max": 3, "type": "Integer"},
    }
    expected = [
        Real(low=0.2, high=2, name="param1"),
        Integer(low=0.3, high=3, name="param2"),
    ]
    assert utils.get_search_space(search_params) == expected


def test_custom_groupby() -> None:
    df = pd.DataFrame(
        [
            ("bird", "Falconiformes", 389.0),
            ("bird", "Psittaciformes", 24.0),
            ("mammal", "Carnivora", 80.2),
            ("mammal", "Primates", 34.12),
            ("mammal", "Carnivora", 58),
        ],
        index=["falcon", "parrot", "lion", "monkey", "leopard"],
        columns=("class", "order", "max_speed"),
    )

    assert_frame_equal(utils.custom_groupby(df, groups=[]), df)
    assert (
        type(utils.custom_groupby(df, groups=["class", "order"]))
        == pd.core.groupby.generic.DataFrameGroupBy
    )
