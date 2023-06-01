from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from skopt.space import Integer, Real

from qolmat.benchmark import utils
from qolmat.imputations import imputers


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
