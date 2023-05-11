import pytest
import pandas as pd
import numpy as np
from typing import List

from qolmat.imputations import imputers_keras
from qolmat.utils.exceptions import KerasExtraNotInstalled

# from __future__ import annotations

try:
    import tensorflow as tf
except ModuleNotFoundError:
    raise KerasExtraNotInstalled

df_complete = pd.DataFrame(
    {
        "col1": [12, 15, 20, 23, 33],
        "col2": [69, 76, 74, 80, 78],
        "col3": [174, 166, 182, 177, 170],
        "col4": [9, 12, 11, 12, 8],
        "col5": [93, 75, 92, 12, 77],
    }
)

df_incomplete = pd.DataFrame(
    {
        "col1": [np.nan, 15, np.nan, 23, 33],
        "col2": [69, 76, 74, 80, 78],
        "col3": [174, 166, 182, 177, np.nan],
        "col4": [9, 12, 11, 12, 8],
        "col5": [93, 75, np.nan, 12, np.nan],
    }
)


@pytest.mark.parametrize("df", [df_incomplete])
def test_ImputerRegressorKeras_fit_transform(df: pd.DataFrame) -> None:
    estimator = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(2, activation="sigmoid"), tf.keras.layers.Dense(1)]
    )
    imputer = imputers_keras.ImputerRegressorKeras(
        estimator=estimator, handler_nan="column", epochs=1
    )
    result = imputer.fit_transform(df)
    expected = pd.DataFrame(
        {
            "col1": [0, 1.6666666666666667, 2, 3, 1.6666666666666667],
            "col2": [-1, 0.3333333333333333, 0.5, 0.3333333333333333, 1.5],
        }
    )
    np.testing.assert_allclose(result, expected)
