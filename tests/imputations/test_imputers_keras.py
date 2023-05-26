import numpy as np
import pandas as pd
import pytest

from qolmat.imputations import imputers_keras
from qolmat.utils.exceptions import KerasExtraNotInstalled

# from __future__ import annotations

try:
    import tensorflow as tf
except ModuleNotFoundError:
    raise KerasExtraNotInstalled

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

    estimator.build(input_shape=(None, 2))
    weights = estimator.get_weights()  # Obtenir les poids actuels du modèle
    weights = weights[:4]  # Supprimer le poids inutile
    weights = [
        np.zeros(weight.shape) for weight in weights
    ]  # Initialiser les poids du modèle avec des nombres aléatoires déterministes
    estimator.set_weights(weights)

    estimator.compile(optimizer="adam", loss="mse")
    imputer = imputers_keras.ImputerRegressorKeras(
        estimator=estimator, handler_nan="column", epochs=1
    )

    result = imputer.fit_transform(df)
    expected = pd.DataFrame(
        {
            "col1": [0.002, 15.0, 19, 23.0, 33.0],
            "col2": [69.0, 76.0, 74.0, 80.0, 78.0],
            "col3": [174.0, 166.0, 182.0, 177.0, 175.5],
            "col4": [9.0, 12.0, 11.0, 12.0, 8.0],
            "col5": [93.0, 75.0, 75, 12.0, 75],
        }
    )
    np.testing.assert_allclose(result, expected, atol=1e-5)
