import numpy as np
import pandas as pd
import pytest
import torch

from qolmat.imputations import imputers_pytorch
from qolmat.utils.exceptions import PyTorchExtraNotInstalled

# from __future__ import annotations

try:
    import torch as nn
except ModuleNotFoundError:
    raise PyTorchExtraNotInstalled

df_incomplete = pd.DataFrame(
    {
        "col1": [np.nan, 15, np.nan, 23, 33],
        "col2": [69, 76, 74, 80, 78],
        "col3": [174, 166, 182, 177, np.nan],
        "col4": [9, 12, 11, 12, 8],
        "col5": [93, 75, np.nan, 12, np.nan],
    }
)

df_completed = pd.DataFrame(
    {
        "col1": [12, 15, 20, 23, 33],
        "col2": [69, 76, 74, 80, 78],
        "col3": [174, 166, 182, 177, 178],
        "col4": [9, 12, 11, 12, 8],
        "col5": [93, 75, 81, 12, 68],
    }
)


@pytest.mark.parametrize("df", [df_incomplete])
def test_ImputerRegressorPyTorch_fit_transform(df: pd.DataFrame) -> None:
    nn.manual_seed(42)
    if nn.cuda.is_available():
        nn.cuda.manual_seed(42)
    estimator = imputers_pytorch.build_mlp(input_dim=2, list_num_neurons=[64, 32])
    imputer = imputers_pytorch.ImputerRegressorPyTorch(
        estimator=estimator, handler_nan="column", epochs=10
    )

    result = imputer.fit_transform(df)
    np.testing.assert_array_equal(df.shape, result.shape)
    np.testing.assert_array_equal(df.index, result.index)
    np.testing.assert_array_equal(df.columns, result.columns)
    np.testing.assert_array_equal(np.isnan(result).any(), np.isnan(df_completed).any())


@pytest.mark.parametrize("df", [df_incomplete])
def test_imputers_pytorch_Autoencoder(df: pd.DataFrame) -> None:
    input = df.values.shape[1]
    latent = 4
    encoder, decoder = imputers_pytorch.build_autoencoder(
        input_dim=input,
        latent_dim=latent,
        output_dim=input,
        list_num_neurons=[4 * latent, 2 * latent],
    )
    autoencoder = imputers_pytorch.ImputerAutoencoder(
        encoder, decoder, epochs=10, lamb=0.01, max_iterations=5, random_state=42
    )
    result = autoencoder.fit_transform(df)
    np.testing.assert_array_equal(df.shape, result.shape)
    np.testing.assert_array_equal(df.index, result.index)
    np.testing.assert_array_equal(df.columns, result.columns)
    np.testing.assert_array_equal(np.isnan(result).any(), np.isnan(df_completed).any())
