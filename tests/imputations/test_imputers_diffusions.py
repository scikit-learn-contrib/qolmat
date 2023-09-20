import numpy as np
import pandas as pd
import pytest

from qolmat.imputations import imputers_pytorch
from qolmat.imputations.diffusions import diffusions
from qolmat.utils.exceptions import PyTorchExtraNotInstalled

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
    },
    index=pd.date_range("2023-04-17", periods=5, freq="D"),
)
df_incomplete.index = df_incomplete.index.set_names("datetime")


@pytest.mark.parametrize("df", [df_incomplete])
def test_ImputerGenerativeModelPytorch_fit_transform(df: pd.DataFrame) -> None:
    expected = pd.Series(
        {
            "col1": False,
            "col2": False,
            "col3": False,
            "col4": False,
            "col5": False,
        }
    )

    model = diffusions.TabDDPM(dim_input=5, num_noise_steps=10, num_blocks=1, dim_embedding=64)
    imputer = imputers_pytorch.ImputerDiffusion(
        model=model, batch_size=2, epochs=2, x_valid=df, print_valid=True
    )

    result = imputer.fit_transform(df)
    np.testing.assert_array_equal(np.isnan(result).any(), expected)

    model = diffusions.TabDDPMTS(dim_input=5, num_noise_steps=10, num_blocks=1, dim_embedding=64)
    imputer = imputers_pytorch.ImputerDiffusion(
        model=model,
        batch_size=2,
        epochs=2,
        x_valid=df,
        print_valid=True,
        index_datetime="datetime",
    )

    result = imputer.fit_transform(df)
    np.testing.assert_array_equal(np.isnan(result).any(), expected)
