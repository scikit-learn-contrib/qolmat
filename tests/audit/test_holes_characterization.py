import numpy as np
import pandas as pd
import pytest

from qolmat.audit.holes_characterization import MCARTest
from qolmat.imputations.imputers import ImputerEM


@pytest.fixture
def mcar_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    matrix = rng.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=200)
    matrix.ravel()[rng.choice(matrix.size, size=20, replace=False)] = np.nan
    return pd.DataFrame(data=matrix)


@pytest.fixture
def mar_hm_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    matrix = rng.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=200)
    matrix[np.argwhere(matrix[:, 0] >= 1.96), 1] = np.nan
    return pd.DataFrame(data=matrix)


@pytest.fixture
def mcar_hc_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    matrix = rng.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=200)
    matrix[np.argwhere(abs(matrix[:, 0]) >= 1.96), 1] = np.nan
    return pd.DataFrame(data=matrix)


def test_mcar__init__():
    with pytest.raises(ValueError):
        _ = MCARTest(method="hello")


@pytest.mark.parametrize(
    "df_input, expected", [("mcar_df", True), ("mar_hm_df", False), ("mcar_hc_df", True)]
)
def test_little_mcar_test(df_input: pd.DataFrame, expected: bool, request):
    mcar_test_little = MCARTest(method="little", imputer=ImputerEM(random_state=42))
    result = mcar_test_little.test(request.getfixturevalue(df_input))
    assert expected == (result > 0.05)
