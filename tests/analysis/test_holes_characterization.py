import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from qolmat.analysis.holes_characterization import LittleTest
from qolmat.benchmark.missing_patterns import UniformHoleGenerator
from qolmat.imputations.imputers import ImputerEM


@pytest.fixture
def mcar_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    matrix = rng.multivariate_normal(
        mean=[0, 0], cov=[[1, 0], [0, 1]], size=200
    )
    df = pd.DataFrame(data=matrix, columns=["Column_1", "Column_2"])
    hole_gen = UniformHoleGenerator(
        n_splits=1, random_state=42, subset=["Column_2"], ratio_masked=0.2
    )
    df_mask = hole_gen.generate_mask(df)
    return df.mask(df_mask)


@pytest.fixture
def mar_hm_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    matrix = rng.multivariate_normal(
        mean=[0, 0], cov=[[1, 0], [0, 1]], size=200
    )

    quantile_95 = norm.ppf(0.975)
    df = pd.DataFrame(matrix, columns=["Column_1", "Column_2"])
    df_nan = df.copy()
    df_nan.loc[df_nan["Column_1"] > quantile_95, "Column_2"] = np.nan

    df_mask = df_nan.isna()
    return df.mask(df_mask)


@pytest.fixture
def mar_hc_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    matrix = rng.multivariate_normal(
        mean=[0, 0], cov=[[1, 0], [0, 1]], size=200
    )

    quantile_95 = norm.ppf(0.975)
    df = pd.DataFrame(matrix, columns=["Column_1", "Column_2"])
    df_nan = df.copy()
    df_nan.loc[df_nan["Column_1"].abs() > quantile_95, "Column_2"] = np.nan

    df_mask = df_nan.isna()
    return df.mask(df_mask)


@pytest.mark.parametrize(
    "df_input, expected",
    [("mcar_df", True), ("mar_hm_df", False), ("mar_hc_df", True)],
)
def test_little_mcar_test(df_input: pd.DataFrame, expected: bool, request):
    mcar_test_little = LittleTest(random_state=42)
    result = mcar_test_little.test(request.getfixturevalue(df_input))
    assert expected == (result > 0.05)


def test_attribute_error():
    with pytest.raises(AttributeError):
        LittleTest(random_state=42, imputer=ImputerEM(model="VAR"))
