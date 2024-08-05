import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from qolmat.analysis.holes_characterization import LittleTest, PKLMTest
from qolmat.benchmark.missing_patterns import UniformHoleGenerator
from qolmat.imputations.imputers import ImputerEM


### Tests for the LittleTest class


@pytest.fixture
def mcar_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    matrix = rng.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=200)
    df = pd.DataFrame(data=matrix, columns=["Column_1", "Column_2"])
    hole_gen = UniformHoleGenerator(
        n_splits=1, random_state=42, subset=["Column_2"], ratio_masked=0.2
    )
    df_mask = hole_gen.generate_mask(df)
    return df.mask(df_mask)


@pytest.fixture
def mar_hm_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    matrix = rng.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=200)

    quantile_95 = norm.ppf(0.975)
    df = pd.DataFrame(matrix, columns=["Column_1", "Column_2"])
    df_nan = df.copy()
    df_nan.loc[df_nan["Column_1"] > quantile_95, "Column_2"] = np.nan

    df_mask = df_nan.isna()
    return df.mask(df_mask)


@pytest.fixture
def mar_hc_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    matrix = rng.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=200)

    quantile_95 = norm.ppf(0.975)
    df = pd.DataFrame(matrix, columns=["Column_1", "Column_2"])
    df_nan = df.copy()
    df_nan.loc[df_nan["Column_1"].abs() > quantile_95, "Column_2"] = np.nan

    df_mask = df_nan.isna()
    return df.mask(df_mask)


@pytest.mark.parametrize(
    "df_input, expected", [("mcar_df", True), ("mar_hm_df", False), ("mar_hc_df", True)]
)
def test_little_mcar_test(df_input: pd.DataFrame, expected: bool, request):
    mcar_test_little = LittleTest(random_state=42)
    result = mcar_test_little.test(request.getfixturevalue(df_input))
    assert expected == (result > 0.05)


def test_attribute_error():
    with pytest.raises(AttributeError):
        LittleTest(random_state=42, imputer=ImputerEM(model="VAR"))


### Tests for the PKLMTest class


@pytest.fixture
def np_matrix_with_nan_mcar() -> np.ndarray:
    rng = np.random.default_rng(42)
    n_rows, n_cols = 10, 4
    matrix = rng.normal(size=(n_rows, n_cols))
    num_nan = int(n_rows * n_cols * 0.40)
    nan_indices = rng.choice(n_rows * n_cols, num_nan, replace=False)
    matrix.flat[nan_indices] = np.nan
    return matrix


def test_draw_features_and_target(np_matrix_with_nan_mcar):
    mcar_test_pklm = PKLMTest()
    _, p = np_matrix_with_nan_mcar.shape
    features_idx, target_idx = mcar_test_pklm.draw_features_and_target(np_matrix_with_nan_mcar)
    assert target_idx not in features_idx
    assert 0 <= target_idx <= (p-1)
    for feature_index in features_idx:
        assert 0 <= feature_index <= (p-1)


@pytest.mark.parametrize("dataframe_fixture, features_idx, target_idx, expected",
    [
        ("np_matrix_with_nan_mcar", np.array([1, 0]), 2, True),
        ("np_matrix_with_nan_mcar", np.array([1, 0, 2]), 3, False)
    ]
)
def test_check_draw(request, dataframe_fixture, features_idx, target_idx, expected):
    dataframe = request.getfixturevalue(dataframe_fixture)
    print(dataframe)
    mcar_test_pklm = PKLMTest()
    result = mcar_test_pklm.check_draw(dataframe, features_idx, target_idx)
    assert result == expected
