import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from qolmat.analysis.holes_characterization import LittleTest, PKLMTest
from qolmat.benchmark.missing_patterns import UniformHoleGenerator
from qolmat.imputations.imputers import ImputerEM
from qolmat.utils.exceptions import TypeNotHandled

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
def multitypes_dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        'int_col': [1, 2, 3],
        'float_col': [1.1, 2.2, 3.3],
        'str_col': ['a', 'b', 'c'],
        'bool_col': [True, False, True],
        'datetime_col': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03'])
    })


@pytest.fixture
def supported_multitypes_dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        'int_col': [1, 2, 3],
        'float_col': [1.1, 2.2, 3.3],
        'str_col': ['a', 'b', 'c'],
        'bool_col': [True, False, True]
    })


@pytest.fixture
def np_matrix_with_nan_mcar() -> np.ndarray:
    rng = np.random.default_rng(42)
    n_rows, n_cols = 10, 4
    matrix = rng.normal(size=(n_rows, n_cols))
    num_nan = int(n_rows * n_cols * 0.40)
    nan_indices = rng.choice(n_rows * n_cols, num_nan, replace=False)
    matrix.flat[nan_indices] = np.nan
    return matrix


@pytest.fixture
def missingness_matrix_mcar(np_matrix_with_nan_mcar):
    return np.isnan(np_matrix_with_nan_mcar).astype(int)


@pytest.fixture
def missingness_matrix_mcar_perm(missingness_matrix_mcar):
    rng = np.random.default_rng(42)
    return rng.permutation(missingness_matrix_mcar)


@pytest.fixture
def oob_probabilities() -> np.ndarray:
    return np.matrix([[0.5, 0.5], [0, 1], [1, 0], [1, 0]]).A


def test__check_pd_df_dtypes_raise_error(multitypes_dataframe):
    with pytest.raises(TypeNotHandled):
        mcar_test_pklm = PKLMTest(random_state=42)
        mcar_test_pklm._check_pd_df_dtypes(multitypes_dataframe)


def test__check_pd_df_dtypes(supported_multitypes_dataframe):
    mcar_test_pklm = PKLMTest(random_state=42)
    mcar_test_pklm._check_pd_df_dtypes(supported_multitypes_dataframe)


def test__encode_dataframe(supported_multitypes_dataframe):
    mcar_test_pklm = PKLMTest(random_state=42)
    np_dataframe = mcar_test_pklm._encode_dataframe(supported_multitypes_dataframe)
    n_rows, n_cols = np_dataframe.shape
    assert n_rows == 3
    assert n_cols == 7


def test__draw_features_and_target_indexes(np_matrix_with_nan_mcar):
    mcar_test_pklm = PKLMTest(random_state=42)
    _, p = np_matrix_with_nan_mcar.shape
    features_idx, target_idx = mcar_test_pklm._draw_features_and_target_indexes(np_matrix_with_nan_mcar)
    assert isinstance(target_idx, np.integer)
    assert isinstance(features_idx, np.ndarray)
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
def test__check_draw(request, dataframe_fixture, features_idx, target_idx, expected):
    dataframe = request.getfixturevalue(dataframe_fixture)
    mcar_test_pklm = PKLMTest()
    result = mcar_test_pklm._check_draw(dataframe, features_idx, target_idx)
    assert result == expected


@pytest.mark.parametrize("dataframe_fixture, features_idx, target_idx",
    [
        ("np_matrix_with_nan_mcar", np.array([1, 0]), 2),
    ]
)
def test__build_dataset(request, dataframe_fixture, features_idx, target_idx):
    dataframe = request.getfixturevalue(dataframe_fixture)
    mcar_test_pklm = PKLMTest()
    X, y = mcar_test_pklm._build_dataset(dataframe, features_idx, target_idx)
    assert X.shape[0] == len(y)
    assert not np.any(np.isnan(X))
    assert not np.any(np.isnan(y))
    assert np.all(np.unique(y) == [0, 1])
    assert X.shape[1] == len(features_idx)
    assert len(y.shape) == 1


@pytest.mark.parametrize("dataframe_fixture, permutation_fixture, features_idx, target_idx",
    [
        ("np_matrix_with_nan_mcar", "missingness_matrix_mcar_perm", np.array([1, 0]), 2),
    ]
)
def test__build_label(
    request,
    dataframe_fixture,
    permutation_fixture,
    features_idx,
    target_idx
):
    dataframe = request.getfixturevalue(dataframe_fixture)
    m_perm = request.getfixturevalue(permutation_fixture)
    mcar_test_pklm = PKLMTest()
    label = mcar_test_pklm._build_label(dataframe, m_perm, features_idx, target_idx)
    assert not np.any(np.isnan(label))
    assert len(label.shape) == 1
    assert np.isin(label, [0, 1]).all()


@pytest.mark.parametrize(
        "oob_fixture, label",
        [
            ("oob_probabilities", np.array([1, 1, 1, 1])),
            ("oob_probabilities", np.array([0, 0, 0, 0])),
        ]
)
def test__U_hat_unique_label(request, oob_fixture, label):
    oob_prob = request.getfixturevalue(oob_fixture)
    mcar_test_pklm = PKLMTest()
    mcar_test_pklm._U_hat(oob_prob, label)


@pytest.mark.parametrize(
        "oob_fixture, label, expected",
        [
            ("oob_probabilities", np.array([1, 0, 0, 0]), 2/3*(np.log(1 - 1e-9) - np.log(1e-9))),
        ]
)
def test__U_hat_computation(request, oob_fixture, label, expected):
    oob_prob = request.getfixturevalue(oob_fixture)
    mcar_test_pklm = PKLMTest()
    u_hat = mcar_test_pklm._U_hat(oob_prob, label)
    assert round(u_hat, 2) == round(expected, 2)
