import sys
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import pytest
from qolmat.utils import utils
from pytest_mock.plugin import MockerFixture
from io import StringIO

from qolmat.utils.exceptions import NotDimension2, SignalTooShort


df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})


@pytest.mark.parametrize("iteration, total", [(1, 1)])
def test_utils_utils_display_progress_bar(iteration: int, total: int, capsys) -> None:
    captured_output = StringIO()
    sys.stdout = captured_output
    utils.progress_bar(
        iteration, total, prefix="Progress", suffix="Complete", decimals=1, length=2, fill="█"
    )
    captured_output.seek(0)
    output = captured_output.read().strip()
    sys.stdout = sys.__stdout__

    output_expected = "Progress |██| 100.0% Complete"
    assert output == output_expected


@pytest.mark.parametrize("values, lag_max", [(pd.Series([1, 2, 3, 4, 5]), 3)])
def test_utils_utils_acf(values, lag_max):
    result = utils.acf(values, lag_max)
    result_expected = pd.Series([1.0, 1.0, 1.0])
    pd.testing.assert_series_equal(result, result_expected, atol=0.001)


X_incomplete = np.array(
    [
        [1, np.nan, 3, 2, np.nan],
        [7, 2, np.nan, 1, 1],
        [np.nan, 4, 3, np.nan, 5],
        [np.nan, 4, 3, 5, 5],
        [4, 4, 3, np.nan, 5],
    ]
)
X_exp_mean = np.array(
    [
        [1.0, 3.5, 3.0, 2.0, 4.0],
        [7.0, 2.0, 3.0, 1.0, 1.0],
        [4.0, 4.0, 3.0, 2.66666667, 5.0],
        [4.0, 4.0, 3.0, 5.0, 5.0],
        [4.0, 4.0, 3.0, 2.66666667, 5.0],
    ]
)
X_exp_median = np.array(
    [
        [1.0, 4.0, 3.0, 2.0, 5.0],
        [7.0, 2.0, 3.0, 1.0, 1.0],
        [4.0, 4.0, 3.0, 2.0, 5.0],
        [4.0, 4.0, 3.0, 5.0, 5.0],
        [4.0, 4.0, 3.0, 2.0, 5.0],
    ]
)
X_exp_zeros = np.array(
    [
        [1.0, 0.0, 3.0, 2.0, 0.0],
        [7.0, 2.0, 0.0, 1.0, 1.0],
        [0.0, 4.0, 3.0, 0.0, 5.0],
        [0.0, 4.0, 3.0, 5.0, 5.0],
        [4.0, 4.0, 3.0, 0.0, 5.0],
    ]
)


@pytest.mark.parametrize("X", [X_incomplete])
@pytest.mark.parametrize(
    "method, X_expected", [("mean", X_exp_mean), ("median", X_exp_median), ("zeros", X_exp_zeros)]
)
def test_utils_utils_impute_nans(X: NDArray, method: str, X_expected: NDArray):
    result = utils.impute_nans(M=X, method=method)
    np.testing.assert_allclose(result, X_expected)


@pytest.mark.parametrize("X", [X_incomplete])
def test_utils_linear_interpolation(X: NDArray):
    result = utils.linear_interpolation(X_incomplete)
    expected = np.array(
        [
            [1, 2, 3, 2, 2],
            [7, 2, 1.5, 1, 1],
            [4, 4, 3, 4, 5],
            [4, 4, 3, 5, 5],
            [4, 4, 3, 4, 5],
        ]
    )
    np.testing.assert_allclose(result, expected)


signal = np.array([1, 4, np.nan, 3, 2])
X_expected3 = np.array(
    [
        [1.0, np.nan, 3.0, 2.0, np.nan, 7.0, 2.0, np.nan, 1.0],
        [1.0, np.nan, 4.0, 3.0, np.nan, 5.0, np.nan, 4.0, 3.0],
        [5.0, 5.0, 4.0, 4.0, 3.0, np.nan, 5.0, np.nan, np.nan],
    ]
)
X_expected2 = np.array(
    [
        [1, 4, np.nan],
        [3, 2, np.nan],
    ]
)


@pytest.mark.parametrize("X", [X_incomplete])
def test_utils_prepare_data_2D_succeed(X: NDArray):
    result = utils.prepare_data(X, 1)
    np.testing.assert_allclose(result, X_incomplete)


@pytest.mark.parametrize("X", [X_incomplete])
def test_utils_prepare_data_1D_succeed(X: NDArray):
    X = X.flatten()
    result = utils.prepare_data(X, 5)
    np.testing.assert_allclose(result, X_incomplete)


@pytest.mark.parametrize("X", [X_incomplete])
def test_utils_prepare_data_2D_uneven(X: NDArray):
    result = utils.prepare_data(X, 3)
    np.testing.assert_allclose(result.shape, (15, 2))


@pytest.mark.parametrize("X", [X_incomplete])
def test_utils_prepare_data_consistant(X: NDArray):
    result1 = utils.prepare_data(X, 1)
    result2 = utils.prepare_data(result1, 2)
    result3 = utils.prepare_data(X, 2)
    np.testing.assert_allclose(result2, result3)


@pytest.mark.parametrize("X", [signal])
def test_utils_fold_signal_1D_fail(X: NDArray):
    np.testing.assert_raises(NotDimension2, utils.fold_signal, X, 100)
