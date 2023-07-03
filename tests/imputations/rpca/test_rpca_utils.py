import numpy as np
from numpy.typing import NDArray
import pytest
from qolmat.imputations.rpca.rpca_utils import (
    approx_rank,
    soft_thresholding,
    svd_thresholding,
    l1_norm,
    toeplitz_matrix,
)
from qolmat.utils.utils import fold_signal

X_incomplete = np.array(
    [
        [1, np.nan, 3, 2, np.nan],
        [7, 2, np.nan, 1, 1],
        [np.nan, 4, 3, np.nan, 5],
        [np.nan, 4, 3, 5, 5],
        [4, 4, 3, np.nan, 5],
    ]
)
X_complete = np.array(
    [[1, 5, 3, 2, 2], [7, 2, 3, 1, 1], [4, 4, 3, 5, 5], [4, 4, 3, 5, 5], [4, 4, 3, 5, 5]]
)


X_exp_row = np.array(
    [[1.0, 0.0, -1.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0, 0.0], [0.0, 0.0, 1.0, 0.0, -1.0]]
)

X_exp_column = np.array(
    [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)

threshold = 0.95
T = 2
dimension = 5


@pytest.mark.parametrize("X", [X_complete])
def test_rpca_utils_approx_rank(X: NDArray):
    result = approx_rank(M=X)
    np.testing.assert_allclose(result, 3)


@pytest.mark.parametrize("X", [X_complete])
@pytest.mark.parametrize("threshold", [threshold])
def test_rpca_utils_soft_thresholding(X: NDArray, threshold: float):
    result = soft_thresholding(X=X, threshold=threshold)
    X_expected = np.array(
        [
            [0.05, 4.05, 2.05, 1.05, 1.05],
            [6.05, 1.05, 2.05, 0.05, 0.05],
            [3.05, 3.05, 2.05, 4.05, 4.05],
            [3.05, 3.05, 2.05, 4.05, 4.05],
            [3.05, 3.05, 2.05, 4.05, 4.05],
        ]
    )
    np.testing.assert_allclose(result, X_expected)


@pytest.mark.parametrize("X", [X_complete])
@pytest.mark.parametrize("threshold", [threshold])
def test_rpca_utils_svd_thresholding(X: NDArray, threshold: float):
    result = svd_thresholding(X=X, threshold=threshold)
    X_expected = np.array(
        [
            [1.22954596, 4.19288775, 2.58695225, 2.11783467, 2.11783467],
            [6.14486781, 1.96110299, 2.72428711, 1.21646981, 1.21646981],
            [3.84704901, 3.8901204, 2.92414337, 4.6397143, 4.6397143],
            [3.84704901, 3.8901204, 2.92414337, 4.6397143, 4.6397143],
            [3.84704901, 3.8901204, 2.92414337, 4.6397143, 4.6397143],
        ]
    )
    np.testing.assert_allclose(result, X_expected, rtol=1e-5)


@pytest.mark.parametrize("X", [X_incomplete])
def test_rpca_utils_l1_norm(X: NDArray):
    result = l1_norm(M=X_complete)
    np.testing.assert_allclose(result, 90)


@pytest.mark.parametrize("T", [T])
@pytest.mark.parametrize("dimension", [dimension])
@pytest.mark.parametrize("model, X_expected", [("row", X_exp_row), ("column", X_exp_column)])
def test_rpca_utils_toeplitz_matrix(T: int, dimension: int, model: str, X_expected: NDArray):
    result = toeplitz_matrix(T=T, dimension=dimension, model=model)
    np.testing.assert_allclose(result, X_expected)
