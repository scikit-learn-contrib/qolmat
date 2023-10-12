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
        [1, 7, np.nan, np.nan],
        [np.nan, 2, 4, 4],
        [-3, np.nan, 3, 3],
        [2, -1, np.nan, 5],
        [np.nan, 1, 5, 5],
    ]
)

X_complete = np.array([[1, 7, 4, 4], [5, 2, 4, 4], [-3, 3, 3, 3], [2, -1, 5, 5], [2, 1, 5, 5]])


@pytest.mark.parametrize("X", [X_complete])
def test_rpca_utils_approx_rank(X: NDArray):
    result = approx_rank(M=X)
    np.testing.assert_allclose(result, 3)


@pytest.mark.parametrize("X", [X_complete])
@pytest.mark.parametrize("threshold", [2])
def test_rpca_utils_soft_thresholding(X: NDArray, threshold: float):
    result = soft_thresholding(X=X, threshold=threshold)
    #     X_complete = np.array(
    #     [[1, 7, 4, 4, 4],
    #      [5, 2, 4, 4, 4],
    #      [-3, 3, 3, 3, 3],
    #      [2, -1, 5, 5, 5],
    #      [2, 1, 5, 5, 5]]
    # )
    X_expected = np.array(
        [
            [0, 5, 2, 2],
            [3, 0, 2, 2],
            [-1, 1, 1, 1],
            [0, 0, 3, 3],
            [0, 0, 3, 3],
        ]
    )
    np.testing.assert_allclose(result, X_expected)


@pytest.mark.parametrize("X", [X_complete])
@pytest.mark.parametrize("threshold", [0.95])
def test_rpca_utils_svd_thresholding(X: NDArray, threshold: float):
    result = svd_thresholding(X=X, threshold=threshold)
    print(result)
    X_expected = np.array(
        [
            [0.928, 6.182, 3.857, 3.857],
            [4.313, 1.842, 3.831, 3.831],
            [-2.355, 2.782, 2.723, 2.723],
            [1.951, -0.610, 4.570, 4.570],
            [1.916, 1.098, 4.626, 4.626],
        ]
    )
    np.testing.assert_allclose(result, X_expected, atol=1e-3)


@pytest.mark.parametrize("X", [X_incomplete])
def test_rpca_utils_l1_norm(X: NDArray):
    result = l1_norm(M=X_complete)
    np.testing.assert_allclose(result, 69)


@pytest.mark.parametrize("T", [2])
@pytest.mark.parametrize("dimension", [5])
def test_rpca_utils_toeplitz_matrix(T: int, dimension: int):
    result = toeplitz_matrix(T=T, dimension=dimension)
    result_np = result.toarray()
    X_exp = np.array(
        [[1, 0, -1, 0, 0], [0, 1, 0, -1, 0], [0, 0, 1, 0, -1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    )
    np.testing.assert_allclose(result_np, X_exp)
