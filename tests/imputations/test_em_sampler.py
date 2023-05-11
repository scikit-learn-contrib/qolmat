import numpy as np
import pytest
from numpy.typing import NDArray

from qolmat.imputations import em_sampler

# from __future__ import annotations

X1 = np.array([[1, 1, 1, 1], [np.nan, np.nan, 4, 2], [1, 3, np.nan, 1]])

# A = np.diag([1, 2, 3])
# A = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
A = np.array([[3, 1, 0], [1, 1, 0], [0, 0, 1]])

mask = np.isnan(X1)

X1_full = np.array(
    [
        [1.0, 1.0, 1.0, 1.0],
        [4.0, 4.0, 4.0, 2.0],
        [1.0, 3.0, 2.0, 1.0],
    ]
)

X_expected = np.array(
    [
        [1.0, 1.0, 1.0, 1.0],
        [-1.0, -1.0, 4.0, 2.0],
        [1.0, 3.0, 0.0, 1.0],
    ]
)


@pytest.mark.parametrize("A", [A])
@pytest.mark.parametrize("X", [X1_full])
@pytest.mark.parametrize("mask", [mask])
def test__gradient_conjugue(A: NDArray, X: NDArray, mask: NDArray):
    X_out = em_sampler._gradient_conjugue(A, X, mask)
    assert X_out.shape == X1.shape
    print("-----")
    print(X)
    print(mask)
    print(X[~mask])
    print(X_out[~mask])
    assert np.allclose(X[~mask], X_out[~mask])
    X0 = X.copy()
    X0[mask] = 0
    assert np.sum(X_out * (A @ X_out)) <= np.sum(X0 * (A @ X0))
    assert np.allclose(
        X_out,
        X_expected,
    )


A_expected = np.array([[0.5, -0.5, 0], [-0.5, 1.5, 0], [0, 0, 1]])


@pytest.mark.parametrize("M", [A])
def test_invert_robust(M: NDArray):
    A_inv = em_sampler.invert_robust(A, epsilon=0)
    print(A)
    print(A_inv)
    assert A_inv.shape == A.shape
    assert np.allclose(
        A_inv,
        A_expected,
    )


@pytest.mark.parametrize("X", [X1])
def test__linear_interpolation(X: NDArray):
    em = em_sampler.MultiNormalEM()
    X_out = em._linear_interpolation(X)
    assert X_out.shape == X.shape
    assert np.allclose(
        X_out,
        X1_full,
    )


# imputations_var = [
#     np.array([1, 2, 3, 3]),
#     np.array([1, 2, 3, 3]),
#     np.array([1, 2, 3, 3]),
#     np.array([1, 2, 3, 3]),
#     np.array([1, 2, 3, 3]),
#     np.array([1, 2, 3, 3]),
# ]
# mu_var = [
#     np.array([1, 2, 3, 3]),
#     np.array([1, 2, 3, 3]),
#     np.array([1, 2, 3, 3]),
#     np.array([1, 2, 3, 3]),
#     np.array([1, 2, 3, 3]),
#     np.array([1, 2, 3, 3]),
# ]
# cov_var = [
#     np.array([1, 2, 3, 3]),
#     np.array([1, 2, 3, 3]),
#     np.array([1, 2, 3, 3]),
#     np.array([1, 2, 3, 3]),
#     np.array([1, 2, 3, 3]),
#     np.array([1, 2, 3, 3]),
# ]
# n_iter_var = 11


# @pytest.mark.parametrize("imputations", [imputations_var])
# @pytest.mark.parametrize("mu", [mu_var])
# @pytest.mark.parametrize("cov", [cov_var])
# @pytest.mark.parametrize("n_iter", [n_iter_var])
# def test_em_sampler_check_convergence(
#    imputations: List[np.ndarray],
#    mu: List[np.ndarray],
#    cov: List[np.ndarray],
#    n_iter: int,
# ) -> None:
#    """Test check convergence for Impute EM"""
#    assert (
#        em_sampler.MultiNormalEM()._check_convergence(imputations, mu, cov, n_iter)
#        == True
#    )
