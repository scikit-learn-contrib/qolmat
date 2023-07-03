import numpy as np
import pytest
from numpy.typing import NDArray

from qolmat.imputations import em_sampler

A = np.array([[3, 1, 0], [1, 1, 0], [0, 0, 1]])
X1 = np.array([[1, 1, 1, 1], [np.nan, np.nan, 4, 2], [1, 3, np.nan, 1]])
mask = np.isnan(X1)

X1_full = np.array(
    [
        [1, 1, 1, 1],
        [4, 4, 4, 2],
        [1, 3, 2, 1],
    ],
    dtype=float,
)

X_expected = np.array(
    [
        [1, 1, 1, 1],
        [-1, -1, 4, 2],
        [1, 3, 0, 1],
    ],
    dtype=float,
)


@pytest.mark.parametrize("A, X, mask", [(A, X1_full, mask)])
def test__gradient_conjugue(A: NDArray, X: NDArray, mask: NDArray):
    X0 = X.copy()
    X0[mask] = 0
    X_out = em_sampler._gradient_conjugue(A, X, mask)
    assert X_out.shape == X1.shape
    assert np.allclose(X[~mask], X_out[~mask])
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


def test_fit_calls_sample_ou_correct_number_of_times(mocker):
    X1 = np.array([[1, 1, 1, 2], [np.nan, np.nan, 4, 2], [1, 3, np.nan, 1]])
    max_iter_em = 3
    em = em_sampler.MultiNormalEM(max_iter_em=max_iter_em)
    mocker.patch("qolmat.imputations.em_sampler.MultiNormalEM._sample_ou", return_value=X1)
    mocker.patch(
        "qolmat.imputations.em_sampler.MultiNormalEM._check_convergence",
        return_value=False,
    )
    mocker.patch("qolmat.imputations.em_sampler.MultiNormalEM.fit_distribution")
    em.fit(X1)
    em_sampler.MultiNormalEM._sample_ou.call_count == max_iter_em
    em_sampler.MultiNormalEM._check_convergence.call_count == max_iter_em
    em_sampler.MultiNormalEM.fit_distribution.call_count == 1


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
