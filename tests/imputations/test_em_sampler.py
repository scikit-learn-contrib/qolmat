import numpy as np
import pytest
from numpy.typing import NDArray

from qolmat.imputations import em_sampler

A: NDArray = np.array([[3, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
A_inverse: NDArray = np.array([[0.5, -0.5, 0], [-0.5, 1.5, 0], [0, 0, 1]], dtype=float)
X_missing: NDArray = np.array(
    [[1, 1, 1, 1], [np.nan, np.nan, 4, 2], [1, 3, np.nan, 1]], dtype=float
)
X_first_guess: NDArray = np.array(
    [[1, 1, 1, 1], [4, 4, 4, 2], [1, 3, 2, 1]],
    dtype=float,
)
X_expected: NDArray = np.array(
    [[1, 1, 1, 1], [-1, -1, 4, 2], [1, 3, 0, 1]],
    dtype=float,
)
mask: NDArray = np.isnan(X_missing)


@pytest.mark.parametrize(
    "A, X_first_guess, X_expected, mask",
    [(A, X_first_guess, X_expected, mask)],
)
def test_gradient_conjugue(
    A: NDArray,
    X_first_guess: NDArray,
    X_expected: NDArray,
    mask: NDArray,
):
    X_result = em_sampler._gradient_conjugue(A, X_first_guess, mask)
    np.testing.assert_allclose(X_result, X_expected, atol=1e-5)
    assert np.sum(X_result * (A @ X_result)) <= np.sum(X_first_guess * (A @ X_first_guess))
    assert np.allclose(X_first_guess[~mask], X_result[~mask])


@pytest.mark.parametrize(
    "A, A_inverse_expected",
    [(A, A_inverse)],
)
def test_invert_robust(A: NDArray, A_inverse_expected: NDArray):
    A_inv = em_sampler.invert_robust(A, epsilon=0)
    assert A_inv.shape == A.shape
    assert np.allclose(A_inv, A_inverse_expected, atol=1e-5)


# @pytest.mark.parametrize("X_missing", [X_missing])
# def test_fit_calls_sample_ou(mocker, X_missing: NDArray):
#     max_iter_em = 3
#     mocker.patch("qolmat.imputations.em_sampler.MultiNormalEM._sample_ou",
# return_value=X_missing)
#     mocker.patch(
#         "qolmat.imputations.em_sampler.MultiNormalEM._check_convergence",
#         return_value=False,
#     )
#     mocker.patch("qolmat.imputations.em_sampler.MultiNormalEM.fit_distribution")
#     em = em_sampler.MultiNormalEM(max_iter_em=max_iter_em)
#     em.fit(X_missing)
#     assert em_sampler.MultiNormalEM._sample_ou.call_count == max_iter_em
#     assert em_sampler.MultiNormalEM._check_convergence.call_count == max_iter_em
#     assert em_sampler.MultiNormalEM.fit_distribution.call_count == 1


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
