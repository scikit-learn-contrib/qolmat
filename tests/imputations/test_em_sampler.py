from typing import List
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
def test_gradient_conjuge(
    A: NDArray,
    X_first_guess: NDArray,
    X_expected: NDArray,
    mask: NDArray,
) -> None:
    """Test the conjugate gradient algorithm."""
    X_result = em_sampler._gradient_conjugue(A, X_first_guess, mask)
    np.testing.assert_allclose(X_result, X_expected, atol=1e-5)
    assert np.sum(X_result * (A @ X_result)) <= np.sum(X_first_guess * (A @ X_first_guess))
    assert np.allclose(X_first_guess[~mask], X_result[~mask])


@pytest.mark.parametrize(
    "A, A_inverse_expected",
    [(A, A_inverse)],
)
def test_invert_robust(A: NDArray, A_inverse_expected: NDArray) -> None:
    """Test the matrix inversion."""
    A_inv = em_sampler.invert_robust(A, epsilon=0)
    assert A_inv.shape == A.shape
    assert np.allclose(A_inv, A_inverse_expected, atol=1e-5)


def test_initialized() -> None:
    """Test that initializations do not crash."""
    em_sampler.EM()
    em_sampler.MultiNormalEM()
    em_sampler.VAR1EM()


@pytest.mark.parametrize("X_missing", [X_missing])
def test_fit_calls(mocker, X_missing: NDArray) -> None:
    """Test number of calls of some methods in MultiNormalEM."""
    max_iter_em = 3
    mock_sample_ou = mocker.patch(
        "qolmat.imputations.em_sampler.MultiNormalEM._sample_ou", return_value=X_missing
    )
    mock_maximize_likelihood = mocker.patch(
        "qolmat.imputations.em_sampler.MultiNormalEM._maximize_likelihood",
        return_value=X_missing,
    )
    mock_check_convergence = mocker.patch(
        "qolmat.imputations.em_sampler.MultiNormalEM._check_convergence",
        return_value=False,
    )
    mock_fit_distribution = mocker.patch(
        "qolmat.imputations.em_sampler.MultiNormalEM.fit_distribution"
    )
    em = em_sampler.MultiNormalEM(max_iter_em=max_iter_em)
    em.fit(X_missing)
    assert mock_sample_ou.call_count == max_iter_em
    assert mock_maximize_likelihood.call_count == 0
    assert mock_check_convergence.call_count == max_iter_em
    assert mock_fit_distribution.call_count == 1


@pytest.mark.parametrize(
    "means, covs, logliks",
    [
        ([np.array([1, 2, 3, 3])] * 15, [np.array([1, 2, 3, 3])] * 15, [1] * 15),
        (
            [np.array([1, 2, 3, 3])] * 15,
            [np.random.uniform(low=0, high=100, size=(1, 4))[0]] * 15,
            [np.random.rand(1)] * 15,
        ),
        (
            [np.random.uniform(low=0, high=100, size=(1, 4))[0]] * 15,
            [np.random.uniform(low=0, high=100, size=(1, 4))[0]] * 15,
            [np.random.rand(1)] * 15,
        ),
        (
            [np.random.uniform(low=0, high=100, size=(1, 4))[0]] * 15,
            [np.random.uniform(low=0, high=100, size=(1, 4))[0]] * 15,
            [1] * 15,
        ),
    ],
)
def test_em_sampler_check_convergence_true(
    means: List[NDArray],
    covs: List[NDArray],
    logliks: List[float],
) -> None:
    """Test the convergence criteria of the MultiNormalEM algorithm."""
    em = em_sampler.MultiNormalEM()
    em.dict_criteria_stop["means"] = means
    em.dict_criteria_stop["covs"] = covs
    em.dict_criteria_stop["logliks"] = logliks
    assert em._check_convergence() == True


@pytest.mark.parametrize(
    "means, covs, logliks",
    [([np.array([1, 2, 3, 3])] * 4, [np.array([1, 2, 3, 3])] * 4, [1] * 4)],
)
def test_em_sampler_check_convergence_false(
    means: List[NDArray],
    covs: List[NDArray],
    logliks: List[float],
) -> None:
    """Test the non-convergence criteria of the MultiNormalEM algorithm."""
    em = em_sampler.MultiNormalEM()
    em.dict_criteria_stop["means"] = means
    em.dict_criteria_stop["covs"] = covs
    em.dict_criteria_stop["logliks"] = logliks
    assert em._check_convergence() == False


def test_no_more_nan_multinormalem() -> None:
    """Test there are no more missing values after the MultiNormalEM algorithm."""
    X = np.array([[1, np.nan, 8, 10], [13, 1, 4, 20], [1, 3, np.nan, 1]], dtype=float)
    assert np.sum(np.isnan(X)) > 0
    assert np.sum(np.isnan(em_sampler.MultiNormalEM().fit_transform(X))) == 0


def test_no_more_nan_var1em() -> None:
    """Test there are no more missing values after the VAR1EM algorithm."""
    X = np.array([[1, np.nan, 8, 10], [13, 1, 4, 20], [1, 3, np.nan, 1]], dtype=float)
    assert np.sum(np.isnan(X)) > 0
    assert np.sum(np.isnan(em_sampler.VAR1EM().fit_transform(X))) == 0
