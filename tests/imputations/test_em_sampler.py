from typing import List

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.datasets import make_spd_matrix
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

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


@pytest.fixture
def generate_multinormal_predefined_mean_cov():
    np.random.seed(41)
    n, d = 500, 20
    mean = np.array([np.random.uniform(d) for _ in range(d)])
    covariance = make_spd_matrix(n_dim=d, random_state=28)
    X = np.random.multivariate_normal(mean=mean, cov=covariance, size=n)
    mask = np.array(np.full_like(X, False), dtype=bool)
    for j in range(X.shape[1]):
        ind = np.random.choice(
            np.arange(X.shape[0]), size=np.int64(np.ceil(X.shape[0] * 0.1)), replace=False
        )
        mask[ind, j] = True
    X_missing = X.copy()
    X_missing[mask] = np.nan
    return {"mean": mean, "covariance": covariance, "X": X.T, "X_missing": X_missing.T}


@pytest.fixture
def generate_var1_process():
    np.random.seed(41)
    d, n = 3, 10_000
    A = np.array([[0.3, 0.1, -0.02], [0.2, 0.03, -0.2], [-0.01, 0.3, 0.4]])
    B = np.array([0.0, 0.04, 0.01])
    omega = make_spd_matrix(n_dim=d, random_state=208) * 1e-6
    X = np.zeros((n, d))
    noise = np.random.multivariate_normal(mean=np.zeros(d), cov=omega, size=n)
    for i in range(1, n):
        X[i] = A.dot(X[i - 1] - B) + B + noise[i]
    mask = np.array(np.full_like(X, False), dtype=bool)
    for j in range(X.shape[1]):
        ind = np.random.choice(
            np.arange(X.shape[0]), size=np.int64(np.ceil(X.shape[0] * 0.1)), replace=False
        )
        mask[ind, j] = True
    X_missing = X.copy()
    X_missing[mask] = np.nan
    return {"X": X.T, "X_missing": X_missing.T, "A": A, "B": B, "omega": omega}


@pytest.fixture
def generate_varp_process():
    np.random.seed(41)
    d, n = 3, 10_000
    A1 = np.array([[0.03, 0.03, 0.13], [0.03, 0.03, 0.14], [0.0, 0.02, 0.23]], dtype=float)
    A2 = np.array([[0.08, 0.1, 0.08], [0.03, 0.16, 0.14], [0.0, 0.2, 0.23]], dtype=float)
    A = [A1, A2]
    B = np.array([0.001, 0.023, 0.019])
    omega = make_spd_matrix(n_dim=d, random_state=208) * 1e-6
    noise = np.random.multivariate_normal(mean=np.zeros(d), cov=omega, size=n)
    X = np.zeros((n, d))
    for i in range(1, n):
        for ind, mat_A in enumerate(A):
            X[i] += mat_A.dot(X[i - (ind + 1)] - B)
        X[i] = X[i] + B + noise[i]
    mask = np.array(np.full_like(X, False), dtype=bool)
    for j in range(X.shape[1]):
        ind = np.random.choice(
            np.arange(X.shape[0]), size=np.int64(np.ceil(X.shape[0] * 0.1)), replace=False
        )
        mask[ind, j] = True
    X_missing = X.copy()
    X_missing[mask] = np.nan
    return {"X": X.T, "X_missing": X_missing.T, "A": A, "B": B, "omega": omega}


@pytest.mark.parametrize(
    "A, X_first_guess, X_expected, mask",
    [(A, X_first_guess, X_expected, mask)],
)
def test_gradient_conjugue(
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


def test_fit_var_model(generate_var1_process):
    """Test the fit for VAR"""
    result_aic = em_sampler.fit_var_model(generate_var1_process["X"].T, p=1, criterion="aic")
    result_bic = em_sampler.fit_var_model(generate_var1_process["X"].T, p=1, criterion="bic")

    assert isinstance(result_aic, VARResultsWrapper)
    assert isinstance(result_bic, VARResultsWrapper)
    assert result_aic.k_ar == 1
    assert result_bic.k_ar == 1


def test_get_lag_p(generate_varp_process):
    """Test if it can retrieve the lag p"""
    lag_p = em_sampler.get_lag_p(generate_varp_process["X"])
    assert lag_p == 2


def test_initialized() -> None:
    """Test that initializations do not crash."""
    em_sampler.EM()
    em_sampler.MultiNormalEM()
    em_sampler.VARpEM()


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


@pytest.mark.parametrize(
    "As, Bs, omegas, logliks",
    [
        (
            [np.array([1, 2, 3, 3])] * 12,
            [np.array([1])] * 12,
            [np.array([1, 2, 3, 3])] * 12,
            [1] * 12,
        )
    ],
)
def test_varem_sampler_check_convergence_true(
    As: List[NDArray],
    Bs: List[NDArray],
    omegas: List[NDArray],
    logliks: List[float],
) -> None:
    """Test the convergence criteria of the VAR1EM algorithm."""
    em = em_sampler.VARpEM(p=1, random_state=32)
    em.dict_criteria_stop["As"] = As
    em.dict_criteria_stop["Bs"] = Bs
    em.dict_criteria_stop["omegas"] = omegas
    em.dict_criteria_stop["logliks"] = logliks
    assert em._check_convergence() == True


@pytest.mark.parametrize(
    "As, Bs, omegas, logliks",
    [([np.array([1, 2, 3, 3])] * 4, [np.array([1])] * 4, [np.array([1, 2, 3, 3])] * 4, [1] * 4)],
)
def test_varem_sampler_check_convergence_false(
    As: List[NDArray],
    Bs: List[NDArray],
    omegas: List[NDArray],
    logliks: List[float],
) -> None:
    """Test the non-convergence criteria of the VAR1EM algorithm."""
    em = em_sampler.VARpEM(p=1, random_state=32)
    em.dict_criteria_stop["As"] = As
    em.dict_criteria_stop["Bs"] = Bs
    em.dict_criteria_stop["omegas"] = omegas
    em.dict_criteria_stop["logliks"] = logliks
    assert em._check_convergence() == False


def test_no_more_nan_multinormalem() -> None:
    """Test there are no more missing values after the MultiNormalEM algorithm."""
    X = np.array([[1, np.nan, 8, 1], [3, 1, 4, 2], [2, 3, np.nan, 1]], dtype=float)
    assert np.sum(np.isnan(X)) > 0
    assert np.sum(np.isnan(em_sampler.MultiNormalEM().fit_transform(X))) == 0


def test_no_more_nan_varpem() -> None:
    """Test there are no more missing values after the VAR1EM algorithm."""
    X = np.array([[1, np.nan, 8, 1], [3, 1, 4, 2], [2, 3, np.nan, 1]], dtype=float)
    assert np.sum(np.isnan(X)) > 0
    assert np.sum(np.isnan(em_sampler.VARpEM(p=1).fit_transform(X))) == 0


def test_mean_covariance_multinormalem(generate_multinormal_predefined_mean_cov):
    """Test the MultiNormalEM provides good mean and covariance estimations."""
    data = generate_multinormal_predefined_mean_cov
    em = em_sampler.MultiNormalEM()
    X_imputed = em.fit_transform(data["X_missing"])
    covariance_imputed = np.cov(X_imputed, rowvar=True)
    mean_imputed = np.mean(X_imputed, axis=1)
    assert np.sum(np.abs(data["mean"] - mean_imputed)) / np.sum(np.abs(data["mean"])) < 1e-1
    assert (
        np.sum(np.abs(data["covariance"] - covariance_imputed))
        / np.sum(np.abs(data["covariance"]))
        < 1e-1
    )


def test_mean_covariance_varpem(generate_multinormal_predefined_mean_cov):
    """Test the MultiNormalEM provides good mean and covariance estimations."""
    data = generate_multinormal_predefined_mean_cov
    em = em_sampler.VARpEM(p=1)
    X_imputed = em.fit_transform(data["X_missing"])
    covariance_imputed = np.cov(X_imputed, rowvar=True)
    mean_imputed = np.mean(X_imputed, axis=1)
    assert np.sum(np.abs(data["mean"] - mean_imputed)) / np.sum(np.abs(data["mean"])) < 1e-1
    assert (
        np.sum(np.abs(data["covariance"] - covariance_imputed))
        / np.sum(np.abs(data["covariance"]))
        < 1e-1
    )


def test_fit_distribution_var1em(generate_var1_process):
    """Test the fit VAR(1) provides good A and B estimates (no imputation)."""
    data = generate_var1_process
    em = em_sampler.VARpEM(p=1)
    em.fit_distribution(data["X"])
    np.testing.assert_allclose(data["A"], em.A, atol=1e-1)
    np.testing.assert_allclose(data["B"], em.B, atol=1e-1)
    np.testing.assert_allclose(data["omega"], em.omega, atol=1e-1)


def test_parameters_after_imputation_var1em(generate_var1_process):
    """Test the VAR(1) provides good A and B estimates."""
    data = generate_var1_process
    em = em_sampler.VARpEM(p=1)
    _ = em.fit_transform(data["X_missing"])
    np.testing.assert_allclose(data["A"], em.list_A[0], rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(data["B"], em.B, rtol=1e-1, atol=1e-1)


def test_parameters_after_imputation_varpem(generate_varp_process):
    """Test the VAR(2) provides good A and B estimates."""
    data = generate_varp_process
    em = em_sampler.VARpEM(p=2)
    _ = em.fit_transform(data["X_missing"])
    np.testing.assert_allclose(data["A"][0], em.list_A[0], rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(data["A"][1], em.list_A[1], rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(data["B"], em.B, rtol=1e-1, atol=1e-1)
