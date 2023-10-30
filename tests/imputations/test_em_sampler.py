from typing import List, Literal
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import linalg
from sklearn.datasets import make_spd_matrix


from qolmat.imputations import em_sampler

np.random.seed(42)

A: NDArray = np.array([[3, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
A_inverse: NDArray = np.array([[0.5, -0.5, 0], [-0.5, 1.5, 0], [0, 0, 1]], dtype=float)
X_missing = np.array(
    [[1, np.nan, 1], [1, np.nan, 3], [1, 4, np.nan], [1, 2, 1], [1, 1, np.nan]], dtype=float
)
X_first_guess: NDArray = np.array(
    [[1, 4, 1], [1, 4, 3], [1, 4, 4], [1, 2, 1], [1, 1, 4]], dtype=float
)
mask: NDArray = np.isnan(X_missing)


# @pytest.fixture
def generate_multinormal_predefined_mean_cov(d=3, n=500):
    rng = np.random.default_rng(42)
    seed = rng.integers(np.iinfo(np.int32).max)
    random_state = np.random.RandomState(seed=seed)
    mean = np.array([rng.uniform(low=0, high=d) for _ in range(d)])
    covariance = make_spd_matrix(n_dim=d, random_state=random_state)
    X = rng.multivariate_normal(mean=mean, cov=covariance, size=n)
    mask = np.array(np.full_like(X, False), dtype=bool)
    for j in range(X.shape[1]):
        ind = rng.choice(
            np.arange(X.shape[0]), size=np.int64(np.ceil(X.shape[0] * 0.1)), replace=False
        )
        mask[ind, j] = True
    X_missing = X.copy()
    X_missing[mask] = np.nan
    # return {"mean": mean, "covariance": covariance, "X": X, "X_missing": X_missing}
    return X, X_missing, mean, covariance


def get_matrix_B(d, p, eigmax=1):
    rng = np.random.default_rng(42)
    B = rng.normal(0, 1, size=(d * p + 1, d))
    U, S, Vt = linalg.svd(B, check_finite=False, full_matrices=False)
    S = rng.uniform(0, eigmax, size=d)
    B = U @ (Vt * S)
    return B


def generate_varp_process(d=3, n=10000, p=1):
    rng = np.random.default_rng(42)
    seed = rng.integers(np.iinfo(np.int32).max)
    random_state = np.random.RandomState(seed=seed)
    B = get_matrix_B(d, p, eigmax=0.9)
    nu = B[0, :]
    list_A = [B[1 + lag * d : 1 + (lag + 1) * d, :] for lag in range(p)]
    S = make_spd_matrix(n_dim=d, random_state=random_state) * 1e-2
    X = np.zeros((n, d))
    U = rng.multivariate_normal(mean=np.zeros(d), cov=S, size=n)
    for i in range(n):
        X[i] = nu + U[i]
        for lag in range(p):
            A = list_A[lag].T
            X[i] += A @ X[i - lag - 1]

    mask = np.array(np.full_like(X, False), dtype=bool)
    for j in range(X.shape[1]):
        ind = rng.choice(
            np.arange(X.shape[0]), size=np.int64(np.ceil(X.shape[0] * 0.1)), replace=False
        )
        mask[ind, j] = True
    X_missing = X.copy()
    X_missing[mask] = np.nan
    return X, X_missing, B, S


@pytest.mark.parametrize(
    "A, X_first_guess, mask",
    [(A, X_first_guess, mask)],
)
def test_gradient_conjugue(
    A: NDArray,
    X_first_guess: NDArray,
    mask: NDArray,
) -> None:
    """Test the conjugate gradient algorithm."""
    X_result = em_sampler._conjugate_gradient(A, X_first_guess, mask)
    X_expected = np.array([[1, -1, 1], [1, -1, 3], [1, 4, 0], [1, 2, 1], [1, 1, 0]], dtype=float)

    np.testing.assert_allclose(X_result, X_expected, atol=1e-5)
    assert np.sum(X_result * (X_result @ A)) <= np.sum(X_first_guess * (X_first_guess @ A))
    assert np.allclose(X_first_guess[~mask], X_result[~mask])


def test_get_lag_p():
    """Test if it can retrieve the lag p"""
    X, _, _, _ = generate_varp_process(d=3, n=1000, p=2)
    varpem = em_sampler.VARpEM()
    varpem.fit(X)
    assert varpem.p == 2


def test_initialized() -> None:
    """Test that initializations do not crash."""
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
    mock_fit_parameters = mocker.patch(
        "qolmat.imputations.em_sampler.MultiNormalEM.fit_parameters"
    )
    mock_combine_parameters = mocker.patch(
        "qolmat.imputations.em_sampler.MultiNormalEM.combine_parameters"
    )
    mock_update_criteria_stop = mocker.patch(
        "qolmat.imputations.em_sampler.MultiNormalEM.update_criteria_stop"
    )
    em = em_sampler.MultiNormalEM(max_iter_em=max_iter_em)
    em.fit(X_missing)
    assert mock_sample_ou.call_count == max_iter_em
    assert mock_maximize_likelihood.call_count == 0
    assert mock_check_convergence.call_count == max_iter_em
    assert mock_fit_parameters.call_count == 1
    assert mock_combine_parameters.call_count == max_iter_em
    assert mock_update_criteria_stop.call_count == max_iter_em


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
    "list_B, list_S, logliks",
    [
        (
            [np.array([1, 2, 3, 3])] * 12,
            [np.array([1, 2, 3, 3])] * 12,
            [1] * 12,
        )
    ],
)
def test_varem_sampler_check_convergence_true(
    list_B: List[NDArray],
    list_S: List[NDArray],
    logliks: List[float],
) -> None:
    """Test the convergence criteria of the VAR1EM algorithm."""
    em = em_sampler.VARpEM(p=1, random_state=42)
    em.dict_criteria_stop["B"] = list_B
    em.dict_criteria_stop["S"] = list_S
    em.dict_criteria_stop["logliks"] = logliks
    assert em._check_convergence() == True


@pytest.mark.parametrize(
    "list_B, list_S, logliks",
    [([np.array([1, 2, 3, 3])] * 4, [np.array([1])] * 4, [1] * 4)],
)
def test_varem_sampler_check_convergence_false(
    list_B: List[NDArray],
    list_S: List[NDArray],
    logliks: List[float],
) -> None:
    """Test the non-convergence criteria of the VAR1EM algorithm."""
    em = em_sampler.VARpEM(p=1, random_state=42)
    em.dict_criteria_stop["B"] = list_B
    em.dict_criteria_stop["S"] = list_S
    em.dict_criteria_stop["logliks"] = logliks
    assert em._check_convergence() == False


def test_no_more_nan_multinormalem() -> None:
    """Test there are no more missing values after the MultiNormalEM algorithm."""
    X = np.array([[1, np.nan, 8, 1], [3, 1, 4, 2], [2, 3, np.nan, 1]], dtype=float)
    model = em_sampler.MultiNormalEM()
    X_imp = model.fit_transform(X)
    assert np.sum(np.isnan(X)) > 0
    assert np.sum(np.isnan(X_imp)) == 0


def test_no_more_nan_varpem() -> None:
    """Test there are no more missing values after the VAR1EM algorithm."""
    _, X_missing, _, _ = generate_varp_process(d=2, n=1000, p=1)
    em = em_sampler.VARpEM(p=1)
    X_imputed = em.fit_transform(X_missing)
    assert np.sum(np.isnan(X_missing)) > 0
    assert np.sum(np.isnan(X_imputed)) == 0


def test_fit_parameters_multinormalem():
    """Test the fit MultiNormalEM provides good parameters estimates (no imputation)."""
    X, X_missing, mean, covariance = generate_multinormal_predefined_mean_cov(d=2, n=10000)
    em = em_sampler.MultiNormalEM()
    em.fit_parameters(X)
    np.testing.assert_allclose(em.means, mean, atol=1e-1)
    np.testing.assert_allclose(em.cov, covariance, atol=1e-1)


def test_mean_covariance_multinormalem():
    """Test the MultiNormalEM provides good mean and covariance estimations."""
    X, X_missing, mean, covariance = generate_multinormal_predefined_mean_cov(d=2, n=1000)
    em = em_sampler.MultiNormalEM()
    X_imputed = em.fit_transform(X_missing)

    em.fit_parameters(X)
    em.fit_parameters(X_imputed)

    covariance_imputed = np.cov(X_imputed, rowvar=False)
    mean_imputed = np.mean(X_imputed, axis=0)
    np.testing.assert_allclose(em.means, mean, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(em.cov, covariance, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(mean_imputed, mean, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(covariance_imputed, covariance, rtol=1e-1, atol=1e-1)


def test_multinormal_em_minimize_llik():
    X, X_missing, mean, covariance = generate_multinormal_predefined_mean_cov(d=2, n=1000)
    imputer = em_sampler.MultiNormalEM(method="mle", random_state=11)
    X_imputed = imputer.fit_transform(X_missing)
    llikelihood_imputed = imputer.get_loglikelihood(X_imputed)
    for _ in range(10):
        Delta = imputer.rng.uniform(0, 1, size=X.shape)
        X_perturbated = X_imputed + Delta
        llikelihood_perturbated = imputer.get_loglikelihood(X_perturbated)
        assert llikelihood_perturbated < llikelihood_imputed
    X_perturbated = X
    X_perturbated[np.isnan(X)] = 0
    llikelihood_perturbated = imputer.get_loglikelihood(X_perturbated)
    assert llikelihood_perturbated < llikelihood_imputed


@pytest.mark.parametrize("method", ["sample", "mle"])
def test_multinormal_em_fit_transform(method: Literal["mle", "sample"]):
    imputer = em_sampler.MultiNormalEM(method=method, random_state=11)
    X = np.array([[1, 1, 1, 1], [np.nan, np.nan, 3, 2], [1, 2, 2, 1], [2, 2, 2, 2]])
    result = imputer.fit_transform(X)
    assert result.shape == X.shape
    np.testing.assert_allclose(result[~np.isnan(X)], X[~np.isnan(X)])


@pytest.mark.parametrize(
    "p",
    [1],
)
def test_fit_parameters_varpem(p: int):
    """Test the fit VAR(1) provides good A and B estimates (no imputation)."""
    X, X_missing, B, S = generate_varp_process(d=2, n=2000, p=p)
    em = em_sampler.VARpEM(p=p)
    em.fit_parameters(X)
    np.testing.assert_allclose(em.S, S, atol=1e-1)
    np.testing.assert_allclose(em.B, B, atol=1e-1)


@pytest.mark.parametrize(
    "p",
    [0, 1, 2],
)
def test_parameters_after_imputation_varpem(p: int):
    """Test the VAR(2) provides good A and B estimates."""
    X, X_missing, B, S = generate_varp_process(d=2, n=1000, p=p)
    em = em_sampler.VARpEM(p=p)
    X_imputed = em.fit_transform(X_missing)
    em.fit_parameters(X_imputed)
    np.testing.assert_allclose(em.B, B, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(em.S, S, rtol=1e-1, atol=1e-1)


def test_varpem_fit_transform():
    imputer = em_sampler.VARpEM(method="sample", random_state=11)
    X = np.array([[1, 1, 1, 1], [np.nan, np.nan, 3, 2], [1, 2, 2, 1], [2, 2, 2, 2]])
    result = imputer.fit_transform(X)
    expected = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.5, 3.0, 2.0],
            [1.0, 2.0, 2.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
        ]
    )
    np.testing.assert_allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize(
    "X, em, p",
    [(X_first_guess, em_sampler.MultiNormalEM(), 0), (X_first_guess, em_sampler.VARpEM(p=2), 2)],
)
def test_gradient_X_loglik(X: NDArray, em: em_sampler.EM, p: int):
    d = 3
    X, _, _, _ = generate_varp_process(d=d, n=10, p=p)
    em.fit_parameters(X)
    rng = np.random.default_rng(42)
    X0 = rng.uniform(0, 10, size=X.shape)
    # X0 = X
    loglik = em.get_loglikelihood(X0)
    grad_L = em.gradient_X_loglik(X0)
    delta = 1e-6 / np.max(np.abs(grad_L))

    U = rng.uniform(0, 1, size=X.shape)
    loglik2 = em.get_loglikelihood(X0 + delta * U)
    dL = (loglik2 - loglik) / delta
    dL_theo = (grad_L * U).sum().sum()
    np.testing.assert_allclose(dL, dL_theo, rtol=1e-1, atol=1e-1)
