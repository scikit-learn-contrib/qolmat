from typing import Any
import numpy as np
import pytest
from numpy.typing import NDArray

from qolmat.imputations import softimpute

X = np.random.rand(100, 100)
X[np.random.choice(100, 10), np.random.choice(100, 10)] = np.nan
X_non_regression_test = np.array(
    [[1, 2, np.nan, 4], [1, 5, 3, np.nan], [4, 2, 3, 2], [1, 1, 5, 4]]
)
X_expected = np.array([[1, 2, 2.9066, 4], [1, 5, 3, 2.1478], [4, 2, 3, 2], [1, 1, 5, 4]])
tau = 1
max_iterations = 30
random_state = 50


def test_initialized_default() -> None:
    """Test that initialization does not crash and
    has default parameters
    """
    model = softimpute.SoftImpute()
    assert model.period == 1
    assert model.rank is None
    assert model.tolerance == 1e-05


def test_initialized_custom() -> None:
    """Test that initialization does not crash and
    has custom parameters
    """
    model = softimpute.SoftImpute(period=2, rank=10)
    assert model.period == 2
    assert model.rank == 10
    assert model.tau is None


@pytest.mark.parametrize("X", [X])
def test_soft_impute_decompose(X: NDArray) -> None:
    """Test fit instance and decomposition is computed"""
    tau = 1
    model = softimpute.SoftImpute(tau=tau)
    Omega = ~np.isnan(X)
    X_imputed = np.where(Omega, X, 0)
    cost_all_in_M = model.cost_function(X, X_imputed, np.full_like(X, 0), Omega, tau)
    cost_all_in_A = model.cost_function(X, np.full_like(X, 0), X_imputed, Omega, tau)
    M, A = model.decompose(X, Omega)
    cost_final = model.cost_function(X, M, A, Omega, tau)
    assert isinstance(model, softimpute.SoftImpute)
    assert M.shape == X.shape
    assert A.shape == X.shape
    assert not np.any(np.isnan(M))
    assert not np.any(np.isnan(A))
    assert cost_final < cost_all_in_M
    assert cost_final < cost_all_in_A


# tests/imputations/test_imputers.py::test_sklearn_compatible_estimator


@pytest.mark.parametrize("X", [X])
def test_soft_impute_convergence(X: NDArray) -> None:
    """Test type of the check convergence"""
    model = softimpute.SoftImpute()
    M = model.random_state.uniform(size=(10, 20))
    U, D, V = np.linalg.svd(M, full_matrices=False)
    ratio = model._check_convergence(U, D, V.T, U, D, V.T)
    assert abs(ratio) < 1e-12


def test_soft_impute_convergence_with_none() -> None:
    """Test check type None and raise error"""
    model = softimpute.SoftImpute()
    with pytest.raises(ValueError):
        _ = model._check_convergence(
            None,
            np.array([1]),
            np.array([1]),
            np.array([1]),
            np.array([1]),
            np.array([1]),
        )


# @pytest.mark.parametrize(
#     "X, X_expected, tau, max_iterations, random_state",
#     [(X_non_regression_test, X_expected, tau, max_iterations, random_state)],
# )
# def test_soft_impute_non_regression(
#     X: NDArray, X_expected: NDArray, tau: float, max_iterations: int, random_state: int
# ) -> None:
#     """Non regression test"""
#     model = softimpute.SoftImpute(
#         tau=tau, max_iterations=max_iterations, random_state=random_state
#     )
#     Omega = ~np.isnan(X)
#     M, A = model.decompose(X, Omega)
#     X_result = M + A
#     np.testing.assert_allclose(X_result, X_expected, rtol=1e-3, atol=1e-3)
