import numpy as np
import pytest
from numpy.typing import NDArray

from qolmat.imputations.softimpute import SoftImpute


@pytest.fixture
def X_random() -> NDArray:
    """Generate random matrix with missing values."""
    rng = np.random.RandomState(42)
    X = rng.rand(100, 100)
    X[rng.choice(100, 10), rng.choice(100, 10)] = np.nan
    return X


@pytest.fixture
def X_non_regression() -> NDArray:
    """Get small test matrix for non-regression tests."""
    return np.array([[1, 2, np.nan, 4], [1, 5, 3, np.nan], [4, 2, 3, 2], [1, 1, 5, 4]])


@pytest.fixture
def X_expected() -> NDArray:
    """Get expected imputed values for non-regression test."""
    return np.array([[1, 2, 2.9066, 4], [1, 5, 3, 2.1478], [4, 2, 3, 2], [1, 1, 5, 4]])


@pytest.fixture
def default_params() -> dict:
    """Get default parameters for SoftImpute."""
    return {"tau": 1, "max_iterations": 30, "random_state": 50}


def test_initialized_default() -> None:
    """Test that initialization does not crash and has default parameters."""
    model = SoftImpute()
    assert model.period == 1
    assert model.rank is None
    assert model.tolerance == 1e-05


def test_initialized_custom() -> None:
    """Test that initialization does not crash and has custom parameters."""
    model = SoftImpute(period=2, rank=10)
    assert model.period == 2
    assert model.rank == 10
    assert model.tau is None


def test_soft_impute_decompose(X_random: NDArray, default_params: dict) -> None:
    """Test fit instance and decomposition is computed."""
    tau = default_params["tau"]
    model = SoftImpute(tau=tau)
    Omega = ~np.isnan(X_random)
    X_imputed = np.where(Omega, X_random, 0)
    cost_all_in_M = model.cost_function(X_random, X_imputed, np.full_like(X_random, 0), Omega, tau)
    cost_all_in_A = model.cost_function(X_random, np.full_like(X_random, 0), X_imputed, Omega, tau)
    M, A = model.decompose(X_random, Omega)
    cost_final = model.cost_function(X_random, M, A, Omega, tau)
    assert isinstance(model, SoftImpute)
    assert M.shape == X_random.shape
    assert A.shape == X_random.shape
    assert not np.any(np.isnan(M))
    assert not np.any(np.isnan(A))
    assert cost_final < cost_all_in_M
    assert cost_final < cost_all_in_A


def test_soft_impute_convergence() -> None:
    """Test type of the check convergence."""
    model = SoftImpute()
    M = model.random_state.uniform(size=(10, 20))
    U, D, V = np.linalg.svd(M, full_matrices=False)
    ratio = model._check_convergence(U, D, V.T, U, D, V.T)
    assert abs(ratio) < 1e-12


def test_soft_impute_convergence_with_none() -> None:
    """Test check type None and raise error."""
    model = SoftImpute()
    with pytest.raises(ValueError):
        _ = model._check_convergence(
            np.array([1]),
            np.array([1]),
            np.array([1]),
            np.array([1]),
            np.array([1]),
            np.array([1]),
        )


def test_decompose_loss_minimized(X_random: NDArray, default_params: dict) -> None:
    """Test that the loss function is at a local minimum."""
    tau = default_params["tau"]
    imputer = SoftImpute(random_state=123, tau=tau)
    Omega = ~np.isnan(X_random)
    M, A = imputer.decompose(X_random, Omega)
    X_imputed = M + A
    cost_imputed = SoftImpute.cost_function(X_imputed, M, A, Omega, tau)
    for i in range(10):
        Delta = 1.1 ** (i - 9) * imputer.random_state.uniform(0, 1, size=X_random.shape)
        X_perturbed = X_imputed + Delta
        cost_perturbed = SoftImpute.cost_function(X_perturbed, M, A, Omega, tau)
        assert cost_perturbed > cost_imputed
    M = np.zeros(X_random.shape)
    A = X_random.copy()
    cost_perturbed = SoftImpute.cost_function(X_random, M, A, Omega, tau)
    assert cost_perturbed > cost_imputed
