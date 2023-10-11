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
    assert model.rank == 2
    assert model.tolerance == 1e-05


def test_initialized_custom() -> None:
    """Test that initialization does not crash and
    has custom parameters
    """
    model = softimpute.SoftImpute(period=2, rank=10)
    assert model.period == 2
    assert model.rank == 10


@pytest.mark.parametrize("X", [X])
def test_soft_impute_fit(X: NDArray) -> None:
    """Test fit instance and decomposition is computed"""
    model = softimpute.SoftImpute()
    model.fit(X)
    assert isinstance(model, softimpute.SoftImpute)
    assert model.u is not None
    assert model.d is not None
    assert model.v is not None


@pytest.mark.parametrize("X", [X])
def test_soft_impute_transform(X: NDArray) -> None:
    """Test transform shape and no more np.nan"""
    model = softimpute.SoftImpute(projected=True)
    model.fit(X)
    X_transformed = model.transform(X)
    assert X_transformed.shape == X.shape
    assert not np.any(np.isnan(X_transformed))


@pytest.mark.parametrize("X", [X])
def test_soft_impute_convergence(X: NDArray) -> None:
    """Test type of the check convergence"""
    model = softimpute.SoftImpute(projected=True)
    model.fit(X)
    U = model.u
    Dsq = model.d
    V = model.v
    ratio = model._check_convergence(U, Dsq, V, U, Dsq, V)
    assert isinstance(ratio, float)


def test_soft_impute_convergence_with_none() -> None:
    """Test check type None and raise error"""
    model = softimpute.SoftImpute(projected=True)
    with pytest.raises(ValueError):
        _ = model._check_convergence(
            None,
            np.array([1]),
            np.array([1]),
            np.array([1]),
            np.array([1]),
            np.array([1]),
        )


@pytest.mark.parametrize(
    "X, X_expected, tau, max_iterations, random_state",
    [(X_non_regression_test, X_expected, tau, max_iterations, random_state)],
)
def test_soft_impute_non_regression(
    X: NDArray, X_expected: NDArray, tau: float, max_iterations: int, random_state: int
) -> None:
    """Non regression test"""
    X_transformed = softimpute.SoftImpute(
        tau=tau, max_iterations=max_iterations, random_state=random_state
    ).fit_transform(X)
    np.testing.assert_allclose(X_transformed, X_expected, rtol=1e-3, atol=1e-3)
