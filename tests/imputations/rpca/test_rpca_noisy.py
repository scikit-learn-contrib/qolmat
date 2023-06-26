import numpy as np
import pytest
from numpy.typing import NDArray

from qolmat.imputations.rpca.rpca_noisy import RPCANoisy
from qolmat.utils import utils

X_complete = np.array([[1, 2], [3, 1]], dtype=float)
X_incomplete = np.array([[1, 2], [3, np.nan]], dtype=float)
omega = np.array([[True, True], [True, False]])
max_iter = 10


@pytest.mark.parametrize("X", [X_complete])
def test_rpca_rpca_noisy_get_params_scale(X: NDArray):
    rpca = RPCANoisy(max_iter=max_iter, tau=0.5, lam=0.1)
    result_dict = rpca.get_params_scale(X)
    result = list(result_dict.values())
    params_expected = [2, np.sqrt(2) / 2, np.sqrt(2) / 2]
    np.testing.assert_allclose(result, params_expected, rtol=1e-5)


@pytest.mark.parametrize("X", [X_incomplete])
def test_rpca_rpca_pcp_zero_tau_zero_lambda(X: NDArray):
    rpca = RPCANoisy(tau=0, lam=0, norm="L2")
    M_result, A_result = rpca.decompose_rpca_signal(X)
    X_expected = utils.linear_interpolation(X.T).T
    np.testing.assert_allclose(X_expected, M_result, rtol=1e-5)
    np.testing.assert_allclose(A_result, np.full_like(X, 0), rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("X, lam", [(X_incomplete, 1), (X_incomplete, 1e3)])
def test_rpca_rpca_pcp_zero_tau(X: NDArray, lam: float):
    rpca = RPCANoisy(tau=0, lam=lam, norm="L2")
    M_result, A_result = rpca.decompose_rpca_signal(X)
    X_expected = utils.linear_interpolation(X.T).T
    np.testing.assert_allclose(X_expected, M_result, rtol=1e-5)
    np.testing.assert_allclose(A_result, np.full_like(X, 0), rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("X, omega, tau", [(X_incomplete, omega, 0.4), (X_incomplete, omega, 2.4)])
def test_rpca_rpca_pcp_zero_lambda(X: NDArray, omega: NDArray, tau: float):
    rpca = RPCANoisy(tau=tau, lam=0, norm="L2")
    M_result, A_result = rpca.decompose_rpca_signal(X)
    if tau <= 2:
        X_expected = np.array([[0, 0], [0, 2 - tau]], dtype=float)
    else:
        X_expected = np.full_like(X_incomplete, 0)
    np.testing.assert_allclose(M_result, X_expected, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(A_result[omega], X[omega], rtol=1e-2, atol=1e-2)
