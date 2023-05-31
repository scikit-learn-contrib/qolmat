import numpy as np
import pytest
from numpy.typing import NDArray

from qolmat.imputations.rpca.rpca_noisy import RPCANoisy
from qolmat.imputations.rpca.rpca_pcp import RPCAPCP

X_incomplete = np.array([[1, np.nan], [4, 2], [np.nan, 4]])
X_flat = np.array([2, np.nan, 0, 4, 3, np.nan])

X_exp_nrows_1__prepare_data = np.array([1.0, np.nan, 4.0, 2.0, np.nan, 4.0])
X_exp_nrows_6__prepare_data = np.concatenate(
    [X_incomplete.reshape(-1, 6).flatten(), np.ones((1, 94)).flatten() * np.nan]
)

period = 100
max_iter = 32
mu = 0.5
tau = 0.5
lam = 1


@pytest.mark.parametrize("X", [X_incomplete])
def test_rpca__prepare_data_2D_fail(X: NDArray):
    rpca_pcp = RPCAPCP(max_iter=max_iter, mu=mu, lam=lam, period=period)
    np.testing.assert_raises(ValueError, rpca_pcp._prepare_data, X)


@pytest.mark.parametrize("X", [X_incomplete])
def test_rpca__prepare_data_2D_succeed(X: NDArray):
    rpca_pcp = RPCAPCP(max_iter=max_iter, mu=mu, lam=lam, period=None)
    result = rpca_pcp._prepare_data(X)
    np.testing.assert_allclose(result, X)


@pytest.mark.parametrize("X", [X_incomplete])
def test_rpca__prepare_data_1D_fail(X: NDArray):
    signal = X.reshape(1, -1)  # X.shape[0] * X.shape[1])
    rpca_pcp = RPCAPCP(max_iter=max_iter, mu=mu, lam=lam, period=None)
    np.testing.assert_raises(ValueError, rpca_pcp._prepare_data, signal)


@pytest.mark.parametrize("X", [X_incomplete])
def test_rpca__prepare_data_1D_succeed(X: NDArray):
    signal = X.reshape(1, -1)  # , X.shape[0] * X.shape[1])
    rpca_pcp = RPCAPCP(max_iter=max_iter, mu=mu, lam=lam, period=3)
    result = rpca_pcp._prepare_data(signal)
    np.testing.assert_allclose(result, X)


X_exp_nrows_2_pcp_decompose_rpca_signal = np.array([2, 3, 0, 4, 3, 0])
X_exp_nrows_3_pcp_decompose_rpca_signal = np.array([2, 4, 0, 4, 3, 4])
X_exp_nrows_1_pcp_decompose_rpca_signal = np.array([[1, 3], [4, 2], [2.5, 4]])


@pytest.mark.parametrize(
    "n_rows, X, X_expected",
    [
        (2, X_flat, X_exp_nrows_2_pcp_decompose_rpca_signal),
        (3, X_flat, X_exp_nrows_3_pcp_decompose_rpca_signal),
        (4, X_flat, X_exp_nrows_3_pcp_decompose_rpca_signal),
        (None, X_incomplete, X_exp_nrows_1_pcp_decompose_rpca_signal),
    ],
)
def test_rpca_pcp_decompose_rpca_signal(n_rows: int, X: NDArray, X_expected: NDArray):
    rpca_pcp = RPCAPCP(max_iter=max_iter, mu=mu, lam=lam, period=n_rows)
    M, A = rpca_pcp.decompose_rpca_signal(X)
    result = M + A
    np.testing.assert_allclose(result, X_expected, atol=1e-3)


X_exp_L1_noisy_decompose_rpca_signal = np.array([1.844, 2.845, -0.155, 3.844, 2.845, -0.155])
X_exp_L2_noisy_decompose_rpca_signal = np.array([0, 6.498, 0, 0, 0, 6.493])


@pytest.mark.parametrize("X", [X_flat])
@pytest.mark.parametrize(
    "norm, X_expected",
    [("L1", X_exp_L1_noisy_decompose_rpca_signal), ("L2", X_exp_L2_noisy_decompose_rpca_signal)],
)
def test_rpca_noisy_decompose_rpca_signal(X: NDArray, norm: str, X_expected: NDArray):
    rpca_noisy = RPCANoisy(period=2, max_iter=max_iter, tau=tau, lam=lam, norm=norm)
    M, A = rpca_noisy.decompose_rpca_signal(X)
    result = M + A
    np.testing.assert_allclose(result, X_expected, atol=1e-3)
