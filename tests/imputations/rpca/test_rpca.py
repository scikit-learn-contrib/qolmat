import numpy as np
from numpy.typing import NDArray
import pytest
from qolmat.imputations.rpca.rpca_pcp import RPCAPCP
from qolmat.imputations.rpca.rpca_noisy import RPCANoisy

X_incomplete = np.array([[1, np.nan], [4, 2], [np.nan, 4]])

X_exp_nrows_1__prepare_data = np.array([1.0, np.nan, 4.0, 2.0, np.nan, 4.0])
X_exp_nrows_6__prepare_data = np.concatenate(
    [X_incomplete.reshape(-1, 6).flatten(), np.ones((1, 94)).flatten() * np.nan]
)

X_exp_nrows_1_pcp_fit_transform = np.array(
    [0.99221825, 2.01197508, 3.96887301, 1.9844365, 2.01197508, 3.96887301]
)
X_exp_nrows_6_pcp_fit_transform = np.array([0.99665, 0.720095, 3.9866, 1.9933, 0.720095, 3.9866])

X_exp_L1_noisy_fit_transform = np.array(
    [0.98726854, 2.9872605, 3.98725648, 1.98726452, 2.9872605, 3.98725648]
)
X_exp_L2_noisy_fit_transform = np.array(
    [0.00158078, 0.00158114, 0.00158227, 0.00158127, 0.00158114, 0.00158227]
)

period = 100
max_iter = 5
mu = 0.5
tau = 0.5
lam = 1


@pytest.mark.parametrize("X", [X_incomplete])
@pytest.mark.parametrize(
    "n_rows, X_expected", [(1, X_exp_nrows_1__prepare_data), (6, X_exp_nrows_6__prepare_data)]
)
def test_rpca__prepare_data(X: NDArray, n_rows: int, X_expected: NDArray):

    signal = X.reshape(-1, X.shape[0] * X.shape[1])
    rpca_pcp = RPCAPCP(period=None, max_iter=max_iter, mu=mu, lam=lam)
    np.testing.assert_raises(ValueError, rpca_pcp._prepare_data, signal)

    signal = X.reshape(-1, n_rows)
    rpca_pcp = RPCAPCP(period=period, max_iter=max_iter, mu=mu, lam=lam)
    result = rpca_pcp._prepare_data(signal)
    result = result.flatten()
    np.testing.assert_allclose(result, X_expected, rtol=1e-5)


@pytest.mark.parametrize("X", [X_incomplete])
@pytest.mark.parametrize(
    "n_rows, X_expected",
    [(1, X_exp_nrows_1_pcp_fit_transform), (6, X_exp_nrows_6_pcp_fit_transform)],
)
def test_rpca_pcp_fit_transform(X: NDArray, n_rows: int, X_expected: NDArray):
    signal = X.reshape(-1, n_rows)
    rpca_pcp = RPCAPCP(period=period, max_iter=max_iter, mu=mu, lam=lam)
    result = rpca_pcp.fit_transform(signal)
    result = result.flatten()
    np.testing.assert_allclose(result, X_expected, rtol=1e-5)


@pytest.mark.parametrize("X", [X_incomplete])
@pytest.mark.parametrize(
    "norm, X_expected",
    [("L1", X_exp_L1_noisy_fit_transform), ("L2", X_exp_L2_noisy_fit_transform)],
)
def test_rpca_noisy_fit_transform(X: NDArray, norm: str, X_expected: NDArray):
    signal = X.reshape(-1, 1)
    rpca_noisy = RPCANoisy(period=period, max_iter=max_iter, tau=tau, lam=lam, norm=norm)
    result = rpca_noisy.fit_transform(signal)
    result = result.flatten()
    np.testing.assert_allclose(result, X_expected, rtol=1e-5)
