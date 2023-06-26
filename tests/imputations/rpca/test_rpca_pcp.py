import numpy as np
import pytest
from numpy.typing import NDArray

from qolmat.imputations.rpca.rpca_pcp import RPCAPCP

X_complete = np.array([[1, 2], [3, 1]], dtype=float)
X_incomplete = np.array([[1, 2], [3, np.nan], [np.nan, 4]], dtype=float)
max_iter = 50
small_mu = 1e-5
large_mu = 1e5


@pytest.mark.parametrize("X", [X_complete])
def test_rpca_rpca_pcp_get_params_scale(X: NDArray):
    rpca_pcp = RPCAPCP(max_iter=max_iter, mu=0.5, lam=0.1)
    result_dict = rpca_pcp.get_params_scale(X)
    result = list(result_dict.values())
    params_expected = [1 / 7, np.sqrt(2) / 2]
    np.testing.assert_allclose(result, params_expected, rtol=1e-5)


@pytest.mark.parametrize("X, mu", [(X_complete, small_mu)])
def test_rpca_rpca_pcp_zero_lambda_small_mu(X: NDArray, mu: float):
    rpca_pcp = RPCAPCP(lam=0, mu=mu)
    M_result, A_result = rpca_pcp.decompose_rpca_signal(X)
    np.testing.assert_allclose(X, A_result, rtol=1e-5)
    np.testing.assert_allclose(M_result, np.full_like(X, 0), rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("X, mu", [(X_complete, large_mu)])
def test_rpca_rpca_pcp_zero_lambda_large_mu(X: NDArray, mu: float):
    rpca_pcp = RPCAPCP(lam=0, mu=mu)
    M_result, A_result = rpca_pcp.decompose_rpca_signal(X)
    np.testing.assert_allclose(X, M_result, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(A_result, np.full_like(X, 0), rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("X, mu", [(X_complete, large_mu)])
def test_rpca_rpca_pcp_large_lambda_small_mu(X: NDArray, mu: float):
    rpca_pcp = RPCAPCP(lam=1e3, mu=mu)
    M_result, A_result = rpca_pcp.decompose_rpca_signal(X)
    np.testing.assert_allclose(X, M_result, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(A_result, np.full_like(X, 0), rtol=1e-5, atol=1e-4)
