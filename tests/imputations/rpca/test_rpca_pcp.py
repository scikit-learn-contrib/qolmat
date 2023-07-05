import numpy as np
from numpy.typing import NDArray
import pytest
from qolmat.imputations.rpca.rpca_pcp import RPCAPCP

X_complete = np.array([[1, 2], [4, 4], [4, 3]])
X_incomplete = np.array([[1, np.nan], [4, 2], [np.nan, 4]])

period = 1
max_iterations = 128
mu = 0.5
lam = 1


@pytest.mark.parametrize("X", [X_complete])
def test_rpca_rpca_pcp_get_params_scale(X: NDArray):
    rpca_pcp = RPCAPCP(period=period, max_iterations=max_iterations, mu=mu, lam=lam)
    result_dict = rpca_pcp.get_params_scale(X)
    result = list(result_dict.values())
    params_expected = [0.08333333333333333, 0.5773502691896258]
    np.testing.assert_allclose(result, params_expected, rtol=1e-5)


@pytest.mark.parametrize("X", [X_incomplete])
def test_rpca_rpca_pcp_decompose_rpca(X: NDArray):
    rpca_pcp = RPCAPCP(period=period, max_iterations=max_iterations, mu=mu, lam=lam)
    M_result, A_result = rpca_pcp.decompose_rpca_signal(X)
    M_expected = np.array([[1, 0.5], [4, 2], [2.06, 4]])
    A_expected = np.array([[0, 0.5], [0, 0], [1.94, 0]])
    np.testing.assert_allclose(M_result, M_expected, atol=1e-2)
    np.testing.assert_allclose(A_result, A_expected, atol=1e-2)
