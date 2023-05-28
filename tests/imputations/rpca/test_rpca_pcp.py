import numpy as np
from numpy.typing import NDArray
import pytest
from qolmat.imputations.rpca.rpca_pcp import RPCAPCP

X_complete = np.array([[1, 2], [4, 4], [4, 3]])
X_incomplete = np.array([[1, np.nan], [4, 2], [np.nan, 4]])

period = 100
max_iter = 5
mu = 0.5
lam = 1


@pytest.mark.parametrize("X", [X_complete])
def test_rpca_rpca_pcp_get_params_scale(X: NDArray):
    rpca_pcp = RPCAPCP(period=period, max_iter=max_iter, mu=mu, lam=lam)
    result_dict = rpca_pcp.get_params_scale(X)
    result = list(result_dict.values())
    params_expected = [0.08333333333333333, 0.5773502691896258]
    np.testing.assert_allclose(result, params_expected, rtol=1e-5)


@pytest.mark.parametrize("X", [X_incomplete])
def test_rpca_rpca_pcp_decompose_rpca(X: NDArray):
    rpca_pcp = RPCAPCP(period=period, max_iter=max_iter, mu=mu, lam=lam)
    M_result, A_result = rpca_pcp.decompose_rpca(X)
    M_expected, A_expected = (
        np.array([[0.94183165, 0.58476157], [4.01157196, 1.99014294], [2.06642095, 3.99619333]]),
        np.array([[0.0, 2.41523843], [0.0, -0.0], [0.43357905, 0.0]]),
    )
    np.testing.assert_allclose(M_result, M_expected, rtol=1e-5)
    np.testing.assert_allclose(A_result, A_expected, rtol=1e-5)
