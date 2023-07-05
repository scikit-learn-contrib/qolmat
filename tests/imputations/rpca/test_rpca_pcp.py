import numpy as np
import pytest
from numpy.typing import NDArray

from qolmat.imputations.rpca.rpca_pcp import RPCAPCP, _check_cost_function_minimized
from qolmat.utils import utils
from qolmat.utils.data import generate_artificial_ts
from qolmat.utils.exceptions import CostFunctionRPCANotMinimized

X_complete = np.array([[1, 2], [3, 1]], dtype=float)
X_incomplete = np.array([[1, 2], [3, np.nan], [np.nan, 4]], dtype=float)
max_iterations = 50
small_mu = 1e-5
large_mu = 1e5
# synthetic temporal data
n_samples = 1000
periods = [100, 20]
amp_anomalies = 0.5
ratio_anomalies = 0.05
amp_noise = 0.1
X_true, A_true, E_true = generate_artificial_ts(
    n_samples, periods, amp_anomalies, ratio_anomalies, amp_noise
)
signal = X_true + A_true + E_true
mask = np.random.choice(len(signal), round(len(signal) / 20))
signal[mask] = np.nan


@pytest.mark.parametrize(
    "obs, lr, ano, lam",
    [
        (
            np.array([[1, 1], [1, 1]], dtype=float),
            np.array([[2, 2], [2, 2]], dtype=float),
            np.array([[2, 2], [2, 2]], dtype=float),
            2,
        )
    ],
)
def test_check_cost_function_minimized_raise_expection(
    obs: NDArray, lr: NDArray, ano: NDArray, lam: float
):
    function_str = "||D||_* + lam ||A||_1"
    with pytest.raises(
        CostFunctionRPCANotMinimized,
        match="PCA algorithm may provide bad results. "
        f"{function_str} is larger at the end "
        "of the algorithm than at the start.",
    ):
        _check_cost_function_minimized(obs, lr, ano, lam)


@pytest.mark.parametrize("X", [X_complete])
def test_rpca_rpca_pcp_get_params_scale(X: NDArray):
    rpca_pcp = RPCAPCP(max_iterations=max_iterations, mu=0.5, lam=0.1)
    result_dict = rpca_pcp.get_params_scale(X)
    result = list(result_dict.values())
    params_expected = [1 / 7, np.sqrt(2) / 2]
    np.testing.assert_allclose(result, params_expected, atol=1e-4)


# The problem is ill-conditioned and the result depends on the parameter mu
@pytest.mark.parametrize("X, mu", [(X_complete, small_mu)])
def test_rpca_rpca_pcp_zero_lambda_small_mu(X: NDArray, mu: float):
    rpca_pcp = RPCAPCP(lam=0, mu=mu)
    X_result, A_result = rpca_pcp.decompose_rpca_signal(X)
    np.testing.assert_allclose(X_result, np.full_like(X, 0), atol=1e-4)
    np.testing.assert_allclose(A_result, X, atol=1e-4)


# The problem is ill-conditioned and the result depends on the parameter mu
@pytest.mark.parametrize("X, mu", [(X_complete, large_mu)])
def test_rpca_rpca_pcp_zero_lambda_large_mu(X: NDArray, mu: float):
    rpca_pcp = RPCAPCP(lam=0, mu=mu)
    X_result, A_result = rpca_pcp.decompose_rpca_signal(X)
    np.testing.assert_allclose(X_result, X, atol=1e-4)
    np.testing.assert_allclose(A_result, np.full_like(X, 0), atol=1e-4)


@pytest.mark.parametrize("X, mu", [(X_complete, large_mu)])
def test_rpca_rpca_pcp_large_lambda_small_mu(X: NDArray, mu: float):
    rpca_pcp = RPCAPCP(lam=1e3, mu=mu)
    X_result, A_result = rpca_pcp.decompose_rpca_signal(X)
    np.testing.assert_allclose(X_result, X, atol=1e-4)
    np.testing.assert_allclose(A_result, np.full_like(X, 0), atol=1e-4)


@pytest.mark.parametrize("signal", [signal])
def test_rpca_temporal_signal(signal: NDArray):
    period = 100
    lam = 0.1
    rpca = RPCAPCP(period=period, lam=lam, mu=0.01)
    X_result, A_result = rpca.decompose_rpca_signal(signal)
    X_input_rpca = utils.linear_interpolation(signal.reshape(period, -1))
    assert np.linalg.norm(X_input_rpca, "nuc") >= np.linalg.norm(
        X_result.reshape(period, -1), "nuc"
    ) + lam * np.sum(np.abs(A_result.reshape(period, -1)))
