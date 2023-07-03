import numpy as np
import pytest
from numpy.typing import NDArray

from qolmat.imputations.rpca.rpca_noisy import RPCANoisy, _check_cost_function_minimized
from qolmat.utils import utils
from qolmat.utils.data import generate_artificial_ts
from qolmat.utils.exceptions import CostFunctionRPCANotMinimized

X_complete = np.array([[1, 2], [3, 1]], dtype=float)
X_incomplete = np.array([[1, 2], [3, np.nan]], dtype=float)
X_interpolated = np.array([[1, 2], [3, 3]], dtype=float)
omega = np.array([[True, True], [True, False]])
max_iter = 100
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
    "obs, lr, ano, lam, tau, norm",
    [
        (
            np.array([[1, 1], [1, 1]], dtype=float),
            np.array([[2, 2], [2, 2]], dtype=float),
            np.array([[2, 2], [2, 2]], dtype=float),
            2,
            2,
            "L2",
        )
    ],
)
def test_check_cost_function_minimized_raise_expection(
    obs: NDArray, lr: NDArray, ano: NDArray, lam: float, tau: float, norm: str
):
    function_str = "||D-M-A||_2 + tau ||D||_* + lam ||A||_2"
    with pytest.raises(
        CostFunctionRPCANotMinimized,
        match="PCA algorithm may provide bad results. "
        f"{function_str} is larger at the end "
        "of the algorithm than at the start.",
    ):
        _check_cost_function_minimized(obs, lr, ano, lam, tau, norm)


@pytest.mark.parametrize("X", [X_complete])
def test_rpca_noisy_get_params_scale(X: NDArray):
    rpca = RPCANoisy(max_iter=max_iter, tau=0.5, lam=0.1)
    result_dict = rpca.get_params_scale(X)
    result = list(result_dict.values())
    params_expected = [2, np.sqrt(2) / 2, np.sqrt(2) / 2]
    np.testing.assert_allclose(result, params_expected, rtol=1e-5)


@pytest.mark.parametrize("X, X_interpolated", [(X_incomplete, X_interpolated)])
def test_rpca_pcp_zero_tau_zero_lambda(X: NDArray, X_interpolated: NDArray):
    rpca = RPCANoisy(tau=0, lam=0, norm="L2")
    X_result, A_result = rpca.decompose_rpca_signal(X)
    np.testing.assert_allclose(X_result, X_interpolated, atol=1e-4)
    np.testing.assert_allclose(A_result, np.full_like(X, 0), atol=1e-4)


@pytest.mark.parametrize(
    "X, lam, X_interpolated",
    [(X_incomplete, 1, X_interpolated), (X_incomplete, 1e3, X_interpolated)],
)
def test_rpca_pcp_zero_tau(X: NDArray, lam: float, X_interpolated: NDArray):
    rpca = RPCANoisy(tau=0, lam=lam, norm="L2")
    X_result, A_result = rpca.decompose_rpca_signal(X)
    np.testing.assert_allclose(X_result, X_interpolated, atol=1e-4)
    np.testing.assert_allclose(A_result, np.full_like(X, 0), atol=1e-4)


@pytest.mark.parametrize(
    "X, tau, X_interpolated",
    [(X_incomplete, 0.4, X_interpolated), (X_incomplete, 2.4, X_interpolated)],
)
def test_rpca_pcp_zero_lambda(X: NDArray, tau: float, X_interpolated: NDArray):
    rpca = RPCANoisy(tau=tau, lam=0, norm="L2")
    X_result, A_result = rpca.decompose_rpca_signal(X)
    np.testing.assert_allclose(X_result, np.full_like(X, 0), atol=1e-4)
    np.testing.assert_allclose(A_result, X_interpolated, atol=1e-4)


@pytest.mark.parametrize("signal", [signal])
def test_rpca_temporal_signal(signal: NDArray):
    period = 100
    tau = 1
    lam = 0.1
    rpca = RPCANoisy(period=period, tau=tau, lam=lam, norm="L2")
    X_result, A_result = rpca.decompose_rpca_signal(signal)
    X_input_rpca = utils.linear_interpolation(signal.reshape(period, -1))
    assert np.linalg.norm(X_input_rpca, "nuc") >= 1 / 2 * np.linalg.norm(
        X_input_rpca - X_result.reshape(period, -1) - A_result.reshape(period, -1),
        "fro",
    ) ** 2 + tau * np.linalg.norm(X_result.reshape(period, -1), "nuc") + lam * np.sum(
        np.abs(A_result.reshape(period, -1))
    )
