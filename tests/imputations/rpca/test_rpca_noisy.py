import warnings

import numpy as np
import pytest
from numpy.typing import NDArray

from qolmat.imputations.rpca import rpca_utils
from qolmat.imputations.rpca.rpca_noisy import RPCANoisy
from qolmat.utils import utils
from qolmat.utils.data import generate_artificial_ts

X_complete = np.array([[1, 3], [2, 1]], dtype=float)
X_incomplete = np.array([[1, 3], [2, np.nan]], dtype=float)
X_interpolated = np.array([[1, 3], [2, 3]], dtype=float)
omega = np.array([[True, True], [True, False]])
max_iterations = 100


@pytest.fixture
def synthetic_temporal_data():
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
    return signal


@pytest.mark.parametrize(
    "obs, lr, ano, omega, lam, tau, norm",
    [
        (
            np.array([[1, 1], [1, 1]], dtype=float),
            np.array([[2, 2], [2, 2]], dtype=float),
            np.array([[2, 2], [2, 2]], dtype=float),
            True * np.ones((2, 2)),
            2,
            2,
            "L2",
        )
    ],
)
def test_check_cost_function_minimized_warning(
    obs: NDArray, lr: NDArray, ano: NDArray, omega: NDArray, lam: float, tau: float, norm: str
):
    """Test warning when the cost function is not minimized."""
    with pytest.warns(UserWarning):
        RPCANoisy()._check_cost_function_minimized(obs, lr, ano, omega, lam, tau)


@pytest.mark.parametrize(
    "obs, lr, ano, omega, lam, tau, norm",
    [
        (
            np.array([[1, 1], [1, 1]], dtype=float),
            np.array([[0, 0], [0, 0]], dtype=float),
            np.array([[0, 0], [0, 0]], dtype=float),
            True * np.ones((2, 2)),
            5,
            0,
            "L2",
        )
    ],
)
def test_check_cost_function_minimized_no_warning(
    obs: NDArray, lr: NDArray, ano: NDArray, omega: NDArray, lam: float, tau: float, norm: str
):
    """Test no warning when the cost function is minimized."""
    with warnings.catch_warnings(record=True) as record:
        RPCANoisy()._check_cost_function_minimized(obs, lr, ano, omega, lam, tau)
    assert len(record) == 0


@pytest.mark.parametrize("X", [X_complete])
def test_rpca_noisy_get_params_scale(X: NDArray):
    """Test the parameters are well scaled."""
    rpca = RPCANoisy(max_iterations=max_iterations, tau=0.5, lam=0.1)
    result_dict = rpca.get_params_scale(X)
    result = list(result_dict.values())
    params_expected = [2, np.sqrt(2) / 2, np.sqrt(2) / 2]
    np.testing.assert_allclose(result, params_expected, rtol=1e-5)


X_test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])


@pytest.mark.parametrize("norm", ["L2"])
def test_rpca_decompose_rpca_signal_shape(norm: str):
    """Test RPCA noisy results if tau and lambda equal zero."""
    rpca = RPCANoisy(rank=2, norm=norm)
    X_result, A_result = rpca.decompose_rpca_signal(X_test)
    assert X_result.shape == X_test.shape
    assert A_result.shape == X_test.shape


@pytest.mark.parametrize("X, X_interpolated", [(X_incomplete, X_interpolated)])
def test_rpca_noisy_zero_tau_zero_lambda(X: NDArray, X_interpolated: NDArray):
    """Test RPCA noisy results if tau and lambda equal zero."""
    rpca = RPCANoisy(tau=0, lam=0, norm="L2")
    X_result, A_result = rpca.decompose_rpca_signal(X)
    np.testing.assert_allclose(X_result, X_interpolated, atol=1e-4)
    np.testing.assert_allclose(A_result, np.full_like(X, 0), atol=1e-4)


@pytest.mark.parametrize(
    "X, lam, X_interpolated",
    [(X_incomplete, 1, X_interpolated), (X_incomplete, 1e3, X_interpolated)],
)
def test_rpca_noisy_zero_tau(X: NDArray, lam: float, X_interpolated: NDArray):
    """Test RPCA noisy results if tau equals zero."""
    rpca = RPCANoisy(tau=0, lam=lam, norm="L2")
    X_result, A_result = rpca.decompose_rpca_signal(X)
    np.testing.assert_allclose(X_result, X_interpolated, atol=1e-4)
    np.testing.assert_allclose(A_result, np.full_like(X, 0), atol=1e-4)


@pytest.mark.parametrize(
    "X, tau, X_interpolated",
    [(X_incomplete, 0.4, X_interpolated), (X_incomplete, 2.4, X_interpolated)],
)
def test_rpca_noisy_zero_lambda(X: NDArray, tau: float, X_interpolated: NDArray):
    """Test RPCA noisy results if lambda equals zero."""
    rpca = RPCANoisy(tau=tau, lam=0, norm="L2")
    X_result, A_result = rpca.decompose_rpca_signal(X)
    np.testing.assert_allclose(X_result, np.full_like(X, 0), atol=1e-4)
    np.testing.assert_allclose(A_result, X_interpolated, atol=1e-4)


def test_rpca_noisy_decompose_rpca(synthetic_temporal_data):
    """Test RPCA noisy results for time series data.
    Check if the cost function is smaller at the end than at the start."""
    signal = synthetic_temporal_data
    period = 100
    tau = 1
    lam = 0.1
    rank = 10
    # rpca = RPCANoisy(period=period, tau=tau, lam=lam, norm="L2")
    D = utils.prepare_data(signal, period)
    Omega = ~np.isnan(D)
    D = utils.linear_interpolation(D)

    low_rank_init = D
    anomalies_init = np.zeros(D.shape)
    cost_init = RPCANoisy.cost_function(D, low_rank_init, anomalies_init, Omega, tau, lam)

    X_result, A_result, _, _ = RPCANoisy.decompose_rpca_algorithm(D, Omega, rank, tau, lam)
    cost_result = RPCANoisy.cost_function(D, X_result, A_result, Omega, tau, lam)

    assert cost_result <= cost_init

    # assert np.linalg.norm(X_input_rpca, "nuc") >= 1 / 2 * np.linalg.norm(
    #     X_input_rpca - X_result.reshape(period, -1) - A_result.reshape(period, -1),
    #     "fro",
    # ) ** 2 + tau * np.linalg.norm(X_result.reshape(period, -1), "nuc") + lam * np.sum(
    #     np.abs(A_result.reshape(period, -1))
    # )


def test_rpca_noisy_temporal_signal_temporal_regularisations(synthetic_temporal_data):
    """Test RPCA noisy results for time series data with temporal regularisations.
    Check if the cost function is smaller at the end than at the start."""
    signal = synthetic_temporal_data
    period = 10
    tau = 1
    lam = 0.3
    rank = 10
    list_periods = [10]
    list_etas = [0.01]
    D = utils.prepare_data(signal, period)
    Omega = ~np.isnan(D)
    D = utils.linear_interpolation(D)

    low_rank_init = D
    anomalies_init = np.zeros(D.shape)

    cost_init = RPCANoisy.cost_function(
        D,
        low_rank_init,
        anomalies_init,
        Omega,
        tau,
        lam,
        list_periods=list_periods,
        list_etas=list_etas,
        norm="L2",
    )

    X_result, A_result, _, _ = RPCANoisy.decompose_rpca_algorithm(
        D, Omega, rank, tau, lam, list_periods=list_periods, list_etas=list_etas, norm="L2"
    )
    cost_result = RPCANoisy.cost_function(
        D,
        X_result,
        A_result,
        Omega,
        tau,
        lam,
        list_periods=list_periods,
        list_etas=list_etas,
        norm="L2",
    )
    assert cost_result <= cost_init
