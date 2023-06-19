import numpy as np
import pytest
from numpy.typing import NDArray

from qolmat.imputations.rpca.rpca_noisy import RPCANoisy

X_complete = np.array([[1, 2], [4, 4], [4, 3]])
X_incomplete = np.array([[1, np.nan], [4, 2], [np.nan, 4]])
Omega_1 = ~np.isnan(X_incomplete)
period = 100
max_iter = 5
tau = 0.5
lam = 1
rank = 1


@pytest.mark.parametrize("X", [X_complete])
@pytest.mark.parametrize("Omega", [Omega_1])
def test_rpca_noisy_decompose_rpca_L1(X: NDArray, Omega: NDArray):
    rpca_noisy = RPCANoisy(period=period, max_iter=max_iter, tau=tau, lam=lam)
    M_result, A_result, U_result, V_result, errors_result = rpca_noisy.decompose_rpca_L1(
        X, Omega=Omega, lam=lam, tau=tau, rank=rank
    )
    M_expected, A_expected, U_expected, V_expected, errors_expected = (
        np.array([[0.8769033, 1.8768972], [3.87689287, 3.87689287], [3.87689129, 2.8768974]]),
        np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        np.array([[49.51236273], [49.51516756], [49.5146066]]),
        np.array([[40.31557351], [40.31557349]]),
        np.array([22.99769623, 517.6202431, 748.420655, 731.25616855, 63.38889854]),
    )
    np.testing.assert_allclose(M_result, M_expected, rtol=1e-5)
    np.testing.assert_allclose(A_result, A_expected, rtol=1e-5)
    np.testing.assert_allclose(U_result, U_expected, rtol=1e-5)
    np.testing.assert_allclose(V_result, V_expected, rtol=1e-5)
    np.testing.assert_allclose(errors_result, errors_expected, rtol=1e-5)


@pytest.mark.parametrize("X", [X_complete])
@pytest.mark.parametrize("Omega", [Omega_1])
def test_rpca_noisy_decompose_rpca_L2(X: NDArray, Omega: NDArray):
    rpca_noisy = RPCANoisy(period=period, max_iter=max_iter, tau=tau, lam=lam)
    M_result, A_result, U_result, V_result, errors_result = rpca_noisy.decompose_rpca_L2(
        X, Omega=Omega, lam=lam, tau=tau, rank=rank
    )
    M_expected, A_expected, U_expected, V_expected, errors_expected = (
        np.array(
            [
                [1995.97050193, 1995.97050301],
                [1996.13508552, 1996.13508659],
                [1996.06095969, 1996.06096077],
            ]
        ),
        np.array([[0.0, 2.28659352], [0.0, 0.0], [2.28660955, 0.0]]),
        np.array([[49.51062306], [49.51470561], [49.5128669]]),
        np.array([[40.31398472], [40.31398475]]),
        np.array([22.99769623, 517.61986854, 748.4207916, 731.25607245, 63.38959319]),
    )
    np.testing.assert_allclose(M_result, M_expected, rtol=1e-5)
    np.testing.assert_allclose(A_result, A_expected, rtol=1e-5)
    np.testing.assert_allclose(U_result, U_expected, rtol=1e-5)
    np.testing.assert_allclose(V_result, V_expected, rtol=1e-5)
    np.testing.assert_allclose(errors_result, errors_expected, rtol=1e-5)


@pytest.mark.parametrize("X", [X_complete])
def test_rpca_noisy_get_params_scale(X: NDArray):
    rpca_noisy = RPCANoisy(period=period, max_iter=max_iter, tau=tau, lam=lam)
    result_dict = rpca_noisy.get_params_scale(X)
    result = list(result_dict.values())
    params_expected = [2, 1 / np.sqrt(3), 1 / np.sqrt(3)]
    np.testing.assert_allclose(result, params_expected, rtol=1e-5)
