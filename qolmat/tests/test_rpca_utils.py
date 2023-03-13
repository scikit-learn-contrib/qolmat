import numpy as np

from qolmat.utils import utils


def test_get_period() -> None:
    signal = [1, 2, 3, 4, 5] * 5
    period1 = utils.get_period(signal)
    signal = [1, 3, 4, 1, 3, 5] * 5
    period2 = utils.get_period(signal)
    assert period1 == 5, 'test failed'
    assert period2 == 6, 'test failed'


def test_signal_to_matrix() -> None:
    signal = [1, 2, 3, 4, 5] * 5
    mat1, nb_add_val1 = utils.signal_to_matrix(signal, 5)
    mat_rep1 = np.array(
        [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
        ]
    )
    mat2, nb_add_val2 = utils.signal_to_matrix(signal[:-3], 5)
    mat_rep2 = np.array(
        [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, np.nan, np.nan, np.nan],
        ]
    )
    assert (mat1.all(), nb_add_val1) == (mat_rep1.all(), 0), 'test failed'
    assert (mat2.all(), nb_add_val2) == (mat_rep2.all(), 3), 'test failed'


def test_impute_nans() -> None:
    mat1 = np.asarray([[1, 2], [4, np.nan], [np.nan, 5], [1, 2]])
    rep1 = utils.impute_nans(mat1, method='mean')
    rep2 = utils.impute_nans(mat1, method='median')
    val1 = np.asarray([[1, 2], [4, 3], [2, 5], [1, 2]])
    val2 = np.asarray([[1, 2], [4, 2], [1, 5], [1, 2]])
    assert rep1.all() == val1.all(), 'test failed'
    assert rep2.all() == val2.all(), 'test failed'


def test_ortho_proj() -> None:
    mat1 = np.asarray([[1, 2], [4, np.nan], [np.nan, 5], [1, 2]])
    mat2 = np.asarray([[1, 2], [4, 8], [2, 5], [1, 2]])
    omega = 1 - (mat1 != mat1)
    rep1 = utils.ortho_proj(mat2, omega)
    val1 = np.asarray([[1, 2], [4, 0], [0, 5], [1, 2]])
    rep2 = utils.ortho_proj(mat1, omega, inv=1)
    val2 = np.asarray([[0, 0], [0, 8], [2, 0], [0, 0]])
    assert rep1.all() == val1.all(), 'test failed'
    assert rep2.all() == val2.all(), 'test failed'


def test_toeplitz_matrix() -> None:
    T = 1
    dim = 5
    rep1 = utils.toeplitz_matrix(T, dim, model='row')
    val1 = np.array(
        [
            [1, -1, 0, 0, 0],
            [0, 1, -1, 0, 0],
            [0, 0, 1, -1, 0],
            [0, 0, 0, 1, -1],
        ]
    )
    rep2 = utils.toeplitz_matrix(T, dim, model='column')
    val2 = np.array(
        [
            [-1, 0, 0, 0],
            [1, -1, 0, 0],
            [0, 1, -1, 0],
            [0, 0, 1, -1],
            [0, 0, 0, 1],
        ]
    )
    T = 2
    rep3 = utils.toeplitz_matrix(T, dim, model='row')
    val3 = np.array([[1, 0, -1, 0, 0], [0, 1, 0, -1, 0], [0, 0, 1, 0, -1]])
    rep4 = utils.toeplitz_matrix(T, dim, model='column')
    val4 = np.array([[-1, 0, 0], [0, -1, 0], [1, 0, -1], [0, 1, 0], [0, 0, 1]])
    assert rep1.all() == val1.all(), 'test failed'
    assert rep2.all() == val2.all(), 'test failed'
    assert rep3.all() == val3.all(), 'test failed'
    assert rep4.all() == val4.all(), 'test failed'


def test_construct_graph() -> None:
    X = np.asarray([[0], [3], [1], [2], [5]])
    nb_nbg = 2
    rep = utils.construct_graph(X, n_neighbors=nb_nbg)
    val = np.asarray(
        [
            [0, 0, 0.3679, 0.1353, 0],
            [0, 0, 0.1353, 0.3679, 0],
            [0.3679, 0, 0, 0.3679, 0],
            [0, 0.3678, 0.3679, 0, 0],
            [0, 0.1353, 0, 0.0498, 0],
        ]
    )
    np.testing.assert_array_almost_equal(rep, val, decimal=4)


def test_get_laplacian() -> None:
    G = np.arange(5) * np.arange(5)[:, np.newaxis]
    G = G[1:, 1:]
    rep = utils.get_laplacian(G)
    val = np.asarray(
        [
            [1, -0.1667, -0.2182, -0.2722],
            [-0.1667, 1, -0.3273, -0.4082],
            [-0.2182, -0.3273, 1, -0.5345],
            [-0.2722, -0.4082, -0.5345, 1],
        ]
    )
    np.testing.assert_array_almost_equal(rep, val, decimal=4)


def test_get_anomaly() -> None:
    X = np.asarray([[5.5, 3, 5.5], [6, 20, 6], [7, 10, 7]])
    A = np.asarray([[4, 1, 1], [9, 2, 6], [6, 1, 3]])
    rep_ano1, rep_noise1 = utils.get_anomaly(A, X, e=1)
    rep_ano2, rep_noise2 = utils.get_anomaly(A, X, e=3)
    val_ano1 = np.asarray([[2, 0, 1], [1, 0, 1], [3, 0, 3]])
    val_noise1 = np.asarray([[0, 9, 0], [0, 2, 0], [0, 1, 0]])
    val_ano2 = np.asarray([[0, 0, 0], [0, 0, 0], [3, 0, 3]])
    val_noise2 = np.asarray([[2, 9, 1], [1, 2, 1], [0, 1, 0]])
    assert rep_ano1.all() == val_ano1.all(), 'test failed'
    assert rep_ano2.all() == val_ano2.all(), 'test failed'
    assert rep_noise1.all() == val_noise1.all(), 'test failed'
    assert rep_noise2.all() == val_noise2.all(), 'test failed'
