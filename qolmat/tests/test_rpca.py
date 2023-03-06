import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
import pytest
from qolmat.benchmark import utils
from qolmat.imputations.rpca import rpca
from typing import List, Optional


X = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
X_real = np.array([[1, 2], [1, 2], [1, 2], [1, 2], [1, 2]])
s_real = 0


@pytest.mark.parametrize('X', [X])
def test_rpca_prepare_data(X: NDArray) -> None:
    """Test prepare data function"""
    X_output, s_output = rpca.RPCA()._prepare_data(X)
    assert np.isnan(X).any() == False
    if len(X.shape) == 1:
        assert s_real == s_output
        np.testing.assert_array_equal(X_real, X_output)
    else:
        assert s_output == 0
        np.testing.assert_array_equal(X, X_output)


X_1d = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
X_1d_real1 = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
X_1d_real2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
X_1d_real3 = np.array(
    [
        [-0.4472136, 0.89442719],
        [-0.4472136, -0.2236068],
        [-0.4472136, -0.2236068],
        [-0.4472136, -0.2236068],
        [-0.4472136, -0.2236068],
    ]
)
X_1d_real4 = np.array([[-0.4472136, -0.89442719], [0.89442719, -0.4472136]])

X_2d = np.array(
    [[1, 1, 1, 1, 1, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]]
)
X_2d_real1 = np.array(
    [[1, 1, 1, 1, 1, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]]
)
X_2d_real2 = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)
X_2d_real3 = np.array([[-0.70710678, -0.70710678], [-0.70710678, 0.70710678]])
X_2d_real4 = np.array(
    [
        [-0.2, -0.2, -0.2, -0.2, -0.2, -0.4, -0.4, -0.4, -0.4, -0.4],
        [
            -0.9797959,
            0.04082483,
            0.04082483,
            0.04082483,
            0.04082483,
            0.08164966,
            0.08164966,
            0.08164966,
            0.08164966,
            0.08164966,
        ],
    ]
)

x = np.zeros((2, 3, 4))


@pytest.mark.parametrize('X', [X_1d])
def test_rpca_fit_transform(X: NDArray) -> None:
    """Test fit_transform function for rpca class"""
    if len(X.shape) == 1:
        a, b, c, d = rpca.RPCA().fit_transform(X, True)
        np.testing.assert_array_equal(X_1d_real1, a)
        np.testing.assert_array_equal(X_1d_real2, b)
        np.testing.assert_almost_equal(X_1d_real3, c)
        np.testing.assert_almost_equal(X_1d_real4, d)

    elif len(X.shape) == 2:
        a, b, c, d = rpca.RPCA().fit_transform(X, True)
        np.testing.assert_array_equal(X_2d_real1, a)
        np.testing.assert_array_equal(X_2d_real2, b)
        np.testing.assert_almost_equal(X_2d_real3, c)
        np.testing.assert_almost_equal(X_2d_real4, d)


####  There is a problem in the fucntion : self.input_data is always = "2DArray"########
# else:
# error = rpca.RPCA().fit_transform(X,False)
# print(error)
# try:
# rpca.RPCA().fit_transform(X,False)
# except ValueError("Data shape not recognized") as exc :
# assert ValueError("Data shape not recognized") == exc
