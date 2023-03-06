from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
import pytest
from qolmat.benchmark import utils
from qolmat.imputations import em_sampler
from typing import List, Optional


X1 = np.array(
    [[1, 1, 1, 1], [np.nan, np.nan, 3, 2], [1, 2, 2, 1], [2, 2, 2, 2]]
)

X1_res = np.array(
    [
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 3.0, 2.0],
        [1.0, 2.0, 2.0, 1.0],
        [2.0, 2.0, 2.0, 2.0],
    ]
)


@pytest.mark.parametrize('X', [X1])
def test_em_sampler_linear_interpolation(X: NDArray) -> None:
    """Test linear_interpolation for Impute EM"""
    res = em_sampler.ImputeEM()._linear_interpolation(X)
    np.testing.assert_array_equal(res, X1_res)


df = pd.DataFrame([[1, 1, 1, 1], [1, 2, 2, 1], [2, 2, 2, 2]])


@pytest.mark.parametrize('df', [df])
def test_em_sampler_convert_numpy(df: NDArray) -> None:
    """Test converge Numpy for Impute EM"""
    assert type(em_sampler.ImputeEM()._convert_numpy(df)) == np.ndarray


imputations_var = [
    np.array([1, 2, 3, 3]),
    np.array([1, 2, 3, 3]),
    np.array([1, 2, 3, 3]),
    np.array([1, 2, 3, 3]),
    np.array([1, 2, 3, 3]),
    np.array([1, 2, 3, 3]),
]
mu_var = [
    np.array([1, 2, 3, 3]),
    np.array([1, 2, 3, 3]),
    np.array([1, 2, 3, 3]),
    np.array([1, 2, 3, 3]),
    np.array([1, 2, 3, 3]),
    np.array([1, 2, 3, 3]),
]
cov_var = [
    np.array([1, 2, 3, 3]),
    np.array([1, 2, 3, 3]),
    np.array([1, 2, 3, 3]),
    np.array([1, 2, 3, 3]),
    np.array([1, 2, 3, 3]),
    np.array([1, 2, 3, 3]),
]
n_iter_var = 11


@pytest.mark.parametrize('imputations', [imputations_var])
@pytest.mark.parametrize('mu', [mu_var])
@pytest.mark.parametrize('cov', [cov_var])
@pytest.mark.parametrize('n_iter', [n_iter_var])
def test_em_sampler_check_convergence(
    imputations: List[np.ndarray],
    mu: List[np.ndarray],
    cov: List[np.ndarray],
    n_iter: int,
) -> None:
    """Test check convergence for Impute EM"""
    assert (
        em_sampler.ImputeEM()._check_convergence(imputations, mu, cov, n_iter)
        == True
    )


X = np.array([[1, 2, 4, 5], [6, 7, 8, 9]])
X_shifted_ystd = np.array(
    [
        [1.0, 2.0, 4.0, 5.0, np.nan, np.nan, np.nan, np.nan],
        [6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 4.0, 5.0],
    ]
)
X_shifted_tmrw = np.array(
    [
        [1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        [6.0, 7.0, 8.0, 9.0, np.nan, np.nan, np.nan, np.nan],
    ]
)


@pytest.mark.parametrize('X', [X])
def test_em_sampler_add_shift(X: NDArray) -> None:
    """Test add shift for Impute EM"""

    X_yest_calculated = em_sampler.ImputeEM()._add_shift(X, True, False)
    X_tmrw_calculated = em_sampler.ImputeEM()._add_shift(X, False, True)
    np.testing.assert_array_equal(X_yest_calculated, X_shifted_ystd)
    np.testing.assert_array_equal(X_tmrw_calculated, X_shifted_tmrw)


X1 = np.array(
    [[1, 1, 1, 1], [np.nan, np.nan, 3, 2], [1, 2, 2, 1], [2, 2, 2, 2]]
)
X1_res = np.array(
    [
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 3.0, 2.0],
        [1.0, 2.0, 2.0, 1.0],
        [2.0, 2.0, 2.0, 2.0],
    ]
)


# @pytest.mark.parametrize("X",[X])
# def test_emr_sampler_em_mle(X:NDArray) -> None:
#    print("*********************")
#    print(em_sampler.ImputeEM()._em_mle(X))
