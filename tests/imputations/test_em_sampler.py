from typing import List

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from qolmat.imputations import em_sampler

# from __future__ import annotations

X1 = np.array([[1, 1, 1, 1], [np.nan, np.nan, 4, 2], [1, 3, np.nan, 1], [2, 2, 2, 2]])

X1_res = np.array(
    [
        [1.0, 1.0, 1.0, 1.0],
        [4.0, 4.0, 4.0, 2.0],
        [1.0, 3.0, 2.0, 1.0],
        [2.0, 2.0, 2.0, 2.0],
    ]
)


df = pd.DataFrame([[1, 1, 1, 1], [1, 2, 2, 1], [2, 2, 2, 2]])


@pytest.mark.parametrize("df", [df])
def test_em_sampler_convert_numpy(df: NDArray) -> None:
    """Test converge Numpy for Impute EM"""
    assert type(em_sampler.EM(method="sample")._convert_numpy(df)) == np.ndarray


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


# @pytest.mark.parametrize("imputations", [imputations_var])
# @pytest.mark.parametrize("mu", [mu_var])
# @pytest.mark.parametrize("cov", [cov_var])
# @pytest.mark.parametrize("n_iter", [n_iter_var])
# def test_em_sampler_check_convergence(
#    imputations: List[np.ndarray],
#    mu: List[np.ndarray],
#    cov: List[np.ndarray],
#    n_iter: int,
# ) -> None:
#    """Test check convergence for Impute EM"""
#    assert (
#        em_sampler.MultiNormalEM()._check_convergence(imputations, mu, cov, n_iter)
#        == True
#    )
