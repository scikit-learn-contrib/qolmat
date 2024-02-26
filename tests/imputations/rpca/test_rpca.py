from typing import Tuple
import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from pytest_mock.plugin import MockerFixture
from qolmat.imputations.rpca.rpca import RPCA

# X_incomplete = np.array([[1, np.nan], [4, 2], [np.nan, 4]])


# X_exp_nrows_1_prepare_data = np.array([1.0, np.nan, 4.0, 2.0, np.nan, 4.0])
# X_exp_nrows_6_prepare_data = np.concatenate(
#     [X_incomplete.reshape(-1, 6).flatten(), np.ones((1, 94)).flatten() * np.nan]
# )

# period = 100
# max_iter = 256
# mu = 0.5
# tau = 0.5
# lam = 1


class RPCAMock(RPCA):
    def __init__(self):
        super().__init__()
        self.Q = None

    def decompose(self, D: NDArray, Omega: NDArray) -> Tuple[NDArray, NDArray]:
        self.call_count = 1
        return D, D


X_incomplete = np.array([[1, np.nan], [4, 2], [np.nan, 4]])
Omega = ~np.isnan(X_incomplete)


def test_rpca_init() -> None:
    rpca = RPCAMock()
    M, A = rpca.decompose(X_incomplete, Omega)
    assert rpca.call_count == 1
