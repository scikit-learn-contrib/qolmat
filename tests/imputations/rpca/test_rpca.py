from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from qolmat.imputations.rpca.rpca import RPCA


class RPCAMock(RPCA):
    """Mock for RPCA."""

    def __init__(self):
        """Mock for init RPCA."""
        super().__init__()
        self.Q = None

    def decompose(self, D: NDArray, Omega: NDArray) -> Tuple[NDArray, NDArray]:
        """Mock for decompose function."""
        self.call_count = 1
        return D, D


X_incomplete = np.array([[1, np.nan], [4, 2], [np.nan, 4]])
Omega = ~np.isnan(X_incomplete)


def test_rpca_init() -> None:
    rpca = RPCAMock()
    M, A = rpca.decompose(X_incomplete, Omega)
    assert rpca.call_count == 1
