from __future__ import annotations

from typing import Union
from typing_extensions import Self

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

from qolmat.utils import utils


class RPCA(BaseEstimator, TransformerMixin):
    """
    This class is the root class of the RPCA methods.

    Parameters
    ----------
    period: Optional[int]
        Number of rows of the array if the array is
        1D and reshaped into a 2D array, by default `None`.
    max_iter: int
        maximum number of iterations of the
        alternating direction method of multipliers,
        by default 1e4.
    tol: float
        Tolerance for stopping criteria, by default 1e-6
    verbose: bool
        default `False`
    """

    def __init__(
        self,
        period: int = 1,
        max_iterations: int = int(1e4),
        tol: float = 1e-6,
        random_state: Union[None, int, np.random.RandomState] = None,
        verbose: bool = True,
    ) -> None:
        self.period = period
        self.max_iterations = max_iterations
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    def fit_basis(self, D: NDArray, Omega: NDArray) -> NDArray:
        n_rows, n_cols = D.shape
        if n_rows == 1 or n_cols == 1:
            self.V = np.array([[1]])
            return self

        M, A, L, Q = self.decompose_rpca(D, Omega)
        return Q

    def decompose_on_basis(self, D: NDArray, Omega: NDArray, Q: NDArray) -> NDArray:
        n_rows, n_cols = D.shape
        if n_rows == 1 or n_cols == 1:
            return D, np.full_like(D, 0)
        M, A, L, Q = self.decompose_rpca(D, Omega)
        return M, A
