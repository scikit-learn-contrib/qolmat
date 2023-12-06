from __future__ import annotations

from typing import Union, Tuple
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

    def fit_basis(self, X: NDArray) -> Self:
        """Fit RPCA model on data

        Parameters
        ----------
        X : NDArray
            Observations

        Returns
        -------
        Self
            Model RPCA
        """
        D = utils.prepare_data(X, self.period)
        Omega = ~np.isnan(D)
        D = utils.linear_interpolation(D)

        n_rows, n_cols = D.shape
        if n_rows == 1 or n_cols == 1:
            self.V = np.array([[1]])
            return self

        _, _, _, Q = self.decompose_rpca(D, Omega)

        self.Q = Q

        return self

    def decompose_on_basis(
        self, D: NDArray, Omega: NDArray, Q: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """Decompose data

        Parameters
        ----------
        D : NDArray
            Observations
        Omega : NDArray
            Boolean matrix indicating the observed values
        Q : NDArray
            Learned basis unitary array of shape (rank, n).

        Returns
        -------
        Tuple[NDArray, NDArray]
        M : np.ndarray
            Low-rank signal matrix of shape (m, n).
        A : np.ndarray
            Anomalies matrix of shape (m, n).
        """
        n_rows, n_cols = D.shape
        if n_rows == 1 or n_cols == 1:
            return D, np.full_like(D, 0)
        M, A, _, _ = self.decompose_rpca(D, Omega)
        return M, A

    def decompose_rpca_signal(self, X: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Compute the noisy RPCA with L1 or L2 time penalisation

        Parameters
        ----------
        X : NDArray
            Observations

        Returns
        -------
        M: NDArray
            Low-rank signal
        A: NDArray
            Anomalies
        """

        D = utils.prepare_data(X, self.period)
        Omega = ~np.isnan(D)
        D = utils.linear_interpolation(D)

        M, A = self.decompose_on_basis(D, Omega, self.Q)

        M_final = utils.get_shape_original(M, X.shape)
        A_final = utils.get_shape_original(A, X.shape)

        return M_final, A_final
