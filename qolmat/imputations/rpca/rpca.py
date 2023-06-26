from __future__ import annotations

from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
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
        period: Optional[int] = None,
        max_iter: int = int(1e4),
        tol: float = 1e-6,
        random_state: Union[None, int, np.random.RandomState] = None,
    ) -> None:
        self.period = period
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def decompose_rpca_signal(
        self,
        X: NDArray,
    ) -> Tuple[NDArray, NDArray]:
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
        D_init = utils.prepare_data(X, self.period)
        Omega = ~np.isnan(D_init)
        # D_proj = rpca_utils.impute_nans(D_init, method="median")
        D_proj = D_init.T
        D_proj = utils.linear_interpolation(D_proj)

        # self.scaler = StandardScaler()
        # D_proj = self.scaler.fit_transform(D_proj)
        D_proj = D_proj.T

        M, A = self.decompose_rpca(D_proj, Omega)

        # M = self.scaler.inverse_transform(M.T).T
        # A = self.scaler.inverse_transform(A.T).T
        M_final = utils.get_shape_original(M, X.shape)
        A_final = utils.get_shape_original(A, X.shape)

        return M_final, A_final
