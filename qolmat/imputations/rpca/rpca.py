from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

from qolmat.imputations.rpca import utils


class RPCA(BaseEstimator, TransformerMixin):
    """
    This class is the root class of the RPCA methods.

    Parameters
    ----------
    n_rows: Optional[int]
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
        self.n_rows = period
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _prepare_data(self, X: NDArray) -> NDArray:
        """
        Transform signal to 2D-array in case of 1D-array.
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        n_rows_X, n_cols_X = X.shape
        if n_rows_X == 1:
            if self.n_rows is None:
                raise ValueError("`n_rows` must be specified when imputing 1D data.")
            elif self.n_rows >= n_cols_X:
                raise ValueError("`n_rows` must be smaller than the signals duration.")
            return utils.fold_signal(X, self.n_rows)
        else:
            if self.n_rows is None:
                return X.copy()
            else:
                raise ValueError("`n_rows` should not be specified when imputing 2D data.")

    def get_shape_original(self, M: NDArray, X: NDArray) -> NDArray:
        """Shapes an output matrix from the RPCA algorithm into the original shape.

        Parameters
        ----------
        M : NDArray
            Matrix to reshape
        X : NDArray
            Matrix of the desired shape

        Returns
        -------
        NDArray
            Reshaped matrix
        """
        size = X.size
        M_flat = M.flatten()[:size]
        return M_flat.reshape(X.shape)
