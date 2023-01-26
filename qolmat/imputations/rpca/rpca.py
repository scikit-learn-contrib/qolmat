from __future__ import annotations

from typing import Optional, Tuple

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
        1D and reshaped into a 2D array, by default ``None`.
    max_iter: int
        maximum number of iterations of the
        alternating direction method of multipliers,
        by default 1e4.
    tol: float
        Tolerance for stopping criteria, by default 1e-6
    verbose: bool
        default ``False``
    """

    def __init__(
        self,
        n_rows: Optional[int] = None,
        max_iter: int = int(1e4),
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> None:

        self.n_rows = n_rows
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def _prepare_data(
        self,
        signal: NDArray
    ) -> Tuple[NDArray, int]:
        """
        Transform signal to 2D-array in case of 1D-array.
        """
        if len(signal.shape) == 1:
            n_rows = utils.get_period(signal) if self.n_rows is None else self.n_rows
            D_init, n_add_values = utils.signal_to_matrix(signal, n_rows=n_rows)
            self.input_data = "1DArray"
        else:
            D_init = signal.copy()
            n_add_values = 0
        return D_init, n_add_values

    def get_params(self):
        """Return the attributes"""
        return {
            "n_rows": self.n_rows,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "verbose": self.verbose,
        }

    def set_params(self, **kargs):
        """Set the attributes"""
        for param_key in kargs.keys():
            if param_key in self.__dict__.keys():
                setattr(self, param_key, kargs[param_key])
        return self

    def fit_transform(
        self,
        X: NDArray,
        return_basis: bool = False
    ) -> RPCA:

        M, _ = self._prepare_data(signal=X)
        A = np.zeros(M.shape, dtype=float)
        
        if self.input_data == "1DArray":
            result = [M.flatten(), A.flatten()]

        else:
            result = [M, A]

        if return_basis:
            U, _, Vh = np.linalg.svd(M, full_matrices=False, compute_uv=True)
            result += [U, Vh]
        return tuple(result)
