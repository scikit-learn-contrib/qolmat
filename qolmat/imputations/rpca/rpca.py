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
    ) -> Tuple[NDArray, int, int]:
        """
        Transform signal to 2D-array in case of 1D-array.
        """
        if len(signal.shape) == 1:
            n_rows = utils.get_period(signal) if self.n_rows is None else self.n_rows
            D_init, n_add_values = utils.signal_to_matrix(signal, n_rows=n_rows)
        else:
            D_init = signal.copy()
            n_add_values = 0
        
        input_data = f"{len(signal.shape)}DArray"
        return D_init, n_add_values, input_data

    def get_params(self) -> dict[str, Union[int, bool]]:
        """Return the attributes"""
        return {
            "n_rows": self.n_rows,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "verbose": self.verbose,
        }

    def set_params(self, **kargs) -> RPCA:
        """Set the attributes"""
        for key, value in kargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"{key} is not a parameter of {type(self).__name__}",
                    f"It is not one of {', '.join(self.__dict__.keys())}"
                    )
        return self

    def fit_transform(
        self,
        X: NDArray,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Parameters
        ----------
        X : NDArray

        Returns
        -------
        M: NDArray
            Low-rank signal
        A: NDArray
            Anomalies
        U: NDArray
            Basis Unitary array
        V: NDArray
            Basis Unitary array

        errors: NDArray
            Array of iterative errors
        """

        M, _, input_data= self._prepare_data(signal=X)
        A = np.zeros(M.shape, dtype=float)
        
        if input_data == "1DArray":
            M, A = M.flatten(), A.flatten()

        errors = None 
        U, _, V = np.linalg.svd(M, full_matrices=False, compute_uv=True)

        return M, A, U, V, errors