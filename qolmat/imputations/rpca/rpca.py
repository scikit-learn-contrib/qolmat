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
        period: Optional[int] = None,
        max_iter: int = int(1e4),
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> None:

        self.n_rows = period
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def _prepare_data(self, X: NDArray) -> Tuple[NDArray, int, int]:
        """
        Transform signal to 2D-array in case of 1D-array.
        """
        n_rows_X, n_cols_X = X.shape
        if n_rows_X == 1:
            if self.n_rows is None:
                raise ValueError("`n_rows`must be specified when imputing 1D data.")
            D_init = utils.fold_signal(X, self.n_rows)
        else:
            D_init = X.copy()

        return D_init

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
                    f"It is not one of {', '.join(self.__dict__.keys())}",
                )
        return self
