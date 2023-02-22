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

        self.period = period
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
            if self.period is None:
                raise ValueError("``period`` argument should not be ``None`` when X is 1D")
            D_init = utils.signal_to_matrix(signal, n_rows=self.period)
        elif len(signal.shape) == 2:
            if self.period is not None:
                raise ValueError("``period`` argument should be ``None`` when X is 2D")
            D_init = signal.copy()
        else:
            raise ValueError("signal should be a 1D or 2D array.")
        return D_init

    def _set_params(self, **kargs) -> RPCA:
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