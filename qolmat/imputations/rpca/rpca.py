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
