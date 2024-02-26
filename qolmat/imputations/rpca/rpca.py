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
    max_iter: int
        maximum number of iterations of the
        alternating direction method of multipliers,
        by default 1e4.
    tolerance: float
        Tolerance for stopping criteria, by default 1e-6
    verbose: bool
        default `False`
    """

    def __init__(
        self,
        max_iterations: int = int(1e4),
        tolerance: float = 1e-6,
        random_state: Union[None, int, np.random.RandomState] = None,
        verbose: bool = True,
    ) -> None:
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.verbose = verbose
