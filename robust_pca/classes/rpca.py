from __future__ import annotations
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin

from robust_pca.utils import  utils


class RPCA(BaseEstimator, TransformerMixin):
    """
    This class is the root class for the RPCA methods.

    Parameters
    ----------
    period: Optional
        period/seasonality of the signal
    maxIter: int, default = 1e4
        maximum number of iterations taken for the solvers to converge
    tol: float, default = 1e-6
        tolerance for stopping criteria
    verbose: bool, default = False
    """

    def __init__(
        self,
        period: Optional[int] = None,
        maxIter: Optional[int] = int(1e4),
        tol: Optional[float] = 1e-6,
        verbose: bool = False,
    ) -> None:
        
        self.period = period
        self.maxIter = maxIter
        self.tol = tol
        self.verbose = verbose
        self.input_data = "2DArray"

    def _prepare_data(self,
                      signal: Optional[ArrayLike] = None,
                      D: Optional[np.ndarray] = None) -> None:
        """
        Prepare data fot RPCA computation:
        Transform signal to matrix if needed
        Get the omega matrix
        Impute the nan values if needed
        """
        if (D is not None) + (signal is not None) != 1:
            raise ValueError("Either D or signal should not be None")
        if (D is None):
            self.period = utils.get_period(signal) if self.period is None else self.period
            D_init, ret = utils.signal_to_matrix(self.signal, self.period)
            self.input_data = "1DArray"
            return D_init, ret if D is None else D.copy(), 0
    
    def get_params(self):
        return {
            "period": self.period,
            "maxIter": self.maxIter,
            "tol": self.tol,
            "verbose": self.verbose
        }

    def fit(
        self,
        signal: Optional[ArrayLike] = None,
        D: Optional[NDArray] = None
        ) -> RPCA:
        X, ret = self._prepare_data(signal=signal, D = D)
        X.flat[-ret:] = np.nan
        A = np.zeros(X.shape, dtype = float)
        self.X = X
        self.A = A
        return self
    def transform(self):
        if self.input_data == "2DArray":
            return self.X.copy()
        elif self.input_data == "1DArray":
            return self.X.flatten()
        else:
            raise ValueError("input data type not recognized")