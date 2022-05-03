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
            D_init, ret = utils.signal_to_matrix(signal, self.period)
            self.input_data = "1DArray"
        else:
            D_init = D
            ret = 0
        return D_init, ret
    
    def get_params(self):
        return {
            "period": self.period,
            "maxIter": self.maxIter,
            "tol": self.tol,
            "verbose": self.verbose
        }

    def set_params(self, **kargs):
        for param_key in kargs.keys():
            if param_key in RPCA.__dict__.keys():
                setattr(self, param_key, kargs[param_key]) 

    def fit_transform(
        self,
        signal: Optional[ArrayLike] = None,
        D: Optional[NDArray] = None
        ) -> RPCA:
        X, _ = self._prepare_data(signal=signal, D = D)
        A = np.zeros(X.shape, dtype = float)

        if self.input_data == "2DArray":
             return X, A
        elif self.input_data == "1DArray":
            return X.flatten(), A.flatten()
        else:
            raise ValueError("Data shape not recognized")