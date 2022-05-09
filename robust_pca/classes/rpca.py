from __future__ import annotations
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin

from robust_pca.utils import  utils


class RPCA(BaseEstimator, TransformerMixin):
    """
    This class is the root class of the RPCA methods.

    Parameters
    ----------
    n_cols: Optional
        Number of columns in case reshaping of the 1D signal.
    maxIter: int, default = 1e4
        maximum number of iterations taken for the solvers to converge
    tol: float, default = 1e-6
        tolerance for stopping criteria
    verbose: bool, default = False
    """

    def __init__(
        self,
        n_cols: Optional[int] = None,
        maxIter: Optional[int] = int(1e4),
        tol: Optional[float] = 1e-6,
        verbose: bool = False,
    ) -> None:
        
        self.n_cols = n_cols
        self.maxIter = maxIter
        self.tol = tol
        self.verbose = verbose

    def _prepare_data(self,
                      signal: NDArray) -> None:
        """
        Prepare data fot RPCA computation:
        Transform signal to matrix if needed
        Get the omega matrix
        Impute the nan values if needed
        """
        if len(signal.shape) == 1:
            self.n_cols = utils.get_period(signal) if self.n_cols is None else self.n_cols
            D_init, ret = utils.signal_to_matrix(signal, self.n_cols)
            self.input_data = "1DArray"
        else:
            D_init = signal.copy()
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
        return self

    def fit_transform(
        self,
        signal: NDArray,
        ) -> RPCA:
        X, _ = self._prepare_data(signal=signal)
        A = np.zeros(X.shape, dtype = float)

        if self.input_data == "2DArray":
             return X, A
        elif self.input_data == "1DArray":
            return X.flatten(), A.flatten()
        else:
            raise ValueError("Data shape not recognized")