from __future__ import annotations
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from robust_pca.utils import  utils


class PcpRPCA:
    """This class implements the basic RPCA decomposition using Alternating Lagrangian Multipliers.
    
    References
    ----------
    Candès, Emmanuel J., et al. "Robust principal component analysis?." 
    Journal of the ACM (JACM) 58.3 (2011): 1-37
    
    Parameters
    ----------
    signal: Optional
        time series we want to denoise
    period: Optional
        period/seasonality of the signal
    D: Optional
        array we want to denoise. If a signal is passed, D corresponds to that signal
    mu: Optional
        parameter for the convergence and shrinkage operator
    lam: Optional
        penalizing parameter for the sparse matrix
    maxIter: int, default = 1e4
        maximum number of iterations taken for the solvers to converge
    tol: float, default = 1e-6
        tolerance for stopping criteria
    verbose: bool, default = False
    """

    def __init__(
        self,
        signal: Optional[List[float]] = None,
        period: Optional[int] = None,
        D: Optional[np.ndarray] = None,
        mu: Optional[float] = None,
        lam: Optional[float] = None,
        maxIter: Optional[int] = int(1e4),
        tol: Optional[float] = 1e-6,
        verbose: bool = False,
    ) -> None:

        if (signal is None) and (D is None):
            raise Exception(
                "You have to provide either a time series (signal) or a matrix (D)"
            )

        self.signal = signal
        self.period = period
        self.D = D
        self.maxIter = maxIter
        self.tol = tol
        self.verbose = verbose
        self.mu = mu
        self.lam = lam

        self._prepare_data()

    def _prepare_data(self) -> None:
        """Prepare data fot RPCA computation:
                Transform signal to matrix if needed
                Get the omega matrix
                Impute the nan values if needed
        """
        
        self.ret = 0
        if (self.D is None) and (self.period is None):
            self.period = utils.get_period(self.signal)
        if self.D is None:
            self.D, self.ret = utils.signal_to_matrix(self.signal, self.period)
        
        self.initial_D = self.D.copy()

    def compute(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the RPCA deocmposition of a matrix 

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            the observed matricx, the low rank matrix and the sparse matrix
        """

        if np.isnan(np.sum(self.D)):
            self.proj_D = utils.impute_nans(self.D, method="median")
        else:
            self.proj_D = self.D

        if self.mu is None:
            self.mu = np.prod(self.proj_D.shape) / (
                4.0 * utils.l1_norm(self.proj_D)
            )

        if self.lam is None:
            self.lam = 1 / np.sqrt(np.max(self.proj_D.shape))

        D_norm = np.linalg.norm(self.proj_D, "fro")

        n, m = self.D.shape
        S = np.zeros((n, m))
        Y = np.zeros((n, m))

        errors = []
        for iteration in range(self.maxIter):
            L = utils.svd_thresholding(
                self.proj_D - S + Y / self.mu, 1 / self.mu
            )
            S = utils.soft_thresholding(
                self.proj_D - L + Y / self.mu, self.lam / self.mu
            )
            Y += self.mu * (self.proj_D - L - S)

            errors.append(np.linalg.norm(self.proj_D - L - S, "fro") / D_norm)
            if errors[-1] <= self.tol:
                if self.verbose:
                    print(f"Converged in {iteration} iterations")
                break

        self.L = L
        self.S = S

        return self.D, L, S
