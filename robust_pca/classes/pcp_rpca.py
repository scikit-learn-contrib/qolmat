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
    period: Optional
        period/seasonality of the signal
    
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
        period: Optional[int] = None,
        mu: Optional[float] = None,
        lam: Optional[float] = None,
        maxIter: Optional[int] = int(1e4),
        tol: Optional[float] = 1e-6,
        verbose: bool = False,
    ) -> None:
        
        self.period = period
        self.maxIter = maxIter
        self.tol = tol
        self.verbose = verbose
        self.mu = mu
        self.lam = lam

    def _prepare_data(self) -> None:
        """Prepare data fot RPCA computation:
                Transform signal to matrix if needed
                Get the omega matrix
                Impute the nan values if needed
        """
        
        self.rest = 0
        if (self.D is None) and (self.period is None):
            self.period = utils.get_period(self.signal)
        if self.D is None:
            self.D, self.rest = utils.signal_to_matrix(self.signal, self.period)
        
        self.initial_D = self.D.copy()

    def fit(
        self,
        signal: Optional[List[float]] = None,
        D: Optional[np.ndarray] = None
        ) -> None:
        """Compute the RPCA decomposition of a matrix based on the PCP method

        Parameters
        ----------
        signal : Optional[List[float]], optional
            list of observations, by default None
        D: Optional
            array we want to denoise. If a signal is passed, D corresponds to that signal

        Raises
        ------
        Exception
            The user has to give either a signal, either a matrix
        """
        
        if (signal is None) and (D is None):
            raise Exception(
                "You have to provide either a time series (signal) or a matrix (D)"
            )
            
        self.signal = signal
        self.D = D
        
        self._prepare_data()

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
        A = np.zeros((n, m))
        Y = np.zeros((n, m))

        errors = []
        for iteration in range(self.maxIter):
            X = utils.svd_thresholding(
                self.proj_D - A + Y / self.mu, 1 / self.mu
            )
            A = utils.soft_thresholding(
                self.proj_D - X + Y / self.mu, self.lam / self.mu
            )
            Y += self.mu * (self.proj_D - X - A)

            errors.append(np.linalg.norm(self.proj_D - X - A, "fro") / D_norm)
            if errors[-1] <= self.tol:
                if self.verbose:
                    print(f"Converged in {iteration} iterations")
                break

        self.X = X
        self.A = A
        self.errors = errors

        return None
    
    def get_params(self):
        return {
            "period": self.period,
            "mu": self.mu,
            "lam": self.lam,
            "maxIter": self.maxIter,
            "tol": self.tol,
            "verbose": self.verbose
        }
