from __future__ import annotations
from typing import Optional, Tuple, List

import numpy as np

from robust_pca.utils import  utils


class RPCA:
    """
    This class implements the basic RPCA decomposition using Alternating Lagrangian Multipliers.
    
    References
    ----------
    Candès, Emmanuel J., et al. "Robust principal component analysis." 
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

    def _prepare_data(self,
                      signal: Optional[List[float]] = None,
                      D: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Prepare data fot RPCA computation:
        Transform signal to matrix if needed
        Get the omega matrix
        Impute the nan values if needed
        """
        if (signal is not None) + (D is not None) !=1:
            raise Exception(
                "You have to provide either a time series (signal) or a matrix (D)"
            )

        self.ret = np.nan
        if (D is None) and (self.period is None):
            self.period = utils.get_period(signal)
        if D is None:
            D_init, self.ret = utils.signal_to_matrix(signal, self.period)
        return D_init

    def compute_rpca(self,
                     signal: Optional[List[float]] = None,
                     D: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the RPCA decomposition

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            the observed matrices, the low rank matrix and the sparse matrix
        """
        D_init = self._prepare_data(signal=signal, D=D)
        if np.any(np.isnan(D_init)):
            proj_D = utils.impute_nans(D_init, method="median")
        else:
            proj_D = D_init

        if self.mu is None:
            self.mu = np.prod(self.proj_D.shape) / (
                4.0 * utils.l1_norm(self.proj_D)
            )

        if self.lam is None:
            self.lam = 1 / np.sqrt(np.max(self.proj_D.shape))

        D_norm = np.linalg.norm(self.proj_D, "fro")

        n, m = D_init.shape
        S = np.zeros((n, m))
        Y = np.zeros((n, m))

        errors = []
        for iteration in range(self.maxIter):
            L = utils.svd_thresholding(
                proj_D - S + Y / self.mu, 1 / self.mu
            )
            S = utils.soft_thresholding(
                proj_D - L + Y / self.mu, self.lam / self.mu
            )
            Y += self.mu * (proj_D - L - S)

            errors.append(np.linalg.norm(proj_D - L - S, "fro") / D_norm)
            if errors[-1] <= self.tol:
                if self.verbose:
                    print(f"Converged in {iteration} iterations")
                break

        L = L
        S = S
        return D_init, L, S
                            
        