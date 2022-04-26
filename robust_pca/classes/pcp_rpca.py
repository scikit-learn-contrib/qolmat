from __future__ import annotations
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robust_pca.classes.rpca import RPCA
from robust_pca.utils import  utils


class PcpRPCA(RPCA):
    """
    This class implements the basic RPCA decomposition using Alternating Lagrangian Multipliers.
    
    References
    ----------
    Candès, Emmanuel J., et al. "Robust principal component analysis." 
    Journal of the ACM (JACM) 58.3 (2011): 1-37
    
    Parameters
    ----------
    mu: Optional
        parameter for the convergence and shrinkage operator
    lam: Optional
        penalizing parameter for the sparse matrix
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

        super().__init__(period=period,
                         maxIter=maxIter,
                         tol = tol,
                         verbose = verbose)
        self.mu = mu
        self.lam = lam
    
    def get_params(self):
        dict_params = super().get_params()
        dict_params["mu"] = self.mu
        dict_params["lam"] = self.lam
        return dict_params

    def fit(
        self,
        signal: Optional[ArrayLike] = None,
        D: Optional[NDArray] = None
        ) -> PcpRPCA:
        """Compute the RPCA decomposition of a matrix based on the PCP method

        Parameters
        ----------
        signal : Optional[ArrayLike], optional
            Observations, by default None
        D: Optional
            array we want to denoise. If a signal is passed, D corresponds to that signal

        Raises
        ------
        Exception
            The user has to give either a signal, either a matrix
        """
        D_init, ret = self._prepare_data(signal = signal, D = D)
        proj_D = utils.impute_nans(D_init, method="median")

        if self.mu is None:
            self.mu = np.prod(proj_D.shape) / (
                4.0 * utils.l1_norm(self.proj_D)
            )

        if self.lam is None:
            self.lam = 1 / np.sqrt(np.max(self.proj_D.shape))

        D_norm = np.linalg.norm(proj_D, "fro")

        n, m = D_init.shape
        A = np.zeros((n, m))
        Y = np.zeros((n, m))

        errors = []
        for iteration in range(self.maxIter):
            X = utils.svd_thresholding(
                proj_D - A + Y / self.mu, 1 / self.mu
            )
            A = utils.soft_thresholding(
                proj_D - X + Y / self.mu, self.lam / self.mu
            )
            Y += self.mu * (proj_D - X - A)

            errors.append(np.linalg.norm(proj_D - X - A, "fro") / D_norm)
            if errors[-1] <= self.tol:
                if self.verbose:
                    print(f"Converged in {iteration} iterations")
                break

        X.flat[-ret:] = np.nan
        A.flat[-ret:] = np.nan
        self.X = X
        self.A = A
        self.errors = errors
        return self

    def transform(self):
        if self.input_data == "2DArray":
            return self.X.copy()
        elif self.input_data == "1DArray":
            return self.X.flatten()
        else:
            raise ValueError("input data type not recognized")



        