from __future__ import annotations


import numpy as np

from typing import Optional, Tuple
from numpy.typing import NDArray

from qolmat.imputations.rpca.rpca import RPCA
from qolmat.imputations.rpca import utils
from qolmat.utils.utils import progress_bar


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
        Parameter for the convergence and shrinkage operator
    lam: Optional
        Penalizing parameter for the sparse array
    """

    def __init__(
        self,
        period: Optional[int] = None,
        mu: Optional[float] = None,
        lam: Optional[float] = None,
        max_iter: int = int(1e4),
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> None:

        super().__init__(
            period=period,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose
        )
        self.mu = mu
        self.lam = lam

    def get_params_scale(self, X: NDArray):
        proj_D = utils.impute_nans(X, method="median")
        mu = proj_D.size / (4.0 * utils.l1_norm(proj_D))
        lam = 1 / np.sqrt(np.max(proj_D.shape))
        dict_params = {"mu": mu, "lam": lam}
        return dict_params

    def fit_transform(
        self,
        X: NDArray,
        ) -> Tuple[
            NDArray,
            NDArray,
        ]:
        """
        Compute the RPCA decomposition of a matrix based on PCP method

        Parameters
        ----------
        X: NDArray

        Returns
        -------
        M: NDArray
            Low-rank signal
        A: NDArray
            Anomalies
        """
        D_init = self._prepare_data(signal=X)
        proj_D = utils.impute_nans(D_init, method="median")

        params_scale = self.get_params_scale(X=proj_D)

        mu = params_scale["mu"] if self.mu is None else self.mu
        lam = params_scale["lam"] if self.lam is None else self.lam

        D_norm = np.linalg.norm(proj_D, "fro")

        A = np.full_like(D_init, 0)
        Y = np.full_like(D_init, 0)

        for iteration in range(self.max_iter):

            M = utils.svd_thresholding(proj_D - A + Y/mu, 1/mu)
            A = utils.soft_thresholding(proj_D - M + Y/mu, lam/mu)
            Y += mu * (proj_D - M - A)

            error = np.linalg.norm(proj_D - M - A, "fro")/D_norm

            if error < self.tol:
                if self.verbose:
                    print(f"Converged in {iteration} iterations")
                break
                    
        if len(X.shape) == 1:
            M = M.T.flatten()[:len(X)]
            A = A.T.flatten()[:len(X)]
        return M, A

    