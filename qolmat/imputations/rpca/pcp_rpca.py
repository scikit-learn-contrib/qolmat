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
        n_rows: Optional[int] = None,
        mu: Optional[float] = None,
        lam: Optional[float] = None,
        max_iter: int = int(1e4),
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> None:

        super().__init__(n_rows=n_rows, max_iter=max_iter, tol=tol, verbose=verbose)
        self.mu = mu
        self.lam = lam

    def get_params(self):
        dict_params = super().get_params()
        dict_params["mu"] = self.mu
        dict_params["lam"] = self.lam
        return dict_params

    def get_params_scale(self, X):
        D_init, _, _ = self._prepare_data(signal=X)
        proj_D = utils.impute_nans(D_init, method="median")
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
            NDArray,
            NDArray,
        ]:
        """
        Compute the RPCA decomposition of a matrix based on PCP method

        Parameters
        ----------
        X : NDArray

        Returns
        -------
        M: NDArray
            Low-rank signal
        A: NDArray
            Anomalies
        U:
            Basis Unitary array
        Vh:
            Basis Unitary array
        """
        D_init, n_add_values, input_data = self._prepare_data(signal=X)
        proj_D = utils.impute_nans(D_init, method="median")

        params_scale = self.get_params_scale(X=proj_D)

        mu = params_scale["mu"] if self.mu is None else self.mu
        lam = params_scale["lam"] if self.lam is None else self.lam

        D_norm = np.linalg.norm(proj_D, "fro")

        #n, m = D_init.shape
        A = np.full_like(D_init, 0)
        Y = np.full_like(D_init, 0)

        errors = np.full((self.max_iter,), fill_value=np.nan)

        for iteration in range(self.max_iter):

            M = utils.svd_thresholding(proj_D - A + Y/mu, 1/mu)
            A = utils.soft_thresholding(proj_D - M + Y/mu, lam/mu)
            Y += mu * (proj_D - M - A)

            error = np.linalg.norm(proj_D - M - A, "fro")/D_norm
            errors[iteration] = error

            if error < self.tol:
                if self.verbose:
                    print(f"Converged in {iteration} iterations")
                break
            
        U, _, Vh = np.linalg.svd(M, full_matrices=False, compute_uv=True)
        
        if n_add_values > 0:
            M.flat[-n_add_values:] = np.nan
            A.flat[-n_add_values:] = np.nan

        if input_data == "1DArray":
            M = M.T.flatten()
            A = A.T.flatten()
        return M, A, U, Vh, errors

    