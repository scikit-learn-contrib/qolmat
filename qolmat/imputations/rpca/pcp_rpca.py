from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from qolmat.imputations.rpca import utils
from qolmat.imputations.rpca.rpca import RPCA
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

    def get_params(self):
        dict_params = super().get_params()
        dict_params["mu"] = self.mu
        dict_params["lam"] = self.lam
        return dict_params

    def get_params_scale(self, D):
        print(D.size, D.shape)
        mu = D.size / (4.0 * utils.l1_norm(D))
        lam = 1 / np.sqrt(np.max(D.shape))
        dict_params = {"mu": mu, "lam": lam}
        return dict_params
    
    def decompose_rpca(self, D: NDArray, mu:float, lam: float) -> Tuple[NDArray, NDArray]:
        D_norm = np.linalg.norm(D, "fro")

        A = np.full_like(D, 0)
        Y = np.full_like(D, 0)

        errors = np.full((self.max_iter,), fill_value=np.nan)

        for iteration in range(self.max_iter):

            M = utils.svd_thresholding(D - A + Y/mu, 1/mu)
            A = utils.soft_thresholding(D - M + Y/mu, lam/mu)
            Y += mu * (D - M - A)

            error = np.linalg.norm(D - M - A, "fro")/D_norm
            errors[iteration] = error

            if error < self.tol:
                print(iteration, ":", error, "vs", self.tol)
                if self.verbose:
                    print(f"Converged in {iteration} iterations")
                break
        return M, A


    def fit_transform(
        self,
        X: NDArray,
        ) -> NDArray:
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
        U: NDArray
            Basis Unitary array
        V: NDArray
            Basis Unitary array

        errors: NDArray
            Array of iterative errors
        """
        X = X.copy().T
        D_init = self._prepare_data(X)
        print("D_init")
        print(D_init.shape)
        proj_D = utils.impute_nans(D_init, method="median")

        params_scale = self.get_params_scale(proj_D)

        mu = params_scale["mu"] if self.mu is None else self.mu
        lam = params_scale["lam"] if self.lam is None else self.lam

        print("mu:", mu)
        print("lam:", lam)

        M, A = self.decompose_rpca(proj_D, mu, lam)
            
        # U, _, V = np.linalg.svd(M, full_matrices=False, compute_uv=True)
        
        print("end")
        print(M.shape)
        if X.shape[0] == 1:
            M = M.reshape(1, -1)[:, :X.size]
            A = A.reshape(1, -1)[:, :X.size]
        M = M.T
        A = A.T
        # return M, A, U, V, errors
        print(M.shape)
        return M

    