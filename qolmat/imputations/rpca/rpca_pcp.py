from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from qolmat.imputations.rpca import utils
from qolmat.imputations.rpca.rpca import RPCA
from qolmat.utils.utils import progress_bar


class RPCAPCP(RPCA):
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
    ) -> None:
        super().__init__(
            period=period,
            max_iter=max_iter,
            tol=tol,
        )
        self.mu = mu
        self.lam = lam

    def get_params_scale(self, D: NDArray):
        mu = D.size / (4.0 * utils.l1_norm(D))
        lam = 1 / np.sqrt(np.max(D.shape))
        dict_params = {"mu": mu, "lam": lam}
        return dict_params

    def decompose_rpca(self, D: NDArray) -> Tuple[NDArray, NDArray]:
        D_proj = utils.impute_nans(D, method="median")

        params_scale = self.get_params_scale(D_proj)

        mu = params_scale["mu"] if self.mu is None else self.mu
        lam = params_scale["lam"] if self.lam is None else self.lam
        Omega = ~np.isnan(D)

        D_norm = np.linalg.norm(D, "fro")

        A: NDArray = np.full_like(D, 0)
        Y: NDArray = np.full_like(D, 0)

        errors: NDArray = np.full((self.max_iter,), fill_value=np.nan)

        M: NDArray = D_proj - A
        for iteration in range(self.max_iter):
            M = utils.svd_thresholding(D_proj - A + Y / mu, 1 / mu)
            A = utils.soft_thresholding(D_proj - M + Y / mu, lam / mu)
            A[~Omega] = (D_proj - M)[~Omega]
            Y += mu * (D_proj - M - A)

            error = np.linalg.norm(D - M - A, "fro") / D_norm
            errors[iteration] = error

            if error < self.tol:
                break
        return M, A

    def decompose_rpca_signal(
        self,
        X: NDArray,
    ) -> Tuple[NDArray, NDArray]:
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
        """
        D = self._prepare_data(X)
        M, A = self.decompose_rpca(D)

        # U, _, V = np.linalg.svd(M, full_matrices=False, compute_uv=True)

        # if X.shape[0] == 1:
        # M = M.reshape(1, -1)[:, : X.size]
        # M = M.reshape(X)
        # A = A.reshape(1, -1)[:, : X.size]
        M = M.reshape(X.shape)
        A = A.reshape(X.shape)
        return M, A
