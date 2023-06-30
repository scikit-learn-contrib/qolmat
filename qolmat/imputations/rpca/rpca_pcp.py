from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from qolmat.imputations.rpca import rpca_utils
from qolmat.utils import utils
from qolmat.imputations.rpca.rpca import RPCA


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
        period: int = 1,
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
        mu = D.size / (4.0 * rpca_utils.l1_norm(D))
        lam = 1 / np.sqrt(np.max(D.shape))
        dict_params = {"mu": mu, "lam": lam}
        return dict_params

    def decompose_rpca(self, D: NDArray, Omega: NDArray) -> Tuple[NDArray, NDArray]:
        params_scale = self.get_params_scale(D)

        mu = params_scale["mu"] if self.mu is None else self.mu
        lam = params_scale["lam"] if self.lam is None else self.lam

        D_norm = np.linalg.norm(D, "fro")

        A: NDArray = np.full_like(D, 0)
        Y: NDArray = np.full_like(D, 0)

        errors: NDArray = np.full((self.max_iter,), fill_value=np.nan)

        M: NDArray = D - A
        for iteration in range(self.max_iter):
            M = rpca_utils.svd_thresholding(D - A + Y / mu, 1 / mu)
            A = rpca_utils.soft_thresholding(D - M + Y / mu, lam / mu)
            A[~Omega] = (D - M)[~Omega]
            Y += mu * (D - M - A)

            error = np.linalg.norm(D - M - A, "fro") / D_norm
            errors[iteration] = error

            if error < self.tol:
                break
        return M, A
