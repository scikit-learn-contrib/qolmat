from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from sklearn import utils as sku

from qolmat.imputations.rpca import rpca_utils
from qolmat.imputations.rpca.rpca import RPCA
from qolmat.utils.exceptions import CostFunctionRPCANotMinimized


def _check_cost_function_minimized(
    observations: NDArray,
    low_rank: NDArray,
    anomalies: NDArray,
    lam: float,
):
    """Check that the functional minimized by the RPCA
    is smaller at the end than at the beginning

    Parameters
    ----------
    observations : NDArray
        observations matrix with first linear interpolation
    low_rank : NDArray
        low_rank matrix resulting from RPCA
    anomalies : NDArray
        sparse matrix resulting from RPCA
    lam : float
        parameter penalizing the L1-norm of the anomaly/sparse part

    Raises
    ------
    CostFunctionRPCANotMinimized
        The RPCA does not minimized the cost function:
        the starting cost is at least equal to the final one.
    """
    value_start = np.linalg.norm(observations, "nuc")
    value_end = np.linalg.norm(low_rank, "nuc") + lam * np.sum(np.abs(anomalies))
    if value_start + 1e-9 < value_end:
        function_str = "||D||_* + lam ||A||_1"
        raise CostFunctionRPCANotMinimized(function_str, value_start, value_end)


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
        random_state: Union[None, int, np.random.RandomState] = None,
        period: int = 1,
        mu: Optional[float] = None,
        lam: Optional[float] = None,
        max_iterations: int = int(1e4),
        tol: float = 1e-6,
    ) -> None:
        super().__init__(
            period=period,
            max_iterations=max_iterations,
            tol=tol,
        )
        self.rng = sku.check_random_state(random_state)
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

        errors: NDArray = np.full((self.max_iterations,), fill_value=np.nan)

        M: NDArray = D - A
        for iteration in range(self.max_iterations):
            M = rpca_utils.svd_thresholding(D - A + Y / mu, 1 / mu)
            A = rpca_utils.soft_thresholding(D - M + Y / mu, lam / mu)
            A[~Omega] = (D - M)[~Omega]

            Y += mu * (D - M - A)

            error = np.linalg.norm(D - M - A, "fro") / D_norm
            errors[iteration] = error

            if error < self.tol:
                break

        _check_cost_function_minimized(D, M, A, lam)

        return M, A
