from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from sklearn import utils as sku

from qolmat.imputations.rpca import rpca_utils
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
        random_state: Union[None, int, np.random.RandomState] = None,
        period: int = 1,
        mu: Optional[float] = None,
        lam: Optional[float] = None,
        max_iterations: int = int(1e4),
        tol: float = 1e-6,
        verbose: bool = True,
    ) -> None:
        super().__init__(period=period, max_iterations=max_iterations, tol=tol, verbose=verbose)
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

        self._check_cost_function_minimized(D, M, A, Omega, lam)

        return M, A

    def _check_cost_function_minimized(
        self,
        observations: NDArray,
        low_rank: NDArray,
        anomalies: NDArray,
        Omega: NDArray,
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
        Omega: NDArrau
            boolean matrix indicating the observed values
        lam : float
            parameter penalizing the L1-norm of the anomaly/sparse part
        """
        cost_start = np.linalg.norm(observations, "nuc")
        cost_end = np.linalg.norm(low_rank, "nuc") + lam * np.sum(Omega * np.abs(anomalies))
        if self.verbose and round(cost_start, 4) - round(cost_end, 4) <= -1e-2:
            function_str = "||D||_* + lam ||A||_1"
            warnings.warn(
                f"RPCA algorithm may provide bad results. Function {function_str} increased from"
                f" {cost_start} to {cost_end} instead of decreasing!"
            )
