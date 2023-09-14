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
    random_state : int, optional
        The seed of the pseudo random number generator to use, for reproductibility.
    period: Optional[int]
        number of rows of the reshaped matrix if the signal is a 1D-array
    rank: Optional[int]
        (estimated) low-rank of the matrix D
    mu: Optional[float]
        Parameter for the convergence and shrinkage operator
    lam: Optional[float]
        penalizing parameter for the sparse matrix
    max_iterations: Optional[int]
        stopping criteria, maximum number of iterations. By default, the value is set to 10_000
    tol: Optional[float]
        stoppign critera, minimum difference between 2 consecutive iterations. By default,
        the value is set to 1e-6
    verbose: Optional[bool]
        verbosity level, if False the warnings are silenced
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
        """
        Get parameters for scaling in RPCA based on the input data.

        Parameters
        ----------
        D : np.ndarray
            Input data matrix of shape (m, n).

        Returns
        -------
        dict
            A dictionary containing the following parameters:
                - "mu" : float
                    Parameter for the convergence and shrinkage operator
                - "lam" : float
                    Regularization parameter for the L1 norm.

        """
        mu = D.size / (4.0 * rpca_utils.l1_norm(D))
        lam = 1 / np.sqrt(np.max(D.shape))
        dict_params = {"mu": mu, "lam": lam}
        return dict_params

    def decompose_rpca(self, D: NDArray, Omega: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Estimate the relevant parameters then compute the PCP RPCA decomposition

        Parameters
        ----------
        D : NDArray
            Matrix of the observations
        Omega: NDArray
            Matrix of missingness, with boolean data

        Returns
        -------
        M: NDArray
            Low-rank signal
        A: NDArray
            Anomalies
        """
        params_scale = self.get_params_scale(D)

        mu = params_scale["mu"] if self.mu is None else self.mu
        lam = params_scale["lam"] if self.lam is None else self.lam

        D_norm = np.linalg.norm(D, "fro")

        A = np.array(np.full_like(D, 0))
        Y = np.array(np.full_like(D, 0))

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
