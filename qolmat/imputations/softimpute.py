"""Script for SoftImpute class."""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn import utils as sku
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from qolmat.imputations.rpca import rpca_utils
from qolmat.utils import utils
from qolmat.utils.utils import RandomSetting

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class SoftImpute(BaseEstimator, TransformerMixin):
    """Class for the Rank Restricted Soft SVD algorithm.

    This class implements the Rank Restricted Soft SVD algorithm presented in
    Hastie, Trevor, et al. "Matrix completion and low-rank SVD
    via fast alternating least squares." The Journal of Machine Learning
    Research 16.1 (2015): 3367-3402, Algorithm 2.1
    Given X the input matrix, we solve for the following problem:
    min_A, B || Proj(X - AB')||_F^2 + tau * (|| A ||_F^2 + || B ||_F^2)

    Parameters
    ----------
    period : int
        Number of rows of the array if the array is 1D and
        reshaped into a 2D array. Corresponds to the period of the time series,
        if 1D time series is passed.
    rank : int
        Estimated rank of the matrix
    tolerance : float
        Tolerance for the convergence criterion
    tau : float
        regularisation parameter
    max_iterations : int
        Maximum number of iterations
    random_state : int, optional
        The seed of the pseudo random number generator to use,
        for reproductibility
    verbose : bool
        flag for verbosity

    Examples
    --------
    >>> import numpy as np
    >>> from qolmat.imputations.softimpute import SoftImpute
    >>> D = np.array(
    ...     [[1, 2, np.nan, 4], [1, 5, 3, np.nan], [4, 2, 3, 2], [1, 1, 5, 4]]
    ... )
    >>> Omega = ~np.isnan(D)
    >>> M, A = SoftImpute(random_state=11).decompose(D, Omega)
    >>> print(M + A)
    [[1.         2.         4.12611456 4.        ]
     [1.         5.         3.         0.87217939]
     [4.         2.         3.         2.        ]
     [1.         1.         5.         4.        ]]

    """

    def __init__(
        self,
        period: int = 1,
        rank: Optional[int] = None,
        tolerance: float = 1e-05,
        tau: Optional[float] = None,
        max_iterations: int = 100,
        random_state: RandomSetting = None,
        verbose: bool = False,
    ):
        self.period = period
        self.rank = rank
        self.tolerance = tolerance
        self.tau = tau
        self.max_iterations = max_iterations
        self.random_state = sku.check_random_state(random_state)
        self.verbose = verbose

    def get_params_scale(self, X: NDArray):
        """Get parameters for scaling in Soft Impute based on the input data.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix of shape (m, n).

        Returns
        -------
        dict
            A dictionary containing the following parameters:
                - "rank" : float
                    Rank estimate for low-rank matrix decomposition.
                - "tau" : float
                    Parameter for the nuclear norm penality

        """
        X = utils.linear_interpolation(X)
        rank = rpca_utils.approx_rank(X)
        tau = 1 / np.sqrt(np.max(X.shape))
        dict_params = {"rank": rank, "tau": tau}
        return dict_params

    def decompose(self, X: NDArray, Omega: NDArray) -> Tuple[NDArray, NDArray]:
        """Compute the Soft Impute decomposition.

        Parameters
        ----------
        X : NDArray
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
        params_scale = self.get_params_scale(X)
        rank = params_scale["rank"] if self.rank is None else self.rank
        tau = params_scale["tau"] if self.tau is None else self.tau
        if tau <= 0:
            raise ValueError(f"Parameter tau has negative value: {tau}")

        # Step 1 : Initializing
        n, m = X.shape
        V = np.zeros((m, rank))
        U = self.random_state.normal(0.0, 1.0, (n, rank))
        U, _, _ = np.linalg.svd(U, full_matrices=False)
        D = np.ones((1, rank))

        A = U * D
        B = V * D
        M = A @ B.T
        cost_start = SoftImpute.cost_function(X, M, A, Omega, tau)
        for iter_ in tqdm(
            range(self.max_iterations),
            desc="Soft Impute decomposition",
            disable=not self.verbose,
        ):
            U_old = U
            V_old = V
            D_old = D

            # Step 2 : Upate on B
            D2_invreg = (D**2 + tau) ** (-1)
            Btilde = (
                (U * D).T @ np.where(Omega, X - A @ B.T, 0) + (B * D**2).T
            ).T
            Btilde = Btilde * D2_invreg

            Utilde, D2tilde, _ = np.linalg.svd(Btilde * D, full_matrices=False)
            V = Utilde
            D = np.sqrt(D2tilde).reshape(1, -1)
            B = V * D

            # Step 3 : Upate on A
            D2_invreg = (D**2 + tau) ** (-1)
            Atilde = (
                (V * D).T @ np.where(Omega, X - A @ B.T, 0).T + (A * D**2).T
            ).T
            Atilde = Atilde * D2_invreg

            Utilde, D2tilde, _ = np.linalg.svd(Atilde * D, full_matrices=False)
            U = Utilde
            D = np.sqrt(D2tilde).reshape(1, -1)
            A = U * D

            # Step 4 : Stopping upon convergence
            ratio = SoftImpute._check_convergence(U_old, D_old, V_old, U, D, V)
            if self.verbose:
                logging.info(f"Iteration {iter_}: ratio = {round(ratio, 4)}")
                if ratio < self.tolerance:
                    logging.info(
                        f"Convergence reached at iteration {iter_} "
                        f"with ratio = {round(ratio, 4)}"
                    )
                    break

        Xstar = np.where(Omega, X - A @ B.T, 0) + A @ B.T
        M = Xstar @ V
        U, D, Rt = np.linalg.svd(M, full_matrices=False)
        D = rpca_utils.soft_thresholding(D, tau)
        M = (U * D) @ Rt @ V.T

        A = np.where(Omega, X - M, 0)

        cost_end = SoftImpute.cost_function(X, M, A, Omega, tau)
        if self.verbose and (cost_end > cost_start + 1e-9):
            warnings.warn(
                f"Convergence failed: cost function increased from"
                f" {cost_start} to {cost_end} instead of decreasing!".format(
                    "%.2f"
                )
            )

        return M, A

    @staticmethod
    def _check_convergence(
        U_old: NDArray,
        D_old: NDArray,
        V_old: NDArray,
        U: NDArray,
        D: NDArray,
        V: NDArray,
    ) -> float:
        """Check if the convergence has been reached.

        Given a pair of iterates (U_old, D_old, V_old) and (U, D, V),
        it computes the relative change in Frobenius norm given by
        || U_old @  D_old^2 @ V_old.T - U @  D^2 @ V.T ||_F^2
        / || U_old @  D_old^2 @ V_old.T ||_F^2

        Parameters
        ----------
        U_old : NDArray
            previous matrix U
        D_old : NDArray
            previous matrix D
        V_old : NDArray
            previous matrix V
        U : NDArray
            current matrix U
        D : NDArray
            current matrix D
        V : NDArray
            current matrix V

        Returns
        -------
        float
            relative change

        """
        if any(arg is None for arg in (U_old, D_old, V_old, U, D, V)):
            raise ValueError("One or more arguments are None.")

        tr_D4 = (D**4).sum()
        tr_D_old4 = (D_old**4).sum()
        DUtU = D**2 * (U.T @ U_old)
        DVtV = D_old**2 * (V_old.T @ V)
        cross_term = (DUtU @ DVtV).diagonal().sum()
        return (tr_D_old4 + tr_D4 - 2 * cross_term) / max(tr_D_old4, 1e-9)

    @staticmethod
    def cost_function(
        X: NDArray,
        M: NDArray,
        A: NDArray,
        Omega: NDArray,
        tau: float,
    ):
        """Compute cost function for different RPCA algorithm.

        Parameters
        ----------
        X : NDArray
            Matrix of observations
        M : NDArray
            Low-rank signal
        A : NDArray
            Anomalies
        Omega : NDArray
            Mask for observations
        tau: Optional[float]
            penalizing parameter for the nuclear norm

        Returns
        -------
        float
            Value of the cost function minimized by the Soft Impute algorithm

        """
        norm_frobenius = np.sum(np.where(Omega, X - M, 0) ** 2)
        norm_nuclear = np.linalg.norm(M, "nuc")
        return norm_frobenius + tau * norm_nuclear
