from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import scipy as scp
from scipy.sparse import dok_matrix, identity
from scipy.sparse.linalg import spsolve
from numpy.typing import NDArray
from sklearn import utils as sku

from qolmat.imputations.rpca import rpca_utils
from qolmat.imputations.rpca.rpca import RPCA
from qolmat.utils import utils


class RpcaNoisy(RPCA):
    """
    This class implements a noisy version of the so-called 'improved RPCA'

    References
    ----------
    Wang, Xuehui, et al. "An improved robust principal component analysis model for anomalies
    detection of subway passenger flow."
    Journal of advanced transportation (2018).

    Chen, Yuxin, et al. "Bridging convex and nonconvex optimization in robust PCA: Noise, outliers
    and missing data."
    The Annals of Statistics 49.5 (2021): 2948-2971.

    Parameters
    ----------
    random_state : int, optional
        The seed of the pseudo random number generator to use, for reproductibility.
    rank: Optional[int]
        Upper bound of the rank to be estimated
    mu: Optional[float]
        initial stiffness parameter for the constraint M = L Q
    tau: Optional[float]
        penalizing parameter for the nuclear norm
    lam: Optional[float]
        penalizing parameter for the sparse matrix
    list_periods: Optional[List[int]]
        list of periods, linked to the Toeplitz matrices
    list_etas: Optional[List[float]]
        list of penalizing parameters for the corresponding period in list_periods
    max_iterations: Optional[int]
        stopping criteria, maximum number of iterations. By default, the value is set to 10_000
    tolerance: Optional[float]
        stoppign critera, minimum difference between 2 consecutive iterations. By default,
        the value is set to 1e-6
    norm: Optional[str]
        error norm, can be "L1" or "L2". By default, the value is set to "L2"
    verbose: Optional[bool]
        verbosity level, if False the warnings are silenced
    """

    def __init__(
        self,
        random_state: Union[None, int, np.random.RandomState] = None,
        rank: Optional[int] = None,
        mu: Optional[float] = None,
        tau: Optional[float] = None,
        lam: Optional[float] = None,
        list_periods: List[int] = [],
        list_etas: List[float] = [],
        max_iterations: int = int(1e4),
        tolerance: float = 1e-6,
        norm: str = "L2",
        verbose: bool = True,
    ) -> None:
        super().__init__(max_iterations=max_iterations, tolerance=tolerance, verbose=verbose)
        self.rng = sku.check_random_state(random_state)
        self.rank = rank
        self.mu = mu
        self.tau = tau
        self.lam = lam
        self.list_periods = list_periods
        self.list_etas = list_etas
        self.norm = norm

    def get_params_scale(self, D: NDArray) -> Dict[str, float]:
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
                - "rank" : int
                    Rank estimate for low-rank matrix decomposition.
                - "tau" : float
                    Regularization parameter for the temporal correlations.
                - "lam" : float
                    Regularization parameter for the L1 norm.

        """
        rank = rpca_utils.approx_rank(D)
        tau = 1.0 / np.sqrt(max(D.shape))
        lam = tau
        return {
            "rank": rank,
            "tau": tau,
            "lam": lam,
        }

    def decompose(self, D: NDArray, Omega: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Compute the noisy RPCA with L1 or L2 time penalisation

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
        M, A, _, _ = self.decompose_with_basis(D, Omega)
        return M, A

    def decompose_with_basis(
        self, D: NDArray, Omega: NDArray
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Compute the noisy RPCA with L1 or L2 time penalisation, and returns the decomposition of
        the low-rank matrix.

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
        L: NDArray
            Coefficients of the low-rank matrix in the reduced basis
        Q: NDArray
            Reduced basis of the low-rank matrix
        """
        D = utils.linear_interpolation(D)
        self.params_scale = self.get_params_scale(D)

        if self.lam is not None:
            self.params_scale["lam"] = self.lam
        if self.rank is not None:
            self.params_scale["rank"] = self.rank
        if self.tau is not None:
            self.params_scale["tau"] = self.tau

        lam = self.params_scale["lam"]
        rank = int(self.params_scale["rank"])
        tau = self.params_scale["tau"]
        mu = 1e-2 if self.mu is None else self.mu

        n_rows, n_cols = D.shape
        for period in self.list_periods:
            if not period < n_rows:
                raise ValueError(
                    "The periods provided in argument in `list_periods` must smaller "
                    f"than the number of rows in the matrix but {period} >= {n_rows}!"
                )

        M, A, L, Q = self.minimise_loss(
            D,
            Omega,
            rank,
            tau,
            lam,
            mu,
            self.list_periods,
            self.list_etas,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            norm=self.norm,
        )

        self._check_cost_function_minimized(D, M, A, Omega, tau, lam)

        return M, A, L, Q

    @staticmethod
    def minimise_loss(
        D: NDArray,
        Omega: NDArray,
        rank: int,
        tau: float,
        lam: float,
        mu: float = 1e-2,
        list_periods: List[int] = [],
        list_etas: List[float] = [],
        max_iterations: int = 10000,
        tolerance: float = 1e-6,
        norm: str = "L2",
    ) -> Tuple:
        """
        Compute the noisy RPCA with a L2 time penalisation.

        This function computes the noisy Robust Principal Component Analysis (RPCA) using a L2 time
        penalisation. It iteratively minimizes a loss function to separate the low-rank and sparse
        components from the input data matrix.

        Parameters
        ----------
        D : np.ndarray
            Observations matrix of shape (m, n).
        Omega : np.ndarray
            Binary matrix indicating the observed entries of D, shape (m, n).
        rank : int
            Estimated low-rank of the matrix D.
        tau : float
            Penalizing parameter for the nuclear norm.
        lam : float
            Penalizing parameter for the sparse matrix.
        mu : float, optional
            Initial stiffness parameter for the constraint on M, L, and Q. Defaults
            to 1e-2.
        list_periods : List[int], optional
            List of periods linked to the Toeplitz matrices. Defaults to [].
        list_etas : List[float], optional
            List of penalizing parameters for the corresponding periods in list_periods. Defaults
            to [].
        max_iterations : int, optional
            Stopping criteria, maximum number of iterations. Defaults to 10000.
        tolerance : float, optional
            Stopping criteria, minimum difference between 2 consecutive iterations.
            Defaults to 1e-6.
        norm : str, optional
            Error norm, can be "L1" or "L2". Defaults to "L2".

        Returns
        -------
        Tuple
            A tuple containing the following elements:
            - M : np.ndarray
                Low-rank signal matrix of shape (m, n).
            - A : np.ndarray
                Anomalies matrix of shape (m, n).
            - L : np.ndarray
                Basis unitary array of shape (m, rank).
            - Q : np.ndarray
                Basis unitary array of shape (rank, n).

        Raises
        ------
        ValueError
            If the periods provided in the argument in `list_periods` are not
            smaller than the number of rows in the matrix.
        """

        rho = 1.1
        n_rows, n_cols = D.shape

        # init
        Y = np.zeros((n_rows, n_cols))
        M = D.copy()
        A = np.zeros((n_rows, n_cols))

        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        U = U[:, :rank]
        S = S[:rank]
        Vt = Vt[:rank, :]

        L = U @ np.diag(np.sqrt(S))
        Q = np.diag(np.sqrt(S)) @ Vt

        if norm == "L1":
            R = [np.ones((n_rows, n_cols)) for _ in list_periods]

        mu_bar = mu * 1e3

        # matrices for temporal correlation
        list_H = [rpca_utils.toeplitz_matrix(period, n_rows) for period in list_periods]
        HtH = dok_matrix((n_rows, n_rows))
        for i_period, _ in enumerate(list_periods):
            HtH += list_etas[i_period] * (list_H[i_period].T @ list_H[i_period])

        Ir = np.eye(rank)
        In = identity(n_rows)

        for _ in range(max_iterations):
            M_temp = M.copy()
            A_temp = A.copy()
            L_temp = L.copy()
            Q_temp = Q.copy()
            if norm == "L1":
                R_temp = R.copy()
                sums = np.zeros((n_rows, n_cols))
                for i_period, _ in enumerate(list_periods):
                    sums += mu * R[i_period] - list_H[i_period] @ Y

                M = spsolve(
                    (1 + mu) * In + HtH,
                    D - A + mu * L @ Q - Y + sums,
                )
            else:
                M = spsolve(
                    (1 + mu) * In + 2 * HtH,
                    D - A + mu * L @ Q - Y,
                )
            M = M.reshape(D.shape)

            A_Omega = rpca_utils.soft_thresholding(D - M, lam)
            A_Omega_C = D - M
            A = np.where(Omega, A_Omega, A_Omega_C)
            Q = scp.linalg.solve(
                a=tau * Ir + mu * (L.T @ L),
                b=L.T @ (mu * M + Y),
            )

            L = scp.linalg.solve(
                a=tau * Ir + mu * (Q @ Q.T),
                b=Q @ (mu * M.T + Y.T),
            ).T

            Y += mu * (M - L @ Q)
            if norm == "L1":
                for i_period, _ in enumerate(list_periods):
                    eta = list_etas[i_period]
                    R[i_period] = rpca_utils.soft_thresholding(R[i_period] / mu, eta / mu)

            mu = min(mu * rho, mu_bar)

            Mc = np.linalg.norm(M - M_temp, np.inf)
            Ac = np.linalg.norm(A - A_temp, np.inf)
            Lc = np.linalg.norm(L - L_temp, np.inf)
            Qc = np.linalg.norm(Q - Q_temp, np.inf)
            error_max = max([Mc, Ac, Lc, Qc])  # type: ignore # noqa
            if norm == "L1":
                for i_period, _ in enumerate(list_periods):
                    Rc = np.linalg.norm(R[i_period] - R_temp[i_period], np.inf)
                    error_max = max(error_max, Rc)  # type: ignore # noqa

            if error_max < tolerance:
                break

        M = L @ Q

        M = M

        return M, A, L, Q

    def decompose_on_basis(
        self,
        D: NDArray,
        Omega: NDArray,
        Q: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """
        Decompose the matrix D with an observation matrix Omega using the noisy RPCA algorithm,
        with a fixed reduced basis given by the matrix Q. This allows to impute new data without
        resolving the optimization problem on the whole dataset.

        Parameters
        ----------
        D : NDArray
            _description_
        Omega : NDArray
            _description_
        Q : NDArray
            _description_

        Returns
        -------
        Tuple[NDArray, NDArray]
            A tuple representing the decomposition of D with:
            - M: low-rank matrix
            - A: sparse matrix
        """
        D = utils.linear_interpolation(D)
        params_scale = self.get_params_scale(D)

        lam = params_scale["lam"] if self.lam is None else self.lam
        rank = params_scale["rank"] if self.rank is None else self.rank
        rank = int(rank)
        tau = params_scale["tau"] if self.tau is None else self.tau
        tolerance = self.tolerance

        n_rows, n_cols = D.shape
        if n_rows == 1 or n_cols == 1:
            return D, np.full_like(D, 0)
        # M, A, L, Q = self.decompose_rpca(D, Omega)
        n_rank, _ = Q.shape
        Ir = np.eye(n_rank)
        A = np.zeros((n_rows, n_cols))
        L = np.zeros((n_rows, n_rank))
        for _ in range(self.max_iterations):
            A_prev = A.copy()
            L_prev = L.copy()
            L = scp.linalg.solve(
                a=2 * tau * Ir + (Q @ Q.T),
                b=Q @ (D - A).T,
            ).T
            A_Omega = rpca_utils.soft_thresholding(D - L @ Q, lam)
            A_Omega_C = D - L @ Q
            A = np.where(Omega, A_Omega, A_Omega_C)

            Ac = np.linalg.norm(A - A_prev, np.inf)
            Lc = np.linalg.norm(L - L_prev, np.inf)

            tolerance = max([Ac, Lc])  # type: ignore # noqa

            if tolerance < tolerance:
                break

        M = L @ Q

        return M, A

    def _check_cost_function_minimized(
        self,
        D: NDArray,
        M: NDArray,
        A: NDArray,
        Omega: NDArray,
        tau: float,
        lam: float,
    ):
        """
        Check that the functional minimized by the RPCA is smaller at the end than at the
        beginning.

        Parameters
        ----------
        D : NDArray
            observations matrix with first linear interpolation
        M : NDArray
            low_rank matrix resulting from RPCA
        A : NDArray
            sparse matrix resulting from RPCA
        Omega: NDArrau
            boolean matrix indicating the observed values
        tau : float
            parameter penalizing the nuclear norm of the low rank part
        lam : float
            parameter penalizing the L1-norm of the anomaly/sparse part
        """
        cost_start = self.cost_function(
            D,
            D,
            np.full_like(D, 0),
            Omega,
            tau,
            lam,
            self.list_periods,
            self.list_etas,
            norm=self.norm,
        )
        cost_end = self.cost_function(
            D,
            M,
            A,
            Omega,
            tau,
            lam,
            self.list_periods,
            self.list_etas,
            norm=self.norm,
        )
        function_str = "1/2 ||D-M-A||_2 + tau ||D||_* + lam ||A||_1"
        if len(self.list_etas) > 0:
            for eta in self.list_etas:
                function_str += f"{eta} ||MH||_{self.norm}"

        if self.verbose and (cost_end > cost_start * (1 + 1e-6)):
            warnings.warn(
                f"RPCA algorithm may provide bad results. Function {function_str} increased from"
                f" {cost_start} to {cost_end} instead of decreasing!".format("%.2f")
            )

    @staticmethod
    def cost_function(
        D: NDArray,
        M: NDArray,
        A: NDArray,
        Omega: NDArray,
        tau: float,
        lam: float,
        list_periods: List[int] = [],
        list_etas: List[float] = [],
        norm: str = "L2",
    ):
        """
        Estimated cost function for the noisy RPCA algorithm

        Parameters
        ----------
        D : NDArray
            Matrix of observations
        M : NDArray
            Low-rank signal
        A : NDArray
            Anomalies
        Omega : NDArray
            Mask for observations
        tau: Optional[float]
            penalizing parameter for the nuclear norm
        lam: Optional[float]
            penalizing parameter for the sparse matrix
        list_periods: Optional[List[int]]
            list of periods, linked to the Toeplitz matrices
        list_etas: Optional[List[float]]
            list of penalizing parameters for the corresponding period in list_periods
        norm: Optional[str]
            error norm, can be "L1" or "L2". By default, the value is set to "L2"


        Returns
        -------
        float
            Value of the cost function minimized by the RPCA
        """

        temporal_norm: float = 0
        if len(list_etas) > 0:
            # matrices for temporal correlation
            list_H = [rpca_utils.toeplitz_matrix(period, D.shape[0]) for period in list_periods]
            if norm == "L1":
                for eta, H_matrix in zip(list_etas, list_H):
                    temporal_norm += eta * np.sum(np.abs(H_matrix @ M))
            elif norm == "L2":
                for eta, H_matrix in zip(list_etas, list_H):
                    temporal_norm += eta * float(np.linalg.norm(H_matrix @ M, "fro"))
        anomalies_norm = np.sum(np.abs(A * Omega))
        cost = (
            1 / 2 * ((Omega * (D - M - A)) ** 2).sum()
            + tau * np.linalg.norm(M, "nuc")
            + lam * anomalies_norm
            + temporal_norm
        )
        return cost
