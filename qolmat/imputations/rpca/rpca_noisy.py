from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import scipy as scp
from scipy import linalg as lscp
from scipy.sparse import dok_matrix
from numpy.typing import NDArray
from sklearn import utils as sku

from qolmat.imputations.rpca import rpca_utils
from qolmat.imputations.rpca.rpca import RPCA


class RPCANoisy(RPCA):
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
    period: Optional[int]
        number of rows of the reshaped matrix if the signal is a 1D-array
    rank: Optional[int]
        (estimated) low-rank of the matrix D
    mu: Optional[float]
        initial stiffness parameter for the constraint on X, L and Q
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
    tol: Optional[float]
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
        period: int = 1,
        rank: Optional[int] = None,
        mu: Optional[float] = None,
        tau: Optional[float] = None,
        lam: Optional[float] = None,
        list_periods: List[int] = [],
        list_etas: List[float] = [],
        max_iterations: int = int(1e4),
        tol: float = 1e-6,
        norm: str = "L2",
        verbose: bool = True,
    ) -> None:
        super().__init__(period=period, max_iterations=max_iterations, tol=tol, verbose=verbose)
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

    def decompose_rpca(self, D: NDArray, Omega: NDArray) -> Tuple[NDArray, NDArray]:
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

        params_scale = self.get_params_scale(D)

        lam = params_scale["lam"] if self.lam is None else self.lam
        rank = params_scale["rank"] if self.rank is None else self.rank
        rank = int(rank)
        tau = params_scale["tau"] if self.tau is None else self.tau
        mu = 1e-2 if self.mu is None else self.mu

        n_rows, _ = D.shape
        for period in self.list_periods:
            if not period < n_rows:
                raise ValueError(
                    "The periods provided in argument in `list_periods` must smaller "
                    f"than the number of rows in the matrix but {period} >= {n_rows}!"
                )

        M, A, U, V = self.decompose_rpca_algorithm(
            D,
            Omega,
            rank,
            tau,
            lam,
            mu,
            self.list_periods,
            self.list_etas,
            max_iterations=self.max_iterations,
            tol=self.tol,
            norm=self.norm,
        )

        self._check_cost_function_minimized(D, M, A, Omega, tau, lam)

        return M, A

    def _check_cost_function_minimized(
        self,
        observations: NDArray,
        low_rank: NDArray,
        anomalies: NDArray,
        Omega: NDArray,
        tau: float,
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
        tau : float
            parameter penalizing the nuclear norm of the low rank part
        lam : float
            parameter penalizing the L1-norm of the anomaly/sparse part
        """
        cost_start = self.cost_function(
            observations,
            observations,
            np.full_like(observations, 0),
            Omega,
            tau,
            lam,
            self.list_periods,
            self.list_etas,
            norm=self.norm,
        )
        cost_end = self.cost_function(
            observations,
            low_rank,
            anomalies,
            Omega,
            tau,
            lam,
            self.list_periods,
            self.list_etas,
            norm=self.norm,
        )
        function_str = "1/2 $ ||D-M-A||_2 + tau ||D||_* + lam ||A||_1"
        if len(self.list_etas) > 0:
            for eta in self.list_etas:
                function_str += f"{eta} ||XH||_{self.norm}"

        if self.verbose and (round(cost_start, 4) - round(cost_end, 4)) <= -1e-2:
            warnings.warn(
                f"RPCA algorithm may provide bad results. Function {function_str} increased from"
                f" {cost_start} to {cost_end} instead of decreasing!".format("%.2f")
            )

    @staticmethod
    def decompose_rpca_algorithm(
        D: NDArray,
        Omega: NDArray,
        rank: int,
        tau: float,
        lam: float,
        mu: float = 1e-2,
        list_periods: List[int] = [],
        list_etas: List[float] = [],
        max_iterations: int = 10000,
        tol: float = 1e-6,
        norm: str = "L2",
    ) -> Tuple:
        """
        Compute the noisy RPCA with a L2 time penalisation

        Parameters
        ----------
        D : np.ndarray
            Observations matrix of shape (m, n).
        Omega : np.ndarray
            Binary matrix indicating the observed entries of D, shape (m, n).
        rank: Optional[int]
            (estimated) low-rank of the matrix D
        tau: Optional[float]
            penalizing parameter for the nuclear norm
        lam: Optional[float]
            penalizing parameter for the sparse matrix
        mu: Optional[float]
            initial stiffness parameter for the constraint on X, L and Q
        list_periods: Optional[List[int]]
            list of periods, linked to the Toeplitz matrices
        list_etas: Optional[List[float]]
            list of penalizing parameters for the corresponding period in list_periods
        max_iterations: Optional[int]
            stopping criteria, maximum number of iterations. By default, the value is set to 10_000
        tol: Optional[float]
            stoppign critera, minimum difference between 2 consecutive iterations. By default,
            the value is set to 1e-6
        norm: Optional[str]
            error norm, can be "L1" or "L2". By default, the value is set to "L2"

        Returns
        -------
        M : np.ndarray
            Low-rank signal matrix of shape (m, n).
        A : np.ndarray
            Anomalies matrix of shape (m, n).
        U : np.ndarray
            Basis Unitary array of shape (m, rank).
        V : np.ndarray
            Basis Unitary array of shape (n, rank).

        """

        rho = 1.1
        n_rows, n_cols = D.shape

        # init
        Y = np.zeros((n_rows, n_cols))
        X = D.copy()
        A = np.zeros((n_rows, n_cols))
        U, S, Vt = np.linalg.svd(X)

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
        In = np.eye(n_rows)

        for _ in range(max_iterations):
            # print("Cost function", cost_function(D, X, A, Omega, tau, lam))
            X_temp = X.copy()
            A_temp = A.copy()
            L_temp = L.copy()
            Q_temp = Q.copy()
            if norm == "L1":
                R_temp = R.copy()
                sums = np.zeros((n_rows, n_cols))
                for i_period, _ in enumerate(list_periods):
                    sums += mu * R[i_period] - list_H[i_period] @ Y

                X = scp.linalg.solve(
                    a=(1 + mu) * In + HtH,
                    b=D - A + mu * L @ Q - Y + sums,
                )
            else:
                X = scp.linalg.solve(
                    a=(1 + mu) * In + 2 * HtH,
                    b=D - A + mu * L @ Q - Y,
                )

            A_Omega = rpca_utils.soft_thresholding(D - X, lam)
            A_Omega_C = D - X
            A = np.where(Omega, A_Omega, A_Omega_C)

            Q = scp.linalg.solve(
                a=tau * Ir + mu * (L.T @ L),
                b=L.T @ (mu * X + Y),
            )

            L = scp.linalg.solve(
                a=tau * Ir + mu * (Q @ Q.T),
                b=Q @ (mu * X.T + Y.T),
            ).T

            Y += mu * (X - L @ Q)
            if norm == "L1":
                for i_period, _ in enumerate(list_periods):
                    eta = list_etas[i_period]
                    R[i_period] = rpca_utils.soft_thresholding(R[i_period] / mu, eta / mu)

            mu = min(mu * rho, mu_bar)

            Xc = np.linalg.norm(X - X_temp, np.inf)
            Ac = np.linalg.norm(A - A_temp, np.inf)
            Lc = np.linalg.norm(L - L_temp, np.inf)
            Qc = np.linalg.norm(Q - Q_temp, np.inf)
            tolerance = max([Xc, Ac, Lc, Qc])  # type: ignore # noqa
            if norm == "L1":
                for i_period, _ in enumerate(list_periods):
                    Rc = np.linalg.norm(R[i_period] - R_temp[i_period], np.inf)
                    tolerance = max(tolerance, Rc)  # type: ignore # noqa

            if tolerance < tol:
                break

        X = L @ Q

        M = X
        U = L
        V = Q

        return M, A, U, V

    @staticmethod
    def cost_function(
        observations: NDArray,
        low_rank: NDArray,
        anomalies: NDArray,
        Omega: NDArray,
        tau: float,
        lam: float,
        list_periods: List[int] = [],
        list_etas: List[float] = [],
        norm: str = "L2",
    ):
        """
        Compute cost function for different RPCA algorithm

        Parameters
        ----------
        observations : NDArray
            Matrix of observations
        low_rank : NDArray
            Low-rank signal
        anomalies : NDArray
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
            list_H = [
                rpca_utils.toeplitz_matrix(period, observations.shape[0])
                for period in list_periods
            ]
            if norm == "L1":
                for eta, H_matrix in zip(list_etas, list_H):
                    temporal_norm += eta * np.sum(np.abs(H_matrix @ low_rank))
            elif norm == "L2":
                for eta, H_matrix in zip(list_etas, list_H):
                    temporal_norm += eta * float(np.linalg.norm(H_matrix @ low_rank, "fro"))
        anomalies_norm = np.sum(np.abs(anomalies * Omega))
        cost = (
            1 / 2 * ((Omega * (observations - low_rank - anomalies)) ** 2).sum()
            + tau * np.linalg.norm(low_rank, "nuc")
            + lam * anomalies_norm
            + temporal_norm
        )
        return cost
