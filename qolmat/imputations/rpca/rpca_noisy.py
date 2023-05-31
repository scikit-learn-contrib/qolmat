from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy as scp
from numpy.typing import NDArray

from qolmat.imputations.rpca import utils
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
    n_rows: Optional[int]
        number of rows of the reshaped matrix if the signal is a 1D-array
    rank: Optional[int]
        (estimated) low-rank of the matrix D
    tau: Optional[float]
        penalizing parameter for the nuclear norm
    lam: Optional[float]
        penalizing parameter for the sparse matrix
    list_periods: Optional[List[int]]
        list of periods, linked to the Toeplitz matrices
    list_etas: Optional[List[float]]
        list of penalizing parameters for the corresponding period in list_periods
    max_iter: Optional[int]
        stopping criteria, maximum number of iterations. By default, the value is set to 10_000
    tol: Optional[float]
        stoppign critera, minimum difference between 2 consecutive iterations. By default,
        the value is set to 1e-6
    norm: Optional[str]
        error norm, can be "L1" or "L2". By default, the value is set to "L2"
    """

    def __init__(
        self,
        period: Optional[int] = None,
        rank: Optional[int] = None,
        tau: Optional[float] = None,
        lam: Optional[float] = None,
        list_periods: List[int] = [],
        list_etas: List[float] = [],
        max_iter: int = int(1e4),
        tol: float = 1e-6,
        norm: Optional[str] = "L2",
    ) -> None:
        super().__init__(period=period, max_iter=max_iter, tol=tol)
        self.rank = rank
        self.tau = tau
        self.lam = lam
        self.list_periods = list_periods
        self.list_etas = list_etas
        self.norm = norm

    def decompose_rpca_L1(
        self, D: NDArray, Omega: NDArray, lam: float, tau: float, rank: int
    ) -> Tuple:
        """
        Compute the noisy RPCA with a L1 time penalisation

        Parameters
        ----------
        D : np.ndarray
            Observations matrix of shape (m, n).
        Omega : np.ndarray
            Binary matrix indicating the observed entries of D, shape (m, n).
        lam : float
            Regularization parameter for the L1 norm.
        tau : float
            Regularization parameter for the temporal correlations.
        rank : int
            Rank parameter for low-rank matrix decomposition.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing:
        M : np.ndarray
            Low-rank signal matrix of shape (m, n).
        A : np.ndarray
            Anomalies matrix of shape (m, n).
        U : np.ndarray
            Basis Unitary array of shape (m, rank).
        V : np.ndarray
            Basis Unitary array of shape (n, rank).
        errors : np.ndarray
            Array of iterative errors.

        """
        m, n = D.shape
        rho = 1.1
        mu = 1e-6
        mu_bar = mu * 1e10

        # init
        Y = np.ones((m, n))
        Y_ = [np.ones((m, n - period)) for period in self.list_periods]

        X = D.copy()
        A = np.zeros((m, n))
        L = np.ones((m, rank))
        Q = np.ones((n, rank))
        R = [np.ones((m, n - period)) for period in self.list_periods]
        # temporal correlations
        H = [utils.toeplitz_matrix(period, n, model="column") for period in self.list_periods]

        ##
        HHT = np.zeros((n, n))
        for index, _ in enumerate(self.list_periods):
            HHT += self.list_etas[index] * (H[index] @ H[index].T)

        Ir = np.eye(rank)
        In = np.eye(n)

        errors = np.full((self.max_iter,), np.nan, dtype=float)

        for iteration in range(self.max_iter):
            X_temp = X.copy()
            A_temp = A.copy()
            L_temp = L.copy()
            Q_temp = Q.copy()
            R_temp = R.copy()

            sums = np.zeros((m, n))
            for index, _ in enumerate(self.list_periods):
                sums += (mu * R[index] - Y_[index]) @ H[index].T

            X = scp.linalg.solve(
                a=((1 + mu) * In + 2 * HHT).T,
                b=(D - A + mu * L @ Q.T - Y + sums).T,
            ).T

            if np.any(np.isnan(D)):
                A_Omega = utils.soft_thresholding(D - X, lam)
                A_Omega_C = D - X
                A = np.where(Omega, A_Omega, A_Omega_C)
            else:
                A = utils.soft_thresholding(D - X, lam)

            L = scp.linalg.solve(
                a=(tau * Ir + mu * (Q.T @ Q)).T,
                b=((mu * X + Y) @ Q).T,
            ).T

            Q = scp.linalg.solve(
                a=(tau * Ir + mu * (L.T @ L)).T,
                b=((mu * X.T + Y.T) @ L).T,
            ).T

            for index, _ in enumerate(self.list_periods):
                R[index] = utils.soft_thresholding(
                    X @ H[index].T - Y_[index] / mu, self.list_etas[index] / mu
                )

            Y += mu * (X - L @ Q.T)
            for index, _ in enumerate(self.list_periods):
                Y_[index] += mu * (X @ H[index].T - R[index])

            # update mu
            mu = min(mu * rho, mu_bar)

            # stopping criteria
            Xc = np.linalg.norm(X - X_temp, np.inf)
            Ac = np.linalg.norm(A - A_temp, np.inf)
            Lc = np.linalg.norm(L - L_temp, np.inf)
            Qc = np.linalg.norm(Q - Q_temp, np.inf)
            Rc = -1
            for index, _ in enumerate(self.list_periods):
                Rc = np.maximum(Rc, np.linalg.norm(R[index] - R_temp[index], np.inf))
            tol = np.amax(np.array([Xc, Ac, Lc, Qc, Rc]))
            errors[iteration] = tol

            if tol < self.tol:
                break
        M = X
        U = L
        V = Q
        return M, A, U, V, errors

    def decompose_rpca_L2(
        self, D: NDArray, Omega: NDArray, lam: float, tau: float, rank: int
    ) -> Tuple:
        """
        Compute the noisy RPCA with a L1 time penalisation

        Parameters
        ----------
        D : np.ndarray
            Observations matrix of shape (m, n).
        Omega : np.ndarray
            Binary matrix indicating the observed entries of D, shape (m, n).
        lam : float
            Regularization parameter for the L2 norm.
        tau : float
            Regularization parameter for the temporal correlations.
        rank : int
            Rank parameter for low-rank matrix decomposition.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing:
        M : np.ndarray
            Low-rank signal matrix of shape (m, n).
        A : np.ndarray
            Anomalies matrix of shape (m, n).
        U : np.ndarray
            Basis Unitary array of shape (m, rank).
        V : np.ndarray
            Basis Unitary array of shape (n, rank).
        errors : np.ndarray
            Array of iterative errors.

        """
        rho = 1.1
        m, n = D.shape

        # init
        Y = np.ones((m, n))
        X = D.copy()
        A = np.zeros((m, n))
        L = np.ones((m, rank))
        Q = np.ones((n, rank))

        mu = 1e-6
        mu_bar = mu * 1e10

        # matrices for temporal correlation
        H = [utils.toeplitz_matrix(period, n, model="column") for period in self.list_periods]
        HHT = np.zeros((n, n))
        for index, _ in enumerate(self.list_periods):
            HHT += self.list_etas[index] * (H[index] @ H[index].T)

        Ir = np.eye(rank)
        In = np.eye(n)

        errors = np.full((self.max_iter,), np.nan, dtype=float)

        for iteration in range(self.max_iter):
            X_temp = X.copy()
            A_temp = A.copy()
            L_temp = L.copy()
            Q_temp = Q.copy()

            X = scp.linalg.solve(
                a=((1 + mu) * In + HHT).T,
                b=(D - A + mu * L @ Q.T - Y).T,
            ).T

            if np.any(~Omega):
                A_omega = utils.soft_thresholding(D - X, lam)
                A_omega_C = D - X
                A = np.where(Omega, A_omega, A_omega_C)
            else:
                A = utils.soft_thresholding(D - X, lam)

            L = scp.linalg.solve(
                a=(tau * Ir + mu * (Q.T @ Q)).T,
                b=((mu * X + Y) @ Q).T,
            ).T

            Q = scp.linalg.solve(
                a=(tau * Ir + mu * (L.T @ L)).T,
                b=((mu * X.T + Y.T) @ L).T,
            ).T

            Y += mu * (X - L @ Q.T)

            mu = min(mu * rho, mu_bar)

            Xc = np.linalg.norm(X - X_temp, np.inf)
            Ac = np.linalg.norm(A - A_temp, np.inf)
            Lc = np.linalg.norm(L - L_temp, np.inf)
            Qc = np.linalg.norm(Q - Q_temp, np.inf)

            tol = max([Xc, Ac, Lc, Qc])
            errors[iteration] = tol
            if tol < self.tol:
                break

        X = L @ Q.T

        M = X
        U = L
        V = Q

        return M, A, U, V, errors

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
        rank = utils.approx_rank(D)
        tau = 1.0 / np.sqrt(max(D.shape))
        lam = tau
        return {
            "rank": rank,
            "tau": tau,
            "lam": lam,
        }

    def decompose_rpca_signal(
        self,
        X: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute the noisy RPCA with L1 or L2 time penalisation

        Parameters
        ----------
        X : NDArray
            Observations

        Returns
        -------
        M: NDArray
            Low-rank signal
        A: NDArray
            Anomalies
        """
        D_init = self._prepare_data(X)
        Omega = ~np.isnan(D_init)
        D_proj = utils.impute_nans(D_init, method="median")

        params_scale = self.get_params_scale(D_proj)

        lam = params_scale["lam"] if self.lam is None else self.lam
        rank = params_scale["rank"] if self.rank is None else self.rank
        rank = int(rank)
        tau = params_scale["tau"] if self.tau is None else self.tau

        if self.norm == "L1":
            M, A, U, V, errors = self.decompose_rpca_L1(D_proj, Omega, lam, tau, rank)
        elif self.norm == "L2":
            M, A, U, V, errors = self.decompose_rpca_L2(D_proj, Omega, lam, tau, rank)

        M_final = self.get_shape_original(M, X)
        A_final = self.get_shape_original(A, X)

        return M_final, A_final
