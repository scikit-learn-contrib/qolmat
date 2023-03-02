from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import scipy as scp
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.extmath import randomized_svd

from qolmat.imputations.rpca import utils
from qolmat.imputations.rpca.rpca import RPCA


class TemporalRPCA(RPCA):
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
    verbose: Optional[bool]
        verbosity. By default, the value is set to False
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
        max_iter: Optional[int] = int(1e4),
        tol: Optional[float] = 1e-6,
        verbose: Optional[bool] = False,
        norm: Optional[str] = "L2",
    ) -> None:
        super().__init__(period=period, max_iter=max_iter, tol=tol, verbose=verbose)
        self.rank = rank
        self.tau = tau
        self.lam = lam
        self.list_periods = list_periods
        self.list_etas = list_etas
        self.norm = norm

    def compute_L1(self, proj_D, omega, lam, tau, rank) -> None:
        """
        compute RPCA with possible temporal regularisations, penalised with L1 norm
        """
        m, n = proj_D.shape
        rho = 1.1
        mu = 1e-6
        mu_bar = mu * 1e10

        # init
        Y = np.ones((m, n))
        Y_ = [np.ones((m, n - period)) for period in self.list_periods]

        X = proj_D.copy()
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
                b=(proj_D - A + mu * L @ Q.T - Y + sums).T,
            ).T
            
            if np.any(np.isnan(proj_D)):
                A_omega = utils.soft_thresholding(proj_D - X, lam)
                A_omega = utils.ortho_proj(A_omega, omega, inverse=False)
                A_omega_C = proj_D - X
                A_omega_C = utils.ortho_proj(A_omega_C, omega, inverse=True)
                A = A_omega + A_omega_C
            else:
                A = utils.soft_thresholding(proj_D - X, lam)

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
                Rc = max(Rc, np.linalg.norm(R[index] - R_temp[index], np.inf))
            tol = max([Xc, Ac, Lc, Qc, Rc])
            errors[iteration] = tol

            if tol < self.tol:
                if self.verbose:
                    print(f"Converged in {iteration} iterations with error: {tol}")
                break
        M = X
        U = L
        V = Q
        return M, A, U, V, errors

    def compute_L2(self, proj_D, omega, lam, tau, rank) -> None:
        """
        compute RPCA with possible temporal regularisations, penalised with L2 norm
        """
        rho = 1.1
        m, n = proj_D.shape

        # init
        Y = np.ones((m, n))
        X = proj_D.copy()
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
                b=(proj_D - A + mu * L @ Q.T - Y).T,
            ).T
            
            if np.any(~omega):
                A_omega = utils.soft_thresholding(proj_D - X, lam)
                A_omega = utils.ortho_proj(A_omega, omega, inverse=False)
                A_omega_C = proj_D - X
                A_omega_C = utils.ortho_proj(A_omega_C, omega, inverse=True)
                A = A_omega + A_omega_C
            else:
                A = utils.soft_thresholding(proj_D - X, lam)

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
                if self.verbose:
                    print(f"Converged in {iteration} iterations with error: {tol}")
                break

        X = L @ Q.T

        M = X
        U = L
        V = Q
    
        return M, A, U, V, errors

    def get_params(self) -> dict:
        dict_params = super().get_params()
        dict_params["tau"] = self.tau
        dict_params["lam"] = self.lam
        dict_params["list_periods"] = self.list_periods
        dict_params["list_etas"] = self.list_etas
        dict_params["norm"] = self.norm
        return dict_params

    def get_params_scale(self, D: NDArray) -> dict:
        rank = utils.approx_rank(D)
        tau = 1.0 / np.sqrt(max(D.shape))
        lam = tau
        return {
            "rank": rank,
            "tau": tau,
            "lam": lam,
        }

    def set_params(self, **kargs):
        _ = super().set_params(**kargs)

        for key, value in kargs.items():
            setattr(self, key, value)

        list_periods = []
        list_etas = []

        for key, value in kargs.items():
            if "period" in key:
                index_period = int(key[7:])
                if f"eta_{index_period}" in kargs.keys():
                    list_periods.append(value)
                    list_etas.append(kargs[f"eta_{index_period}"])
                else:
                    raise ValueError(f"No etas' index correspond to {key}")

        self.list_periods = list_periods
        self.list_etas = list_etas
        return self

    def fit_transform(
        self,
        X: NDArray,
    ) -> NDArray:
        """
        Compute the noisy RPCA with time "penalisations"

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
        U:
            Basis Unitary array
        V:
            Basis Unitary array
        
        errors:
            Array of iterative errors
        """
        X = X.copy().T
        D_init = self._prepare_data(X)
        omega = ~np.isnan(D_init)
        proj_D = utils.impute_nans(D_init, method="median")

        params_scale = self.get_params_scale(proj_D)

        lam = params_scale["lam"] if self.lam is None else self.lam
        rank = params_scale["rank"] if self.rank is None else self.rank
        tau = params_scale["tau"] if self.tau is None else self.tau

        if self.norm == "L1":
            M, A, U, V, errors = self.compute_L1(proj_D, omega, lam, tau, rank)
        elif self.norm == "L2":
            M, A, U, V, errors = self.compute_L2(proj_D, omega, lam, tau, rank)
        
        if X.shape[0] == 1:
            M = M.reshape(1, -1)[:, :X.size]
            A = A.reshape(1, -1)[:, :X.size]
        M = M.T
        A = A.T

        # return M, A, U, V, errors    
        return M


class OnlineTemporalRPCA(TemporalRPCA):
    """
    This class implements an online version of TemporalRPCA
    that processes one sample per time instance and hence its memory cost
    is independent of the number of samples
    It is based on stochastic optimization of an equivalent reformulation
    of the batch TemporalRPCA

    Parameters
    ----------
    period: Optional[int]
        number of rows of the reshaped matrix if the signal is a time series
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
        stoppign critera, minimum difference between 2 consecutive iterations. By default, the
        value is set to 1e-6
    verbose: Optional[bool]
        verbosity. By default, the value is set to False
    burnin: Optional[float]
        proportion of the entire matrix (the first (burnin x 100)% columns) for the batch part.
        Has to be between 0 and 1.
        By default, the value is set to 0.
    nwin: Optional[int]
        size of the sliding window (number of column). By default, the value is set to 0.
    online_tau: Optional[float]
        penalizing parameter for the nuclear norm, online part
    online_lam: Optional[float]
        penalizing parameter for the sparse matrix, online part
    online_list_etas: Optional[List[float]]
        list of penalizing parameters for the corresponding period in list_periods, online part
    """

    def __init__(
        self,
        period: Optional[int] = None,
        rank: Optional[int] = None,
        tau: Optional[float] = None,
        lam: Optional[float] = None,
        list_periods: Optional[List[int]] = [],
        list_etas: Optional[List[float]] = [],
        max_iter: Optional[int] = int(1e4),
        tol: Optional[float] = 1e-6,
        verbose: Optional[bool] = False,
        burnin: Optional[float] = 0,
        nwin: Optional[float] = 0,
        online_tau: Optional[float] = None,
        online_lam: Optional[float] = None,
        online_list_etas: Optional[ArrayLike] = [],
    ) -> None:
        super().__init__(
            period=period,
            rank=rank,
            tau=tau,
            lam=lam,
            list_periods=list_periods,
            list_etas=list_etas,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
        )

        self.burnin = burnin
        self.nwin = nwin
        self.online_tau = online_tau
        self.online_lam = online_lam
        self.online_list_etas = online_list_etas
        self.norm = "L2"

    def get_params(self):
        return {
            "n_rows": self.n_rows,
            "estimated_rank": self.rank,
            "tau": self.tau,
            "lam": self.lam,
            "list_periods": self.list_periods,
            "list_etas": self.list_etas,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "verbose": self.verbose,
            "norm": self.norm,
            "burnin": self.burnin,
            "nwin": self.nwin,
            "online_tau": self.online_tau,
            "online_lam": self.online_lam,
            "online_list_etas": self.online_list_etas,
        }

   

    def get_params_scale_online(
        self,
        D:NDArray, Lhat: NDArray
    ) -> dict[str, float]:
        # D_init = self._prepare_data(signal=X)
        params_scale = self.get_params_scale(D)
        # burnin = int(D_init.shape[1] * self.burnin)

        # super_class = TemporalRPCA(**super().get_params())
        # Lhat, _, _ = super_class.fit_transform(X=D_init[:, :burnin])
        _, sigmas_hat, _ = np.linalg.svd(Lhat)
        online_tau = 1.0 / np.sqrt(len(Lhat)) / np.mean(sigmas_hat[: params_scale["rank"]])
        online_lam = 1.0 / np.sqrt(len(Lhat))
        params_scale["online_tau"] = online_tau
        params_scale["online_lam"] = online_lam
        return params_scale

    def fit_transform(
        self,
        X: NDArray,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """
        Compute an online version of RPCA with temporal regularisations

        Parameters
        ----------
        X : NDArray

        Returns
        -------
        M: NDArray
            Low-rank signal
        A: NDArray
            Anomalies
        U:
            Basis Unitary array. ``None``
        V:
            Basis Unitary array. ``None``
        errors:
            Array of iterative errors. ``None``
        """
        X = X.copy().T
        D_init = self._prepare_data(X)
        burnin = int(self.burnin * D_init.shape[1])
        
        if burnin < len(D_init):
            raise ValueError(f"'self.burnin={self.burnin} is to small. Only {burnin} columns kept for {len(D_init)} rows",
                            "Increase self.burnin!")
        nwin = self.nwin

        m, n = D_init.shape
        # super_class = TemporalRPCA(**super().get_params())
        # Lhat, Shat, _, _, _ =super_class.fit_transform(X=D_init[:, :burnin])
        
        proj_D = utils.impute_nans(D_init, method="median")
        omega = ~np.isnan(D_init)

        params_scale = self.get_params_scale(proj_D)

        lam = params_scale["lam"] if self.lam is None else self.lam
        rank = params_scale["rank"] if self.rank is None else self.rank
        tau = params_scale["tau"] if self.tau is None else self.tau

        if self.norm == "L1":
            M, A, U, V, errors = self.compute_L1(proj_D, omega, lam, tau, rank)
        elif self.norm == "L2":
            M, A, U, V, errors = self.compute_L2(proj_D, omega, lam, tau, rank)

        Lhat, Shat, _ = np.linalg.svd(M, full_matrices=False, compute_uv=True)

        params_scale = self.get_params_scale_online(proj_D, Lhat)

        online_tau = params_scale["online_tau"] if self.online_tau is None else self.online_tau 
        online_lam = params_scale["online_lam"] if self.online_lam is None else self.online_lam 

        if len(self.online_list_etas) == 0:
            self.online_list_etas = self.list_etas
        
        approx_rank =  utils.approx_rank(proj_D[:, :burnin])

        # TODO : is it really Lhat that should be used here?!
        Uhat, sigmas_hat, Vhat = randomized_svd(
            Lhat, n_components=approx_rank, n_iter=5, random_state=42
        )
        U = Uhat[:, :approx_rank]@(np.sqrt(np.diag(sigmas_hat[:approx_rank])))

        if self.nwin == 0:
            Vhat_win = Vhat.copy()
        else:
            Vhat_win = Vhat[:, -nwin:]

        A = np.zeros((approx_rank, approx_rank))
        B = np.zeros((m, approx_rank))

        for col in range(Vhat_win.shape[1]):
            sums = np.zeros(A.shape)
            for index, period in enumerate(self.list_periods):
                if col >= period:
                    vec = Vhat_win[:, col] - Vhat_win[:, col - period]
                    sums += 2 * self.list_etas[index] * (np.outer(vec, vec))
            A = A + np.outer(Vhat_win[:, col], Vhat_win[:, col]) + sums

            if nwin == 0:
                B = B + np.outer(proj_D[:, col] - Shat[:, col], Vhat_win[:, col])
            else:
                B = B + np.outer(
                    proj_D[:, burnin - nwin + col] - Shat[:, burnin - nwin + col],
                    Vhat_win[:, col],
                )
        
        m_lhat, n_lhat = Lhat.shape
        m_shat, n_shat = Shat.shape
        m_vhat, n_vhat = Vhat.shape

        lv = np.empty(shape=(m_vhat, n - burnin), dtype=float)

        Shat_grow = np.empty((m_shat, (n - burnin) + n_shat), dtype=float)
        Shat_grow[:, :n_shat] = Shat

        Vhat_win_grow = np.empty((m_vhat, (n - burnin) + n_vhat), dtype=float)
        Vhat_win_grow[:, : Vhat_win.shape[1]] = Vhat_win

        Lhat_grow = np.empty((m_lhat, n), dtype=float)
        Lhat_grow[:, :n_lhat] = Lhat

        for i in range(burnin, n):
            ri = proj_D[:, i]
            vi, si = utils.solve_projection(
                ri,
                U,
                online_tau,
                online_lam,
                self.online_list_etas,
                self.list_periods,
                Lhat_grow[:, :i],
            )
            lv[:, i - burnin] = vi

            Shat_grow[:, i] = si
            Vhat_win_grow[:, nwin + (i - burnin)] = vi
            vi_delete = Vhat_win_grow[:, (i - burnin)]

            if len(self.list_periods) > 0:
                sums = np.zeros((len(vi), len(vi)))
                for k, period in enumerate(self.list_periods):
                    if i - burnin >= period:
                        vec = vi - lv[:, i - period - burnin]
                        sums += 2 * self.online_list_etas[k] * (np.outer(vec, vec))
                if self.nwin == 0:
                    A = A + np.outer(vi, vi) + sums
                else:
                    A = A + np.outer(vi, vi) + sums - np.outer(vi_delete, vi_delete)
            else:
                if self.nwin == 0:
                    A = A + np.outer(vi, vi)
                else:
                    A = A + np.outer(vi, vi) - np.outer(vi_delete, vi_delete)
            if self.nwin == 0:
                B = B + np.outer(ri - si, vi)
            else:
                B = (
                    B
                    + np.outer(ri - si, vi)
                    - np.outer(
                        proj_D[:, i - self.nwin] - Shat_grow[:, i - self.nwin],
                        vi_delete,
                    )
                )
            U = utils.update_col(online_tau, U, A, B)
            Lhat_grow[:, i] = U @ vi

        if len(X) == 1:
            M = Lhat_grow.T.flatten()
            A = Shat_grow.T.flatten()
            M, A = M[:(M.size - n_add_values)], A[:(M.size - n_add_values)]
        else:
            M = Lhat_grow
            A = Shat_grow
        return M, A, None, None, None
