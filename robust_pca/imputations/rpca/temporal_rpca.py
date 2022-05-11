from __future__ import annotations
from typing import Optional, List

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.extmath import randomized_svd

from robust_pca.classes.rpca import RPCA
from robust_pca.utils import utils


class TemporalRPCA(RPCA):
    """
    This class implements a noisy version of the so-called improved RPCA

    References
    ----------
    Wang, Xuehui, et al. "An improved robust principal component analysis model for anomalies detection of subway passenger flow."
    Journal of advanced transportation 2018 (2018).

    Chen, Yuxin, et al. "Bridging convex and nonconvex optimization in robust PCA: Noise, outliers and missing data."
    The Annals of Statistics 49.5 (2021): 2948-2971.

    Parameters
    ----------
    rank: Optional
        (estimated) low-rank of the matrix D
    tau: Optional
        penalizing parameter for the nuclear norm
    lam: Optional
        penalizing parameter for the sparse matrix
    list_periods: Optional
        list of periods, linked to the Toeplitz matrices
    list_etas: Optional
        list of penalizing parameters for the corresponding period in list_periods
    """

    def __init__(
        self,
        n_rows: Optional[int] = None,
        rank: Optional[int] = None,
        tau: Optional[float] = None,
        lam: Optional[float] = None,
        list_periods: List[int] = [],
        list_etas: List[float] = [],
        maxIter: Optional[int] = int(1e4),
        tol: Optional[float] = 1e-6,
        verbose: Optional[bool] = False,
        norm: Optional[str] = "L2",
    ) -> None:
        super().__init__(n_rows=n_rows, maxIter=maxIter, tol=tol, verbose=verbose)
        self.rank = rank
        self.tau = tau
        self.lam = lam
        self.list_periods = list_periods
        self.list_etas = list_etas
        self.norm = norm

    def compute_L1(self, proj_D, omega) -> None:
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
        L = np.ones((m, self.rank))
        Q = np.ones((n, self.rank))
        R = [np.ones((m, n - period)) for period in self.list_periods]
        # temporal correlations
        H = [
            utils.toeplitz_matrix(period, n, model="column")
            for period in self.list_periods
        ]

        Ir = np.eye(self.rank)
        In = np.eye(n)

        ##
        HHT = np.zeros((n, n))
        for index, _ in enumerate(self.list_periods):
            HHT += self.list_etas[index] * (H[index] @ H[index].T)

        errors = []
        for iteration in range(self.maxIter):
            # save current variable values
            X_temp = X.copy()
            A_temp = A.copy()
            L_temp = L.copy()
            Q_temp = Q.copy()
            R_temp = R.copy()

            sums = np.zeros((m, n))
            for index, _ in enumerate(self.list_periods):
                sums += (mu * R[index] - Y_[index]) @ H[index].T
            X = (proj_D - A + mu * L @ Q.T - Y + sums) @ np.linalg.inv(
                (1 + mu) * In + 2 * HHT
            )

            if np.any(np.isnan(proj_D)):
                A_omega = utils.soft_thresholding(proj_D - X, self.lam)
                A_omega = utils.ortho_proj(A_omega, omega, inverse=False)
                A_omega_C = proj_D - X
                A_omega_C = utils.ortho_proj(A_omega_C, omega, inverse=True)
                A = A_omega + A_omega_C
            else:
                A = utils.soft_thresholding(proj_D - X, self.lam)

            L = (mu * X + Y) @ Q @ np.linalg.inv(self.tau * Ir + mu * (Q.T @ Q))
            Q = (mu * X.T + Y.T) @ L @ np.linalg.inv(self.tau * Ir + mu * (L.T @ L))

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
            errors.append(tol)

            if tol < self.tol:
                if self.verbose:
                    print(f"Converged in {iteration} iterations with error: {tol}")
                break
        return X, A, errors

    def compute_L2(self, proj_D, omega) -> None:
        """
        compute RPCA with possible temporal regularisations, penalised with L2 norm
        """
        rho = 1.1
        m, n = proj_D.shape

        # init
        Y = np.ones((m, n))
        X = proj_D.copy()
        A = np.zeros((m, n))
        L = np.ones((m, self.rank))
        Q = np.ones((n, self.rank))

        mu = 1e-6
        mu_bar = mu * 1e10

        # matrices for temporal correlation
        H = [
            utils.toeplitz_matrix(period, n, model="column")
            for period in self.list_periods
        ]
        HHT = np.zeros((n, n))
        for index, _ in enumerate(self.list_periods):
            HHT += self.list_etas[index] * (H[index] @ H[index].T)

        Ir = np.eye(self.rank)
        In = np.eye(n)

        errors = []
        for iteration in range(self.maxIter):
            X_temp = X.copy()
            A_temp = A.copy()
            L_temp = L.copy()
            Q_temp = Q.copy()
            X = (proj_D - A + mu * L @ Q.T - Y) @ np.linalg.inv((1 + mu) * In + HHT)

            if np.any(~omega):
                A_omega = utils.soft_thresholding(proj_D - X, self.lam)
                A_omega = utils.ortho_proj(A_omega, omega, inverse=False)
                A_omega_C = proj_D - X
                A_omega_C = utils.ortho_proj(A_omega_C, omega, inverse=True)
                A = A_omega + A_omega_C
            else:
                A = utils.soft_thresholding(proj_D - X, self.lam)
            L = (mu * X + Y) @ Q @ np.linalg.inv(self.tau * Ir + mu * (Q.T @ Q))
            Q = (mu * X.T + Y.T) @ L @ np.linalg.inv(self.tau * Ir + mu * (L.T @ L))
            Y += mu * (X - L @ Q.T)

            mu = min(mu * rho, mu_bar)

            Xc = np.linalg.norm(X - X_temp, np.inf)
            Ac = np.linalg.norm(A - A_temp, np.inf)
            Lc = np.linalg.norm(L - L_temp, np.inf)
            Qc = np.linalg.norm(Q - Q_temp, np.inf)

            tol = max([Xc, Ac, Lc, Qc])
            errors.append(tol)

            if tol < self.tol:
                if self.verbose:
                    print(f"Converged in {iteration} iterations with error: {tol}")
                break

        X = L @ Q.T
        return X, A, errors

    def get_params(self) -> dict:
        dict_params = super().get_params()
        dict_params["tau"] = self.tau
        dict_params["lam"] = self.lam
        dict_params["list_periods"] = self.list_periods
        dict_params["list_etas"] = self.list_etas
        dict_params["norm"] = self.norm
        return dict_params

    def set_params(self, **kargs):
        _ = super().set_params(**kargs)
        for param_key in kargs.keys():
            setattr(self, param_key, kargs[param_key])

        list_periods = []
        list_etas = []
        for param_key in kargs.keys():
            if "period" in param_key:
                index_period = int(param_key[7:])
                if f"eta_{index_period}" in kargs.keys():
                    list_periods.append(kargs[param_key])
                    list_etas.append(kargs[f"eta_{index_period}"])
                else:
                    raise ValueError(f"No etas' index correspond to {param_key}")
        self.list_periods = list_periods
        self.list_etas = list_etas
        return self

    def fit_transform(
        self,
        signal: NDArray,
    ) -> None:
        """
        Compute the noisy RPCA with time "penalisations"

        Parameters
        ----------
        signal : NDArray
            Observations
        """

        D_init, n_add_values = self._prepare_data(signal=signal)
        omega = ~np.isnan(D_init)
        proj_D = utils.impute_nans(D_init, method="median")

        if self.rank is None:
            self.rank = utils.approx_rank(proj_D)
        if self.tau is None:
            self.tau = 1.0 / np.sqrt(max(D_init.shape))
        if self.lam is None:
            self.lam = 1.0 / np.sqrt(max(D_init.shape))

        if self.norm == "L1":
            X, A, errors = self.compute_L1(proj_D, omega)
        elif self.norm == "L2":
            X, A, errors = self.compute_L2(proj_D, omega)
        X = X.T
        A = A.T
        if n_add_values > 0:
            X.flat[-n_add_values:] = np.nan
            A.flat[-n_add_values:] = np.nan

        if self.input_data == "2DArray":
            return X, A, errors
        elif self.input_data == "1DArray":
            ts_X = X.flatten()
            ts_A = A.flatten()
            ts_X = ts_X[~np.isnan(ts_X)]
            ts_A = ts_A[~np.isnan(ts_A)]
            return ts_X, ts_A, errors
        else:
            raise ValueError("input data type not recognized")

    def get_params_scale(self, signal: NDArray) -> None:
        D_init, _ = self._prepare_data(signal=signal)
        proj_D = utils.impute_nans(D_init, method="median")
        rank = utils.approx_rank(proj_D)
        tau = 1.0 / np.sqrt(max(D_init.shape))
        lam = tau
        return {
            "rank": rank,
            "tau": tau,
            "lam": lam,
        }


class OnlineTemporalRPCA(TemporalRPCA):
    """
    This class implements an online version of Temporal RPCA
    that processes one sample per time instance and hence its memory cost
    is independent of the number of samples
    It is based on stochastic optimization of an equivalent reformulation
    of the batch TemporalRPCA

    Parameters
    ----------
    TemporalRPCA
    """

    def __init__(
        self,
        n_rows: Optional[int] = None,
        rank: Optional[int] = None,
        tau: Optional[float] = None,
        lam: Optional[float] = None,
        list_periods: Optional[List[int]] = [],
        list_etas: Optional[List[float]] = [],
        maxIter: Optional[int] = int(1e4),
        tol: Optional[float] = 1e-6,
        verbose: Optional[bool] = False,
        norm: Optional[str] = "L2",
        burnin: Optional[float] = 0,
        nwin: Optional[float] = 0,
        online_tau: Optional[float] = None,
        online_lam: Optional[float] = None,
        online_list_etas: Optional[ArrayLike] = [],
    ) -> None:
        super().__init__(
            n_rows=n_rows,
            rank=rank,
            tau=tau,
            lam=lam,
            list_periods=list_periods,
            list_etas=list_etas,
            maxIter=maxIter,
            tol=tol,
            verbose=verbose,
            norm=norm,
        )

        self.burnin = burnin
        self.nwin = nwin
        self.online_tau = online_tau
        self.online_lam = online_lam
        self.online_list_etas = online_list_etas

    def fit_transform(
        self,
        signal: NDArray,
    ) -> None:
        """
        Compute an online version of RPCA with temporal regularisations

        Parameters
        ----------
        signal : Optional[ArrayLike], optional
            list of observations, by default None
        D: Optional
            array we want to denoise. If a signal is passed, D corresponds to that signal
        """
        D_init, n_add_values = self._prepare_data(signal=signal)
        burnin = self.burnin
        nwin = self.nwin

        m, n = D_init.shape
        Lhat, Shat, _ = super().fit(D=D_init[:, :burnin])

        proj_D = utils.impute_nans(D_init, method="median")
        approx_rank = utils.approx_rank(proj_D[:, :burnin], th=0.99)

        _, sigmas_hat, _ = np.linalg.svd(Lhat)

        if self.tau is None:
            self.tau = 1.0 / np.sqrt(max(m, n))
        if self.lam is None:
            self.lam = 1.0 / np.sqrt(max(m, n))

        # mburnin, _ = len(D_init) # D[:,:self.burnin].shape
        if self.online_tau is None:
            self.online_tau = self.tau
        if self.online_lam is None:
            self.online_lam = self.lam
        if len(self.online_list_etas) == 0:
            self.online_list_etas = self.list_etas.copy()

        Uhat, sigmas_hat, Vhat = randomized_svd(
            Lhat, n_components=approx_rank, n_iter=5, random_state=0
        )
        U = Uhat[:, :approx_rank].dot(np.sqrt(np.diag(sigmas_hat[:approx_rank])))
        if nwin == 0:
            Vhat_win = Vhat.copy()
        else:
            Vhat_win = Vhat[:, nwin:]
        A = np.zeros((approx_rank, approx_rank))
        B = np.zeros((m, approx_rank))
        for col in range(Vhat_win.shape[1]):
            sums = np.zeros(A.shape)
            for period, index in self.list_periods:
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

        lv = np.empty(shape=(n - burnin, proj_D.shape[1]), dtype=float)

        for row in range(burnin, n):
            ri = proj_D[:, row]
            vi, si = utils.solve_projection(
                ri,
                U,
                self.online_tau,
                self.online_lam,
                self.online_list_etas,
                self.list_periods,
                Lhat,
            )
            lv[row - burnin, :] = vi

            Shat = np.hstack((Shat, si.reshape(m, 1)))
            vi_delete = Vhat_win[:, 0]
            Vhat_win = np.hstack((Vhat_win[:, 1:], vi.reshape(approx_rank, 1)))

            if (len(self.list_periods) > 0) and (row >= max(self.list_periods)):
                sums = np.zeros((lv.shape[1], lv.shape[1]))
                for period, index in enumerate(self.list_periods):
                    vec = vi - lv[row - period - burnin]
                    sums += 2 * self.online_list_etas[index] * (np.outer(vec, vec))
                A = A + np.outer(vi, vi) + sums
            else:
                A = A + np.outer(vi, vi) - np.outer(vi_delete, vi_delete)
            if nwin == 0:
                B = B + np.outer(ri - si, vi)
            else:
                B = (
                    B
                    + np.outer(ri - si, vi)
                    - np.outer(proj_D[:, row - nwin] - Shat[:, row - nwin], vi_delete)
                )
            U = utils.update_col(self.online_tau, U, A, B)
            Lhat = np.hstack((Lhat, U.dot(vi).reshape(m, 1)))

        if n_add_values > 0:
            Lhat.flat[-n_add_values:] = np.nan
            Shat.flat[-n_add_values:] = np.nan

        if self.input_data == "2DArray":
            return Lhat, Shat
        elif self.input_data == "1DArray":
            ts_X = Lhat.flatten()
            ts_A = Shat.flatten()
            return ts_X[~np.isnan(ts_X)], ts_A[~np.isnan(ts_A)]
        else:
            raise ValueError("Data shape not recognized")

    def get_params(self):
        return {
            "n_rows": self.n_rows,
            "estimated_rank": self.rank,
            "tau": self.tau,
            "lam": self.lam,
            "list_periods": self.list_periods,
            "list_etas": self.list_etas,
            "maxIter": self.maxIter,
            "tol": self.tol,
            "verbose": self.verbose,
            "norm": self.norm,
            "burnin": self.burnin,
            "nwin": self.nwin,
            "online_tau": self.online_tau,
            "online_lam": self.online_lam,
            "online_list_etas": self.online_list_etas,
        }

    def get_params_scale(
        self,
        signal: Optional[ArrayLike] = None,
        D: Optional[NDArray] = None,
    ) -> None:
        D_init, _ = self._prepare_data(signal=signal, D=D)
        params_scale = super().get_params_scale(signal=signal, D=D)
        burnin = int(D_init.shape[1] * self.burnin)

        Lhat, _, _ = super().fit(D=D_init[:, :burnin])
        _, sigmas_hat, _ = np.linalg.svd(Lhat)
        online_tau = (
            1.0 / np.sqrt(len(D_init)) / np.mean(sigmas_hat[: params_scale["rank"]])
        )
        online_lam = 1.0 / np.sqrt(len(D_init))
        params_scale["online_tau"] = online_tau
        params_scale["online_lam"] = online_lam
        return params_scale
