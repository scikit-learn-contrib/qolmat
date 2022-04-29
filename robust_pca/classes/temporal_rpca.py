from __future__ import annotations
from typing import Optional, List

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.extmath import randomized_svd
import skopt

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
        period: Optional[int] = None,
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
        super().__init__(period=period,
                         maxIter=maxIter,
                         tol = tol,
                         verbose = verbose)
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
        H = [utils.toeplitz_matrix(period, n, model="column") for period in self.list_periods]

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

            sums = np.zeros((m,n))
            for index, _ in enumerate(self.list_periods):
                sums += (mu * R[index] - Y_[index])  @ H[index].T 
            X = (proj_D - A + mu * L @ Q.T - Y + sums) @ np.linalg.inv((1 + mu) * In + 2 * HHT)
            
            if np.any(np.isnan(proj_D)):
                A_omega = utils.soft_thresholding(self.proj_D - X, self.lam)
                A_omega = utils.ortho_proj(A_omega, omega, inverse=False)
                A_omega_C = proj_D - X
                A_omega_C = utils.ortho_proj(A_omega_C, omega, inverse=True)
                A = A_omega + A_omega_C
            else:
                A = utils.soft_thresholding(self.proj_D - X, self.lam)

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
                    print(
                        f"Converged in {iteration} iterations with error: {tol}"
                    )
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
        X = proj_D
        A = np.zeros((m, n))
        L = np.ones((m, self.rank))
        Q = np.ones((n, self.rank))

        mu = 1e-6
        mu_bar = mu * 1e10
        
        # matrices for temporal correlation
        H = [utils.toeplitz_matrix(period, n, model="column") for period in self.list_periods]
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
            X =  (proj_D - A + mu * L @ Q.T - Y) @ np.linalg.inv((1 + mu) * In + HHT)
            
            if np.any(~omega):
                A_omega = utils.soft_thresholding(proj_D - X, self.lam)
                A_omega = utils.ortho_proj(A_omega, omega, inverse=False)
                A_omega_C = proj_D - X
                A_omega_C = utils.ortho_proj(A_omega_C, omega, inverse=True)
                A = A_omega + A_omega_C
            else:
                A = utils.soft_thresholding(self.proj_D - X, self.lam)
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
                    print(
                        f"Converged in {iteration} iterations with error: {tol}"
                    )
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

    def fit_transform(
        self,
        signal: Optional[ArrayLike] = None,
        D: Optional[NDArray] = None
    ) -> None:
        """
        Compute the noisy RPCA with time "penalisations"

        Parameters
        ----------
        signal : Optional[ArrayLike], optional
            list of observations, by default None
        D : Optional[NDArray], optional
            array of observation, by default None
            
        Raises
        ------
        Exception
            The user has to give either a signal, either a matrix
        """

        D_init, ret = self._prepare_data(signal = signal, D = D)
        omega = (~np.isnan(D_init))
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
        if ret > 0:  
            X.flat[-ret:] = np.nan
            A.flat[-ret:] = np.nan
        
        if self.input_data == "2DArray":
            return X, A, errors
        elif self.input_data == "1DArray":
            ts_X = X.flatten()
            ts_A = A.flatten()
            ts_x = ts_X[~np.isnan(ts_X)]
            ts_A = ts_A[~np.isnan(ts_A)]
            return ts_X, ts_A, errors
        else:
            raise ValueError("input data type not recognized")
    
    def get_params_scale(
        self,
        signal: Optional[ArrayLike] = None,
        D: Optional[NDArray] = None,
    ) -> None:
        D_init, ret = self._prepare_data(signal=signal, D=D)
        proj_D = utils.impute_nans(D_init, method="median")
        print(np.sum(np.isnan(proj_D)))
        print(np.nanmedian(proj_D, axis=0))
        print(proj_D.shape)
        rank = utils.approx_rank(proj_D)
        tau = 1.0/np.sqrt(max(D_init.shape))
        lam = tau
        return {
            "rank":rank,
            "tau":tau,
            "lam":lam,
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
    TemporalRPCA : _type_
        _description_
    """
    def __init__(
        self,
        period: Optional[int] = None,
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
        online_tau: Optional[float]=None, 
        online_lam: Optional[float]=None, 
        online_list_periods: Optional[ArrayLike] = [],
        online_list_etas: Optional[ArrayLike] = []
    ) -> None:
        super().__init__(
                period = period,
                rank = rank,
                tau = tau,
                lam = lam,
                list_periods = list_periods,
                list_etas = list_etas,
                maxIter = maxIter,
                tol = tol,
                verbose = verbose,
                norm = norm
        )
    
        self.burnin = burnin
        self.nwin = nwin
        self.online_tau = online_tau
        self.online_lam = online_lam
        self.online_list_periods = online_list_periods
        self.online_list_etas = online_list_etas

    def fit_transform(
        self,
        signal: Optional[ArrayLike] = None,
        D: Optional[NDArray] = None,
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
        D_init, ret = self._prepare_data(signal=signal, D=D)
        burnin = int(D_init.shape[1]*self.burnin)
        nwin = int(D_init.shape[1]*self.nwin)
        
        m, n = D_init.shape
        Lhat, Shat, _ = super().fit(D=D_init[:,:burnin])

        proj_D = utils.impute_nans(D_init, method="median")
        approx_rank = utils.approx_rank(proj_D[:,:burnin], th=0.99)
        
        _, sigmas_hat, _ = np.linalg.svd(Lhat)   
   
        if self.tau is None:
            self.tau = 1.0/np.sqrt(max(m, n))
        if self.lam is None:
            self.lam = 1.0/np.sqrt(max(m, n))
            
        mburnin, _ = len(D_init) # D[:,:self.burnin].shape
        if self.online_tau is None:
            self.online_tau = 1.0/np.sqrt(mburnin)/np.mean(sigmas_hat[:approx_rank])
        if self.online_lam is None:
            self.online_lam = 1.0/np.sqrt(mburnin)
            
        Uhat, sigmas_hat, Vhat = randomized_svd(Lhat, n_components = approx_rank, n_iter=5, random_state=0)
        U = Uhat[:,:approx_rank].dot(np.sqrt(np.diag(sigmas_hat[:approx_rank])))
        if nwin == 0:
            Vhat_win = Vhat.copy()
        else:
            Vhat_win = Vhat[:, nwin:]
        A = np.zeros((approx_rank, approx_rank))
        B = np.zeros((m, approx_rank))
        for col in range(Vhat_win.shape[1]):
            sums = np.zeros(A.shape)
            for period, index in self.list_periods:
                vec = Vhat_win[:, col] - Vhat_win[:, col-period]
                sums += 2 * self.list_etas[index] * (np.outer(vec, vec))
            A = A + np.outer(Vhat_win[:, col], Vhat_win[:, col]) + sums
            
            if nwin == 0:
                B = B + np.outer(proj_D[:, col] - Shat[:, col], Vhat_win[:, col])
            else:
                B = B + np.outer(proj_D[:, burnin - nwin + col] - Shat[:, burnin - nwin + col], 
                                 Vhat_win[:, col])
    
        
        lv = np.empty(shape=(n-burnin, proj_D.shape[1]), dtype=float)
        
        for row in range(burnin, n):
            ri = proj_D[:, row]
            vi, si = utils.solve_projection(
                ri, 
                U, 
                self.online_tau, 
                self.online_lam, 
                self.online_list_etas, 
                self.online_list_periods, 
                Lhat
            )
            lv[row-burnin, :] = vi

            Shat = np.hstack((Shat, si.reshape(m,1)))
            vi_delete = Vhat_win[:,0]
            Vhat_win = np.hstack((Vhat_win[:,1:], vi.reshape(approx_rank,1)))
            
            if (len(self.online_list_periods) > 0) and (row >= max(self.online_list_periods)):
                sums = np.zeros((lv.shape[1], lv.shape[1]))
                for period, index in enumerate(self.online_list_periods):
                    vec = vi - lv[row - period - burnin]
                    sums += 2 * self.online_list_etas[index] * (np.outer(vec, vec))
                A = A + np.outer(vi, vi) + sums
            else:
                A = A + np.outer(vi, vi) - np.outer(vi_delete, vi_delete)
            if nwin == 0:
                B = B + np.outer(ri - si, vi)
            else:
                B = B + np.outer(ri - si, vi) - np.outer(proj_D[:, row - nwin] - Shat[:, row - nwin], vi_delete)
            U = utils.update_col(self.online_tau, U, A, B)
            Lhat = np.hstack((Lhat, U.dot(vi).reshape(m,1)))
        
        if self.input_data == "2DArray":
             return Lhat, Shat
        elif self.input_data == "1DArray":
            return Lhat.flatten(), Shat.flatten()
        else:
            raise ValueError("Data shape not recognized")
    
    def get_params(self):
        return {
            "period": self.period,
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
            "online_list_periods": self.online_list_periods,
            "online_list_etas": self.online_list_etas,
        }
    
    
    def get_params_scale(
        self,
        signal: Optional[ArrayLike] = None,
        D: Optional[NDArray] = None,
    ) -> None:
        D_init, _ = self._prepare_data(signal=signal, D=D)
        params_scale = super().get_params_scale(signal=signal, D=D)
        burnin = int(D_init.shape[1]*self.burnin)

        Lhat, _, _ = super().fit(D=D_init[:,:burnin])
        _, sigmas_hat, _ = np.linalg.svd(Lhat)   
        online_tau = 1.0/np.sqrt(len(D_init))/np.mean(sigmas_hat[:params_scale["rank"]])
        online_lam = 1.0/np.sqrt(len(D_init))
        params_scale["online_tau"] = online_tau
        params_scale["online_lam"] = online_lam
        return params_scale



class TemporalRPCAHyperparams(TemporalRPCA):
    """
    This class implements the noisy RPCA with hyperparameters' selection

    Parameters
    ----------
    NoisyRPCA : Type[NoisyRPCA]
        [description]
        
    hyperparams_tau : Optional[List[float]], optional
            list with 2 values: min and max for the search space for the param tau, by default []
    hyperparams_lam : Optional[List[float]], optional
        list with 2 values: min and max for the search space for the param lam, by default []
    hyperparams_etas : Optional[List[List[float]]], optional
        list of lists; each sublit contains 2 values: min and max for the search space for the assoiated param eta
        by default [[]]
    """
    
    def __init__(
        self,
        period: Optional[int] = None,
        rank: Optional[int] = None,
        tau: Optional[float] = None,
        lam: Optional[float] = None,
        list_periods: Optional[List[int]] = [],
        list_etas: Optional[List[float]] = [],
        maxIter: Optional[int] = int(1e4),
        tol: Optional[float] = 1e-6,
        verbose: Optional[bool] = False,
        norm: Optional[str] = "L2",
        hyperparams_tau: Optional[List[float]] = [],
        hyperparams_lam: Optional[List[float]] = [],
        hyperparams_etas: Optional[List[List[float]]] = [[]],
        cv:  Optional[int] = 5,
    ) -> None:
        super().__init__(
            period, rank, tau, lam, list_periods, list_etas, maxIter, tol, verbose, norm
        )
        
        self.cv = cv
        self.hyperparams_tau = hyperparams_tau
        self.hyperparams_lam = hyperparams_lam
        self.hyperparams_etas = hyperparams_etas
        self.add_hyperparams()
    
    def add_hyperparams(
        self,
    ) -> None:
        """Define the search space associated to each hyperparameter

        Parameters
        ----------
        hyperparams_tau : Optional[List[float]], optional
            list with 2 values: min and max for the search space for the param tau, by default []
        hyperparams_lam : Optional[List[float]], optional
            list with 2 values: min and max for the search space for the param lam, by default []
        hyperparams_etas : Optional[List[List[float]]], optional
            list of lists; each sublit contains 2 values: min and max for the search space for the assoiated param eta
            by default [[]]
        """
        self.search_space = []
        if len(self.hyperparams_tau) > 0:
            self.search_space.append(
                skopt.space.Real(
                    low=self.hyperparams_tau[0], high=self.hyperparams_tau[1], name="tau"
                )
            )
        if len(self.hyperparams_lam) > 0:
            self.search_space.append(
                skopt.space.Real(
                    low=self.hyperparams_lam[0], high=self.hyperparams_lam[1], name="lam"
                )
            )
        if len(self.hyperparams_etas[0]) > 0:  # TO DO: more cases
            for i in range(len(self.hyperparams_etas)):
                self.search_space.append(
                    skopt.space.Real(
                        low=self.hyperparams_etas[i][0],
                        high=self.hyperparams_etas[i][1],
                        name=f"eta_{i}",
                    )
                )

    def objective(self, args):
        """Define the objective function to minimise during the optimisation process

        Parameters
        ----------
        args : List[List]
            entire search space

        Returns
        -------
        float
            criterion to minimise
        """
        
        self.tau = args[0]
        self.lam = args[1]
        self.list_etas = [args[i + 2] for i in range(len(self.list_periods))]

        n1, n2 = self.initial_D.shape
        nb_missing = int(n1 * n2 * 0.05)

        errors = []
        for _ in range(self.cv):
            indices_x = np.random.choice(n1, nb_missing)
            indices_y = np.random.choice(n2, nb_missing)
            data_missing = self.initial_D.copy().astype("float")
            data_missing[indices_x, indices_y] = np.nan

            self.D = data_missing

            super().fit(D=data_missing)

            error = (
                np.linalg.norm(
                    self.initial_D[indices_x, indices_y]
                    - self.X[indices_x, indices_y],
                    1,
                )
                / nb_missing
            )
            if error == error:
                errors.append(error)

        if len(errors) == 0:
            print("Warning: not converged - return default 10^10")
            return 10 ** 10

        return np.mean(errors)

    def fit(
        self,
        signal: Optional[ArrayLike] = None,
        D: Optional[NDArray] = None,
    ) -> None:
        """Decompose a matrix into a low rank part and a sparse part
        Hyperparams are set by Bayesian optimisation and cross-validation 

        Parameters
        ----------
        signal : Optional[List[float]], optional
            list of observations, by default None
        D: Optional
            array we want to denoise. If a signal is passed, D corresponds to that signal
        """
        
        if (signal is None) and (D is None):
            raise Exception(
                "You have to provide either a time series (signal) or a matrix (D)"
            )
            
        self.signal = signal
        self.D = D
        
        self._prepare_data()
        
        res = skopt.gp_minimize(
            self.objective,
            self.search_space,
            n_calls=10,
            random_state=42,
            n_jobs=-1,
        )

        if self.verbose:
            print(f"Best parameters : {res.x}")
            print(f"Best result : {res.fun}")

        self.tau = res.x[0]
        self.lam = res.x[1]
        self.list_etas = res.x[2:]
        super().fit(D=self.D)

        return None