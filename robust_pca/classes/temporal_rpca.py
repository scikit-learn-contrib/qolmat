from __future__ import annotations
from re import X
from typing import Optional, Tuple, List

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
        lam1: Optional[float] = None,
        lam2: Optional[float] = None,
        list_periods: Optional[List[int]] = [],
        list_etas: Optional[List[float]] = [],
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
        self.lam1 = lam1
        self.lam2 = lam2
        self.list_periods = list_periods
        self.list_etas = list_etas
        self.norm = norm
    
    def compute_L1(self, proj_D, omega) -> None:
        """
        compute RPCA with possible temporal regularisations, penalised with L1 norm 
        """
        m, n = D.shape
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
            
            if np.any(np.isnan(D)):
                A_omega = utils.soft_thresholding(self.proj_D - X, self.lam2)
                A_omega = utils.ortho_proj(A_omega, omega, inverse=False)
                A_omega_C = proj_D - X
                A_omega_C = utils.ortho_proj(A_omega_C, omega, inverse=True)
                A = A_omega + A_omega_C
            else:
                A = utils.soft_thresholding(self.proj_D - X, self.lam2)

            L = (mu * X + Y) @ Q @ np.linalg.inv(self.lam1 * Ir + mu * (Q.T @ Q))
            Q = (mu * X.T + Y.T) @ L @ np.linalg.inv(self.lam1 * Ir + mu * (L.T @ L))

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
        m, n = self.D.shape

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
            
            if np.any(np.isnan(self.D)):
                A_omega = utils.soft_thresholding(proj_D - X, self.lam2)
                A_omega = utils.ortho_proj(A_omega, omega, inverse=False)
                A_omega_C = proj_D - X
                A_omega_C = utils.ortho_proj(A_omega_C, omega, inverse=True)
                A = A_omega + A_omega_C
            else:
                A = utils.soft_thresholding(self.proj_D - X, self.lam2)
            L = (mu * X + Y) @ Q @ np.linalg.inv(self.lam1 * Ir + mu * (Q.T @ Q))
            Q = (mu * X.T + Y.T) @ L @ np.linalg.inv(self.lam1 * Ir + mu * (L.T @ L))
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
        dict_params["lam1"] = self.lam1
        dict_params["lam2"] = self.lam2
        dict_params["list_periods"] = self.list_periods
        dict_params["list_etas"] = self.list_etas
        dict_params["norm"] = self.norm
        return dict_params

    def fit(
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

        if self.norm == "L1":
            X, A, E = self.compute_L1(proj_D, omega)
        elif self.norm == "L2":
            X, A, E = self.compute_L2(proj_D, omega)
        X = X.T
        A = A.T
        E = E.T
        X.flat[-ret:] = np.nan
        A.flat[-ret:] = np.nan
        E.flat[-ret:] = np.nan
        self.X = X
        self.A = A
        self.E = E
        return self
    
    def transform(self):
        if self.input_data == "2DArray":
            return self.X + self.E
        elif self.input_data == "1DArray":
            return (self.X + self.E).flatten()
        else:
            raise ValueError("input data type not recognized")

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
        lam1: Optional[float] = None,
        lam2: Optional[float] = None,
        list_periods: Optional[List[int]] = [],
        list_etas: Optional[List[float]] = [],
        maxIter: Optional[int] = int(1e4),
        tol: Optional[float] = 1e-6,
        verbose: Optional[bool] = False,
        norm: Optional[str] = "L2",
        burnin: Optional[float] = 0,
        nwin: Optional[int]=0,
        online_lam1: Optional[float]=None, 
        online_lam2: Optional[float]=None, 
        online_list_periods: Optional[ArrayLike] = [],
        online_list_etas: Optional[ArrayLike] = []
    ) -> None:
        super().__init__(
                period = period,
                rank = rank,
                lam1 = lam1,
                lam2 = lam2,
                list_periods = list_periods,
                list_etas = list_etas,
                maxIter = maxIter,
                tol = tol,
                verbose = verbose,
                norm = norm
        )
    
        self.burnin = burnin
        self.nwin = nwin
        self.online_lam1 = online_lam1
        self.online_lam2 = online_lam2
        self.online_list_periods = online_list_periods
        self.online_list_etas = online_list_etas

    def fit(
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
        
        self.burnin = int(D.shape[1]*self.burnin)

        m, n = D_init.shape

        a = super().fit(D=D_init[:,:self.burnin])
        Lhat = a.X
        Shat = a.A
        
        r = utils.approx_rank(self.proj_D[:,:self.burnin], th=0.99)
        
        _, sigmas_hat, _ = np.linalg.svd(Lhat)   
   
        if self.lam1 is None:
            self.lam1 = 1.0/np.sqrt(max(m, n))
        if self.lam2 is None:
            self.lam2 = 1.0/np.sqrt(max(m, n))
            
        mburnin,nburnin = self.D[:,:self.burnin].shape
        if self.online_lam1 is None:
            self.online_lam1 = 1.0/np.sqrt(mburnin)/np.mean(sigmas_hat[:r])
        if self.online_lam2 is None:
            self.online_lam2 = 1.0/np.sqrt(mburnin)
            
            
        self.D = utils.impute_nans(self.D, method="median")
        
        #U = np.random.rand(m,r)
        Uhat, sigmas_hat, Vhat = randomized_svd(Lhat, n_components=r, n_iter=5, random_state=0)
        U = Uhat[:,:r].dot(np.sqrt(np.diag(sigmas_hat[:r])))
        if self.nwin == 0:
            Vhat_win = Vhat.copy()
        else:
            Vhat_win = Vhat[:, -self.nwin:]
        A = np.zeros((r, r))
        B = np.zeros((m, r))
        for i in range(Vhat_win.shape[1]):
            sums = np.zeros(A.shape)
            for k in range(len(self.list_periods)):
                vec = Vhat_win[:, i] - Vhat_win[:, i-self.list_periods[k]]
                sums += 2 * self.list_etas[k] * (np.outer(vec, vec))
            A = A + np.outer(Vhat_win[:, i], Vhat_win[:, i]) + sums
            
            if self.nwin == 0:
                B = B + np.outer(self.D[:, i] - Shat[:, i], Vhat_win[:, i])
            else:
                B = B + np.outer(self.D[:, self.burnin - self.nwin + i] - Shat[:, self.burnin - self.nwin + i], 
                             Vhat_win[:, i])
    
        
        lv = []
        for i in range(self.burnin, n):
            ri = self.D[:, i]
            vi, si = utils.solve_projection(
                ri, 
                U, 
                self.online_lam1, 
                self.online_lam2, 
                self.online_list_etas, 
                self.online_list_periods, 
                Lhat
            )
            lv.append(vi)
            Shat = np.hstack((Shat, si.reshape(m,1)))
            vi_delete = Vhat_win[:,0]
            Vhat_win = np.hstack((Vhat_win[:,1:], vi.reshape(r,1)))
            if (len(self.online_list_periods) > 0) and (len(lv) > max(self.online_list_periods)):
                sums = np.zeros((len(vi), len(vi)))
                for k in range(len(self.online_list_periods)):
                    vec = vi - lv[i-self.online_list_periods[k]-self.burnin]
                    sums += 2 * self.online_list_etas[k] * (np.outer(vec, vec))
                A = A + np.outer(vi, vi) + sums
            else:
                A = A + np.outer(vi, vi) - np.outer(vi_delete, vi_delete)
            if self.nwin == 0:
                B = B + np.outer(ri - si, vi)
            else:
                B = B + np.outer(ri - si, vi) - np.outer(self.D[:, i - self.nwin] - Shat[:, i - self.nwin], vi_delete)
            U = utils.update_col(self.online_lam1, U, A, B)
            Lhat = np.hstack((Lhat, U.dot(vi).reshape(m,1)))
            
        self.D = self.initial_D
        self.X = Lhat
        self.A = Shat
            
        return None
    
    def get_params(self):
        return {
            "period": self.period,
            "estimated_rank": self.rank,
            "lam1": self.lam1,
            "lam2": self.lam2,
            "list_periods": self.list_periods,
            "list_etas": self.list_etas,
            "maxIter": self.maxIter,
            "tol": self.tol,
            "verbose": self.verbose,
            "norm": self.norm,
            "burnin": self.burnin,
            "nwin": self.nwin,
            "online_lam1": self.online_lam1,
            "online_lam2": self.online_lam2,
            "online_list_periods": self.online_list_periods,
            "online_list_etas": self.online_list_etas,
        }

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
        lam1: Optional[float] = None,
        lam2: Optional[float] = None,
        list_periods: Optional[List[int]] = [],
        list_etas: Optional[List[float]] = [],
        maxIter: Optional[int] = int(1e4),
        tol: Optional[float] = 1e-6,
        verbose: Optional[bool] = False,
        norm: Optional[str] = "L2",
        hyperparams_lam1: Optional[List[float]] = [],
        hyperparams_lam2: Optional[List[float]] = [],
        hyperparams_etas: Optional[List[List[float]]] = [[]],
        cv:  Optional[int] = 5,
    ) -> None:
        super().__init__(
            period, rank, lam1, lam2, list_periods, list_etas, maxIter, tol, verbose, norm
        )
        
        self.cv = cv
        self.hyperparams_lam1 = hyperparams_lam1
        self.hyperparams_lam2 = hyperparams_lam2
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
        if len(self.hyperparams_lam1) > 0:
            self.search_space.append(
                skopt.space.Real(
                    low=self.hyperparams_lam1[0], high=self.hyperparams_lam1[1], name="lam1"
                )
            )
        if len(self.hyperparams_lam2) > 0:
            self.search_space.append(
                skopt.space.Real(
                    low=self.hyperparams_lam2[0], high=self.hyperparams_lam2[1], name="tau2"
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
        
        self.lam1 = args[0]
        self.lam2 = args[1]
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

        self.lam1 = res.x[0]
        self.lam2 = res.x[1]
        self.list_etas = res.x[2:]
        super().fit(D=self.D)

        return None