from __future__ import annotations
from re import X
from typing import Optional, Tuple, List

import numpy as np
from sklearn.utils.extmath import randomized_svd
import skopt

from robust_pca.utils import utils


class TemporalRPCA:
    """This class implements a noisy version of the so-called improved RPCA
    
    References
    ----------
    Wang, Xuehui, et al. "An improved robust principal component analysis model for anomalies detection of subway passenger flow." 
    Journal of advanced transportation 2018 (2018).
    
    Chen, Yuxin, et al. "Bridging convex and nonconvex optimization in robust PCA: Noise, outliers and missing data." 
    The Annals of Statistics 49.5 (2021): 2948-2971.
    
    Parameters
    ----------
    signal: Optional
        time series we want to denoise
    period: Optional
        period/seasonality of the signal
    D: Optional
        array we want to denoise. If a signal is passed, D corresponds to that signal
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
    maxIter: int, default = 1e4
        maximum number of iterations taken for the solvers to converge
    tol: float, default = 1e-6
        tolerance for stopping criteria
    verbose: bool, default = False
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

        self.period = period
        self.rank = rank
        self.lam1 = lam1
        self.lam2 = lam2
        self.list_periods = list_periods
        self.list_etas = list_etas
        self.maxIter = maxIter
        self.tol = tol
        self.verbose = verbose
        self.norm = norm
    

    def _prepare_data(self) -> None:
        """Prepare data fot RPCA computation:
                Transform signal to matrix if needed
                Get the omega matrix
                Impute the nan values if needed
        """
        
        self.rest = 0
        if (self.D is None) and (self.period is None):
            self.period = utils.get_period(self.signal)
        if self.D is None:
            self.D, self.rest = utils.signal_to_matrix(self.signal, self.period)

        self.initial_D = self.D.copy()
        self.initial_D_proj = utils.impute_nans(self.initial_D, method="median")
        
        self.omega = 1 - (self.D != self.D)
        if np.isnan(np.sum(self.D)):
            self.proj_D = utils.impute_nans(self.D, method="median")
        else:
            self.proj_D = self.D

    def compute_L1(self) -> None:
        """compute RPCA with possible tmeporal regularisations, penalised with L1 norm 
        """
        m, n = self.D.shape
        p = len(self.list_periods)
        rho = 1.1
        mu = 1e-6
        mu_bar = mu * 1e10

        # init
        Y = np.ones((m, n))
        Y_ = dict()
        for i in range(p):
            Y_[str(i)] = np.ones((m, n - self.list_periods[i]))

        X = self.proj_D
        A = np.zeros((m, n))
        L = np.ones((m, self.rank))
        Q = np.ones((n, self.rank))
        # U, s, Vt = np.linalg.svd(self.proj_D, full_matrices=False)
        # L = U[:, : self.rank] @ np.sqrt(np.diag(s[: self.rank]))
        # Q = Vt[: self.rank, :].T @ np.sqrt(np.diag(s[: self.rank])).T
        R = {}
        for i in range(p):
            R[str(i)] = np.ones((m, n - self.list_periods[i]))

        # temporal correlations
        H = dict()
        for i in range(p):
            H[str(i)] = utils.toeplitz_matrix(self.list_periods[i], n, model="column")

        Ir = np.eye(self.rank)
        In = np.eye(n)

        ##
        HHT = np.zeros((n, n))
        for i in range(len(self.list_periods)):
            HHT += self.list_etas[i] * (H[str(i)] @ H[str(i)].T)


        errors = []
        for iteration in range(self.maxIter):
            # save current variable values
            X_temp = X.copy()
            A_temp = A.copy()
            L_temp = L.copy()
            Q_temp = Q.copy()
            R_temp = R.copy()

            
            sums = np.zeros((m,n))
            for i in range(p):
                sums += (mu * R[str(i)] - Y_[str(i)])  @ H[str(i)].T 
            X = (self.proj_D - A + mu * L @ Q.T - Y + sums) @ np.linalg.inv((1 + mu) * In + 2 * HHT)
            
            if np.sum(np.isnan(self.D)) > 0:
                A_omega = utils.soft_thresholding(self.proj_D - X, self.lam2)
                A_omega = utils.ortho_proj(A_omega, self.omega, inv=0)
                A_omega_C = self.proj_D - X
                A_omega_C = utils.ortho_proj(A_omega_C, self.omega, inv=1)
                A = A_omega + A_omega_C
            else:
                A = utils.soft_thresholding(self.proj_D - X, self.lam2)

            L = (mu * X + Y) @ Q @ np.linalg.inv(self.lam1 * Ir + mu * (Q.T @ Q))
            Q = (mu * X.T + Y.T) @ L @ np.linalg.inv(self.lam1 * Ir + mu * (L.T @ L))

            for i in range(p):
                R[str(i)] = utils.soft_thresholding(
                    X @ H[str(i)].T - Y_[str(i)] / mu, self.list_etas[i] / mu
                )
                
            Y += mu * (X - L @ Q.T)
            for i in range(p):
                Y_[str(i)] += mu * (X @ H[str(i)].T - R[str(i)])

            # update mu
            mu = min(mu * rho, mu_bar)

            # stopping criteria
            Xc = np.linalg.norm(X - X_temp, np.inf)
            Ac = np.linalg.norm(A - A_temp, np.inf)
            Lc = np.linalg.norm(L - L_temp, np.inf)
            Qc = np.linalg.norm(Q - Q_temp, np.inf)
            Rc = -1
            for i in range(p):
                Rc = max(Rc, np.linalg.norm(R[str(i)] - R_temp[str(i)], np.inf))
            tol = max([Xc, Ac, Lc, Qc, Rc])

            errors.append(tol)

            if tol < self.tol:
                if self.verbose:
                    print(
                        f"Converged in {iteration} iterations with error: {tol}"
                    )
                break

        self.X = X
        self.A = A
        self.errors = errors
        
        return None
        
    def compute_L2(self) -> None:
        """compute RPCA with possible tmeporal regularisations, penalised with L1 norm
        """
        rho = 1.1
        m, n = self.D.shape
        
        # init
        Y = np.ones((m, n))

        
        X = self.proj_D
        A = np.zeros((m, n))
        L = np.ones((m, self.rank))
        Q = np.ones((n, self.rank))

        mu = 1e-6
        mu_bar = mu * 1e10
        
        # matrices for temporal correlation
        H = dict()
        for i in range(len(self.list_periods)):
            H[str(i)] = utils.toeplitz_matrix(self.list_periods[i], n, model="column")

        HHT = np.zeros((n, n))
        for i in range(len(self.list_periods)):
            HHT += self.list_etas[i] * (H[str(i)] @ H[str(i)].T)


        Ir = np.eye(self.rank)
        In = np.eye(n)
            
    
        errors = []
        for iteration in range(self.maxIter):
            
            X_temp = X.copy()
            A_temp = A.copy()
            L_temp = L.copy()
            Q_temp = Q.copy()
            
            
            X =  (self.proj_D - A + mu * L @ Q.T - Y) @ np.linalg.inv((1 + mu) * In + HHT)
            if np.sum(np.isnan(self.D)) > 0:
                A_omega = utils.soft_thresholding(self.proj_D - X, self.lam2)
                A_omega = utils.ortho_proj(A_omega, self.omega, inv=0)
                A_omega_C = self.proj_D - X
                A_omega_C = utils.ortho_proj(A_omega_C, self.omega, inv=1)
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

        self.X = L @ Q.T
        self.A = A
        self.errors = errors
        
        return None
    
    def get_params(self) -> dict:
        return {
            "period": self.period,
            "lam1": self.lam1,
            "lam2": self.lam2,
            "list_periods": self.list_periods,
            "list_etas": self.list_etas,
            "maxIter": self.maxIter,
            "norm": self.norm,
            "tol": self.tol,
            "verbose": self.verbose
        }
        
    def fit(
        self,
        signal: Optional[List[float]] = None,
        D: Optional[np.ndarray] = None
    ) -> None:
        """Compute the noisy RPCA with time "penalisations"

        Parameters
        ----------
        signal : Optional[List[float]], optional
            list of observations, by default None
        D : Optional[np.ndarray], optional
            array of observation, by default None
            
        Raises
        ------
        Exception
            The user has to give either a signal, either a matrix
        """

        if (signal is None) and (D is None):
            raise Exception(
                "You have to provide either a time series (signal) or a matrix (D)"
            )
            
        self.signal = signal
        self.D = D
        
        self._prepare_data()

        self.omega = 1 - (self.D != self.D)

        if np.isnan(np.sum(self.D)):
            self.proj_D = utils.impute_nans(self.D, method="median")
        else:
            self.proj_D = self.D

        if self.rank is None:
            self.rank = utils.approx_rank(self.proj_D)

        if self.norm == "L1":
            self.compute_L1()
            
        elif self.norm == "L2":
            self.compute_L2()
        
        return None


class TemporalRPCAHyperparams(TemporalRPCA):
    """This class implements the noisy RPCA with hyperparameters' selection

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
        signal: Optional[List[float]] = None,
        D: Optional[np.ndarray] = None,
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
    
    

class OnlineTemporalRPCA(TemporalRPCA):
    """This class implements an online version of Temporal RPCA 
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
        online_list_periods: Optional[List[float]] = [],
        online_list_etas: Optional[List[float]] = []
    ) -> None:
        super().__init__(
            period, rank, lam1, lam2, list_periods, list_etas, maxIter, tol, verbose, norm
        )
    
        self.burnin = burnin
        self.nwin = nwin
        self.online_lam1 = online_lam1
        self.online_lam2 = online_lam2
        self.online_list_periods = online_list_periods
        self.online_list_etas = online_list_etas

        
    def fit(
        self,
        signal: Optional[List[float]] = None,
        D: Optional[np.ndarray] = None,
    ) -> None:
        """Compute an online version of RPCA with temporal regularisations

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
        
        self.burnin = int(self.D.shape[1]*self.burnin)

        m, n = self.initial_D.shape
        
        a = TemporalRPCA(
                    period=self.period,
                    lam1=self.lam1,
                    lam2=self.lam2,
                    list_periods=self.list_periods,
                    list_etas=self.list_etas,
                    maxIter=self.maxIter,
                    tol=self.tol,
                    verbose=self.verbose,
                    norm=self.norm
                )
        a.fit(D=self.D[:,:self.burnin])
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