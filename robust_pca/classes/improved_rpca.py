from __future__ import annotations
from typing import Optional, Tuple, List, Type

import numpy as np
import pandas as pd
import skopt

from robust_pca.utils import utils

class ImprovedRPCA:
    """This class implements the improved RPCA decomposition with missing data using Alternating Lagrangian Multipliers.
    
    References
    ----------
    Wang, Xuehui, et al. "An improved robust principal component analysis model for anomalies detection of subway passenger flow." 
    Journal of advanced transportation 2018 (2018).
    
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
        signal: Optional[List[float]] = None,
        period: Optional[int] = None,
        D: Optional[np.ndarray] = None,
        rank: Optional[int] = None,
        lam: Optional[float] = None,
        list_periods: Optional[List[int]] = [],
        list_etas: Optional[List[float]] = [],
        maxIter: Optional[int] = int(1e4),
        tol: Optional[float] = 1e-6,
        verbose: bool = False,
    ) -> None:


        if (signal is None) and (D is None):
            raise Exception(
                "You must provide either a time series (signal) or a matrix (D)"
            )

        self.signal = signal
        self.period = period
        self.D = D
        self.rank = rank
        self.lam = lam
        self.list_periods = list_periods
        self.list_etas = list_etas
        self.maxIter = maxIter
        self.tol = tol
        self.verbose = verbose
        
        self._prepare_data()

    def _prepare_data(self) -> None:
        """Prepare data fot RPCA computation:
                Transform signal to matrix if needed
                Get the omega matrix
                Impute the nan values if needed
        """
        
        self.ret = 0
        if (self.D is None) and (self.period is None):
            self.period = utils.get_period(self.signal)
        if self.D is None:
            self.D, self.ret = utils.signal_to_matrix(self.signal, self.period)

        self.initial_D = self.D.copy()
        self.initial_D_proj = utils.impute_nans(self.initial_D, method="median")
        
        self.omega = 1 - (self.D != self.D)
        if np.isnan(np.sum(self.D)):
            self.proj_D = utils.impute_nans(self.D, method="median")
        else:
            self.proj_D = self.D
        
    def compute_improved_rpca(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decompose a matrix into a low rank part and a sparse part

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            the low rank matrix and the sparse matrix
        """

        self.omega = 1 - (self.D != self.D)
        if np.isnan(np.sum(self.D)):
            self.proj_D = utils.impute_nans(self.D, method="median")
        else:
            self.proj_D = self.D
        if self.rank is None:
            self.rank = utils.approx_rank(self.proj_D)

        rho = 1.1
        m, n = self.D.shape

        # init
        Y1 = np.ones((m, n))
        Y2 = np.ones((m, n))
        Y = dict()
        for i in range(len(self.list_periods)):
            Y[str(i)] = np.ones((m - self.list_periods[i], n))

        X = self.proj_D
        A = np.zeros((m, n))
        L = np.ones((m, self.rank))
        Q = np.ones((self.rank, n))
        S = dict()
        for i in range(len(self.list_periods)):
            S[str(i)] = np.ones((m - self.list_periods[i], n))

        mu = 1e-6
        mu_bar = mu * 1e10

        # matrices for temporal correlation
        H = dict()
        for i in range(len(self.list_periods)):
            H[str(i)] = utils.toeplitz_matrix(self.list_periods[i], m)

        Ik = np.eye(self.rank)
        Im = np.eye(m)

        ##
        HTH = np.zeros((m, m))
        for i in range(len(self.list_periods)):
            HTH += H[str(i)].T @ H[str(i)]
        X_tmp1 = np.linalg.inv(2 * Im + HTH)

        errors1, errors2, errors3 = [], [], []
        for iteration in range(self.maxIter):
            # save current variable value
            X_temp = X.copy()
            A_temp = A.copy()
            L_temp = L.copy()
            Q_temp = Q.copy()
            S_temp = S.copy()

            # solve X
            HTS = np.zeros((m, n))
            HTY = np.zeros((m, n))
            for i in range(len(self.list_periods)):
                HTS += H[str(i)].T @ S[str(i)]
                HTY += H[str(i)].T @ Y[str(i)]
            X_tmp2 = mu * (self.proj_D - A + (L @ Q) + HTS) + Y1 - Y2 - HTY
            X = (1 / mu * X_tmp1) @ X_tmp2

            # solve A
            if np.sum(np.isnan(self.D)) > 0:
                A_omega = utils.soft_thresholding(
                    self.proj_D - X + Y1 / mu, self.lam / mu
                )
                A_omega = utils.ortho_proj(A_omega, self.omega, inv=0)
                A_omega_C = self.proj_D - X + Y1 / mu
                A_omega_C = utils.ortho_proj(A_omega_C, self.omega, inv=1)
                A = A_omega + A_omega_C
            else:
                A = utils.soft_thresholding(
                    self.proj_D - X + Y1 / mu, self.lam / mu
                )

            # solve S
            for i in range(len(self.list_periods)):
                S[str(i)] = utils.soft_thresholding(
                    H[str(i)] @ X + Y[str(i)] / mu, self.list_etas[i] / mu
                )

            # solve L
            L_tmp1 = (mu * X + Y2) @ Q.T
            L_tmp2 = Ik + mu * Q @ Q.T
            L = L_tmp1 @ np.linalg.inv(L_tmp2)

            # solve Q
            Q_tmp1 = Ik + mu * L.T @ L
            Q_tmp2 = mu * L.T @ X + L.T @ Y2
            Q = np.linalg.inv(Q_tmp1) @ Q_tmp2

            # update Lagrangian multipliers
            Y1 += mu * (self.proj_D - X - A)
            Y2 += mu * (X - L @ Q)
            for i in range(len(self.list_periods)):
                Y[str(i)] += mu * (H[str(i)] @ X - S[str(i)])

            # update mu
            mu = min(mu * rho, mu_bar)

            # stopping criteria
            Xc = np.linalg.norm(X - X_temp, np.inf)
            Ac = np.linalg.norm(A - A_temp, np.inf)
            Lc = np.linalg.norm(L - L_temp, np.inf)
            Qc = np.linalg.norm(Q - Q_temp, np.inf)
            if len(self.list_periods) > 0:
                Sc = max(
                    [
                        np.linalg.norm(S[str(i)] - S_temp[str(i)], np.inf)
                        for i in range(len(self.list_periods))
                    ]
                )
            else:
                Sc = np.inf

            tol1 = max([Xc, Ac, Lc, Qc, Sc])

            if np.sum(np.isnan(self.D)) > 0:
                tol2 = np.linalg.norm(
                    self.proj_D - utils.impute_nans(X) - utils.impute_nans(A),
                    "fro",
                )
            else:
                tol2 = utils.l1_norm(self.proj_D - X - A)

            tol3 = np.linalg.norm(self.proj_D - X - A, "fro") / np.linalg.norm(
                self.proj_D, "fro"
            )

            errors1.append(tol1)
            errors2.append(tol2)
            errors3.append(tol3)

            if (tol1 < self.tol) and (tol2 < self.tol):
                if self.verbose:
                    print(
                        f"Converged in {iteration} iterations with error: {tol1} & {tol2}"
                    )
                break

            self.X = X
            self.A = A

        return self.initial_D, X, A


class ImprovedRPCAHyperparams(ImprovedRPCA):
    """This class implements the improved RPCA with hyperparameters' selection

    Parameters
    ----------
    ImprovedRPCA : Type[ImprovedRPCA]
        [description]
    """
        
    def add_hyperparams(
        self,
        hyperparams_lam: Optional[List[float]] = [],
        hyperparams_etas: Optional[List[List[float]]] = [[]],
        cv: Optional[int] = 5
    ) -> None:
        """Define the search space associated to each hyperparameter

        Parameters
        ----------
        hyperparams_lam : Optional[List[float]], optional
            list with 2 values: min and max for the search space for the param lam, by default []
        hyperparams_etas : Optional[List[List[float]]], optional
            list of lists; each sublit contains 2 values: min and max for the search space for the assoiated param eta
            by default [[]]
        cv: Optional[int], optional
            to specify the number of folds
        """
        self.cv = cv

        self.search_space = []
        if len(hyperparams_lam) > 0:
            self.search_space.append(
                skopt.space.Real(
                    low=hyperparams_lam[0], high=hyperparams_lam[1], name="lam"
                )
            )
        if len(hyperparams_etas[0]) > 0:  # TO DO: more cases
            for i in range(len(hyperparams_etas)):
                self.search_space.append(
                    skopt.space.Real(
                        low=hyperparams_etas[i][0],
                        high=hyperparams_etas[i][1],
                        name=f"eta_{i}",
                    )
                )

    def objective(self, args):
        """Define the objective function to minimise during the optimisation process

        Parameters
        ----------
        args : list[list]
            entire search space

        Returns
        -------
        float
            criterion to minimise
        """
        
        self.lam = args[0]
        self.list_etas = [args[i + 1] for i in range(len(self.list_periods))]

        n1, n2 = self.initial_D.shape
        nb_missing = int(n1 * n2 * 0.05)

        errors = []
        for _ in range(self.cv):
            indices_x = np.random.choice(n1, nb_missing)
            indices_y = np.random.choice(n2, nb_missing)
            data_missing = self.initial_D.copy().astype("float")
            data_missing[indices_x, indices_y] = np.nan

            self.D = data_missing

            _, X, _ = self.compute_improved_rpca()

            error = (
                np.linalg.norm(
                    self.initial_D_proj[indices_x, indices_y]
                    - X[indices_x, indices_y],
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

    def compute_improved_rpca_hyperparams(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decompose a matrix into a low rank part and a sparse part
        Hyperparams are set by Bayesian optimisation and cross-validation 

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            the low rank matrix and the sparse matrix
        """
        
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

        self.lam = res.x[0]
        self.list_etas = res.x[1:]
        D, X, A = self.compute_improved_rpca()

        return D, X, A