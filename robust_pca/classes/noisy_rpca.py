from __future__ import annotations
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import skopt

from robust_pca.utils import utils


class NoisyRPCA:
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
        signal: Optional[List[float]] = None,
        period: Optional[int] = None,
        D: Optional[np.ndarray] = None,
        rank: Optional[int] = None,
        lam: Optional[float] = None,
        tau: Optional[float] = None,
        list_periods: Optional[List[int]] = [],
        list_etas: Optional[List[float]] = [],
        maxIter: Optional[int] = int(1e4),
        tol: Optional[float] = 1e-6,
        verbose: bool = False,
    ) -> None:

        if (signal is None) and (D is None):
            raise Exception(
                "You have to provide either a time series (signal) or a matrix (D)"
            )

        self.signal = signal
        self.period = period
        self.D = D
        self.rank = rank
        self.lam = lam
        self.tau = tau
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

    def compute_noisy_rpca(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the noisy RPCA with time "penalisations"

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            observations, low-rank and sparse matrices.
        """

        omega = 1 - (self.D != self.D)

        if np.isnan(np.sum(self.D)):
            self.proj_D = utils.impute_nans(self.D, method="median")
        else:
            self.proj_D = self.D

        if self.rank is None:
            self.rank = utils.approx_rank(self.proj_D)

        m, n = self.D.shape
        p = len(self.list_periods)
        q = 1
        rho = 1.1
        mu = 1e-6
        mu_bar = mu * 1e10

        # init
        Y0 = np.ones((m, n))
        Y_ = {}
        for i in range(p):
            Y_[str(i)] = np.ones((m - self.list_periods[i], n))

        W = self.proj_D
        U, s, Vt = np.linalg.svd(self.proj_D, full_matrices=False)
        X = U[:, : self.rank] @ np.sqrt(np.diag(s[: self.rank]))
        Y = Vt[: self.rank, :].T @ np.sqrt(np.diag(s[: self.rank])).T
        S = np.zeros((m, n))
        R = {}
        for i in range(p):
            R[str(i)] = np.ones((m - self.list_periods[i], n))

        # temporal correlations
        H = {}
        for i in range(p):
            H[str(i)] = utils.toeplitz_matrix(self.list_periods[i], m)

        Ik = np.eye(self.rank)
        Im = np.eye(m)

        ##
        HTH = np.zeros((m, m))
        for i in range(p):
            HTH += H[str(i)].T @ H[str(i)]

        errors1, errors2 = [], []
        for iteration in range(self.maxIter):

            # save current variable values
            W_temp = W.copy()
            S_temp = S.copy()
            X_temp = X.copy()
            Y_temp = Y.copy()
            R_temp = R.copy()

            # update W
            HTR = np.zeros((m, n))
            HTY = np.zeros((m, n))
            for i in range(p):
                HTR += H[str(i)].T @ R[str(i)]
                HTY += H[str(i)].T @ Y_[str(i)]

            W_tmp1 = np.linalg.inv((1 / q + mu) * Im + mu * HTH)
            W_tmp2 = (
                1 / q * (self.proj_D - S) + mu * X @ Y.T - Y0 + mu * HTR + HTY
            )
            W = W_tmp1 @ W_tmp2

            if np.sum(np.isnan(self.D)) > 0:
                W_tmp1 = np.linalg.inv(mu * Im + mu * HTH)
                W_tmp2 = mu * X @ Y.T - Y0 + mu * HTR + HTY
                W_omegaC = W_tmp1 @ W_tmp2
                W = utils.ortho_proj(W, omega, inv=0) + utils.ortho_proj(
                    W_omegaC, omega, inv=1
                )

            # update S
            if np.sum(np.isnan(self.D)) > 0:
                S = utils.soft_thresholding(
                    self.proj_D - utils.impute_nans(X @ Y.T) + Y0 / mu, self.tau / mu
                )
            else:
                S = utils.soft_thresholding(self.proj_D - X @ Y.T + Y0  / mu, self.tau / mu)

            # update X
            X = (mu * W @ Y + Y0 @ Y) @ np.linalg.inv(
                (self.lam / q) * Ik + mu * Y.T @ Y
            )

            # update Y
            Y = (
                (mu * W.T + Y0.T)
                @ X
                @ np.linalg.inv((self.lam / q) * Ik + mu * X.T @ X)
            )

            # update R
            for i in range(p):
                R[str(i)] = utils.soft_thresholding(
                    H[str(i)] @ W - Y_[str(i)] / mu, self.list_etas[i] / mu
                )

            # update Lagrangian multipliers
            Y0 += mu * (W - X @ Y.T)
            for i in range(p):
                Y_[str(i)] += mu * (R[str(i)] - H[str(i)] @ W)

            # update mu
            mu = min(mu * rho, mu_bar)

            # stopping criteria
            Wc = np.linalg.norm(W - W_temp, np.inf)
            Sc = np.linalg.norm(S - S_temp, np.inf)
            Xc = np.linalg.norm(X - X_temp, np.inf)
            Yc = np.linalg.norm(Y - Y_temp, np.inf)
            Rc = -1
            for i in range(p):
                Rc = max(Rc, np.linalg.norm(R[str(i)] - R_temp[str(i)], np.inf))
            tol1 = max([Wc, Sc, Xc, Yc, Rc])

            errors1.append(tol1)
            errors2.append(
                np.linalg.norm(self.proj_D - W - S, "fro")
                / np.linalg.norm(self.proj_D, "fro")
            )

            if tol1 < self.tol:
                if self.verbose:
                    print(
                        f"Converged in {iteration} iterations with error: {tol1}"
                    )
                break

            self.W = W
            self.S = S

        return self.initial_D, W, S


class NoisyRPCAHyperparams(NoisyRPCA):
    """This class implement the noisy RPCA with hyperparameters' selection

    Parameters
    ----------
    NoisyRPCA : Type[NoisyRPCA]
        [description]
    """
    def add_hyperparams(
        self,
        hyperparams_tau: Optional[List[float]] = [],
        hyperparams_lam: Optional[List[float]] = [],
        hyperparams_etas: Optional[List[List[float]]] = [[]],
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
        if len(hyperparams_lam) > 0:
            self.search_space.append(
                skopt.space.Real(
                    low=hyperparams_lam[0], high=hyperparams_lam[1], name="lam"
                )
            )
        if len(hyperparams_tau) > 0:
            self.search_space.append(
                skopt.space.Real(
                    low=hyperparams_tau[0], high=hyperparams_tau[1], name="tau"
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
        args : List[List]
            entire search space

        Returns
        -------
        float
            criterion to minimise
        """
        self.lam = args[0]
        self.tau = args[1]
        self.list_etas = [args[i + 2] for i in range(len(self.list_periods))]

        n1, n2 = self.initial_D.shape
        nb_missing = int(n1 * n2 * 0.05)

        errors = []
        for _ in range(2):
            indices_x = np.random.choice(n1, nb_missing)
            indices_y = np.random.choice(n2, nb_missing)
            data_missing = self.initial_D.copy().astype("float")
            data_missing[indices_x, indices_y] = np.nan

            self.D = data_missing

            _, W, _ = self.compute_improve_rpca()

            error = (
                np.linalg.norm(
                    self.initial_D[indices_x, indices_y]
                    - W[indices_x, indices_y],
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

    def compute_improve_rpca_hyperparams(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        self.tau = res.x[1]
        self.list_etas = res.x[2:]
        D, W, S = self.compute_noisy_rpca()

        return D, W, S
