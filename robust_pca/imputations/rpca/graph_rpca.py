from __future__ import annotations
from typing import Optional, Tuple, List, Type

import numpy as np
import skopt

from robust_pca.utils import utils


class GraphRPCA:
    """This class implements Fast Robust PCA on Graphs using the FISTA algorithm

    References
    ----------
    Shahid, Nauman, et al. "Fast robust PCA on graphs."
    IEEE Journal of Selected Topics in Signal Processing 10.4 (2016): 740-756.

    Parameters
    ----------
    period : Optional[int]
        period/seasonality of the signal
    rank : Optional[int]
        (estimated) low-rank of the matrix D
    gamma1 : int
        regularizing parameter for the graph G1, constructed from the columns of D
    gamma2 : int
        regularizing parameter for the graph G1, constructed from the rows of D
    G1 : Optional[np.ndarray]
        graph G1, constructed from the columns of D
    G2 : Optional[np.ndarray]
        graph G2, constructed from the rows of D
    nbg1 : Optional[int]
        number of closest neighbors to construct graph G1, default=10
    nbg2 : Optional[int]
        number of closest neighbors to construct graph G2, default=10
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
        gamma1: Optional[float] = None,
        gamma2: Optional[float] = None,
        G1: Optional[np.ndarray] = None,
        G2: Optional[np.ndarray] = None,
        nbg1: Optional[int] = 10,
        nbg2: Optional[int] = 10,
        maxIter: Optional[int] = int(1e4),
        tol: Optional[float] = 1e-6,
        cv: Optional[int] = 5,
        verbose: Optional[bool] = False,
    ) -> None:
        self.period = period
        self.rank = rank
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.G1 = G1
        self.G2 = G2
        self.nbg1 = nbg1
        self.nbg2 = nbg2
        self.maxIter = maxIter
        self.tol = tol
        self.verbose = verbose

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

    def fit(
        self,
        signal: Optional[List] = None,
        D: Optional[np.ndarray] = None,
    ) -> None:
        """Compute the RPCA on graph.

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

        self.omega = 1 - (self.D != self.D)
        if np.isnan(np.sum(self.D)):
            self.proj_D = utils.impute_nans(self.D, method="median")
        else:
            self.proj_D = self.D
        if self.rank is None:
            self.rank = utils.approx_rank(self.proj_D)

        if self.G1 is None:
            self.G1 = utils.construct_graph((self.D).T, n_neighbors=self.nbg1)
        if self.G2 is None:
            self.G2 = utils.construct_graph((self.D), n_neighbors=self.nbg2)

        laplacian1 = utils.get_laplacian(self.G1)
        laplacian2 = utils.get_laplacian(self.G2)

        X = self.D.copy()
        Y = self.D.copy()
        t_past = 1

        lam = 1 / (
            2 * self.gamma1 * np.linalg.norm(laplacian1, 2)
            + 2 * self.gamma2 * np.linalg.norm(laplacian2, 2)
        )

        errors = []
        for iterations in range(self.maxIter):

            X_past = X.copy()
            Y_past = Y.copy()

            grad_g = 2 * (self.gamma1 * Y @ laplacian1 + self.gamma2 * laplacian2 @ Y)

            X = utils.proximal_operator(Y_past - lam * grad_g, self.D, lam)
            t = (1 + (1 + 4 * t_past**2) ** 0.5) / 2
            Y = X + (t_past - 1) / t * (X - X_past)

            errors.append(
                np.linalg.norm(Y - Y_past, "fro") / np.linalg.norm(Y_past, "fro")
            )
            if errors[-1] < self.tol:
                if self.verbose:
                    print(
                        f"Converged in {iterations} iterations with an error equals to {errors[-1]}."
                    )
                break

            t = t_past

        self.errors = errors

        self.X = X
        self.A = self.initial_D - X

        return None


####################################################################
# TO DO
# construct graph with missing values
# important for the cross validation to work
####################################################################
class GraphRPCAHyperparams(GraphRPCA):
    """This class implements the graph RPCA with hyperparameters' selection

    Parameters
    ----------
    GraphRPCA : Type[GraphRPCA]
        [description]
    hyperparams_gamma1 : Optional[List[float]], optional
            list with 2 values: min and max for the search space for the param gamma1, by default []
    hyperparams_gamma2 : Optional[List[float]], optional
        list with 2 values: min and max for the search space for the param gamma2, by default []
    cv: Optional[int], optional
        to specify the number of folds
    """

    def __init__(
        self,
        period: Optional[int] = None,
        rank: Optional[int] = None,
        gamma1: Optional[float] = None,
        gamma2: Optional[float] = None,
        G1: Optional[np.ndarray] = None,
        G2: Optional[np.ndarray] = None,
        nbg1: Optional[int] = 10,
        nbg2: Optional[int] = 10,
        maxIter: Optional[int] = int(1e4),
        tol: Optional[float] = 1e-6,
        hyperparams_gamma1: Optional[List[float]] = [],
        hyperparams_gamma2: Optional[List[float]] = [],
        cv: Optional[int] = 5,
        verbose: Optional[bool] = False,
    ) -> None:
        super().__init__(
            period, rank, gamma1, gamma2, G1, G2, nbg1, nbg2, maxIter, tol, verbose
        )

        self.cv = cv
        self.hyperparams_gamma1 = hyperparams_gamma1
        self.hyperparams_gamma2 = hyperparams_gamma2
        self.add_hyperparams()

    def add_hyperparams(
        self,
    ) -> None:
        """Define the search space associated to each hyperparameter"""
        self.search_space = []
        if len(self.hyperparams_gamma1) > 0:
            self.search_space.append(
                skopt.space.Real(
                    low=self.hyperparams_gamma1[0],
                    high=self.hyperparams_gamma1[1],
                    name="gamma1",
                )
            )
        if len(self.hyperparams_gamma2) > 0:
            self.search_space.append(
                skopt.space.Real(
                    low=self.hyperparams_gamma2[0],
                    high=self.hyperparams_gamma2[1],
                    name="gamma2",
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

        self.gamma1 = args[0]
        self.gamma2 = args[1]

        n1, n2 = self.initial_D.shape
        nb_missing = int(n1 * n2 * 0.05)

        errors = []
        for _ in range(self.cv):
            indices_x = np.random.choice(n1, nb_missing)
            indices_y = np.random.choice(n2, nb_missing)
            data_missing = self.initial_D.copy().astype("float")
            data_missing[indices_x, indices_y] = np.nan

            self.D = data_missing

            # _, W, _ = self.compute_graph_rpca()
            super().fit(D=self.D)

            error = (
                np.linalg.norm(
                    self.initial_D[indices_x, indices_y] - self.X[indices_x, indices_y],
                    1,
                )
                / nb_missing
            )
            if error == error:
                errors.append(error)

        if len(errors) == 0:
            print("Warning: not converged - return default 10^10")
            return 10**10

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

        self.gamma1 = res.x[0]
        self.gamma2 = res.x[1]
        super().fit(D=self.D)

        return None
