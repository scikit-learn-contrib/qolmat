from __future__ import annotations
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

import skopt

import utils


class improve_rpca:
    def __init__(
        self,
        signal: Optional[List[float]] = [],
        period: Optional[int] = None,
        D: Optional[np.ndarray] = None,
        rank: Optional[int] = None,
        lam: Optional[float] = None,
        list_periods: Optional[List[int]] = [],
        list_etas: Optional[List[float]] = [],
        maxIter: Optional[int] = int(1e4),
        tol: Optional[float] = 1e-6,
        verbose: Optional[str] = False,
    ) -> None:

        if (signal == []) and (D is None):
            raise Exception(
                "You must provide either a time series (signal) or a matrix (D)"
            )
            
        if lam is None:
            raise Exception(
                "You must provide value for lambda or use the improve_rpca_hyperparams class"
            )
            
        if len(list_periods) != len(list_etas):
            raise Exception(
                "list_periods and list_etas must have the same length"
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

    def get_period(self) -> None:
        """
        Retrieve the "period" of a series based on the ACF
        """
        ss = pd.Series(self.signal)
        val = []
        for i in range(100):
            val.append(ss.autocorr(lag=i))

        ind_sort = sorted(range(len(val)), key=lambda k: val[k])

        self.period = ind_sort[::-1][1]

    def signal_to_matrix(self) -> None:
        """Shape a time series into a matrix"""

        modulo = len(self.signal) % self.period
        ret = (self.period - modulo) % self.period
        self.signal += [np.nan] * ret

        self.D = np.array(self.signal).reshape(-1, self.period)
        self.ret = ret

    def projection_observation(self) -> None:
        self.omega = 1 - (self.D != self.D)

        if np.isnan(np.sum(self.D)):
            self.proj_D = utils.impute_nans(self.D, method="median")
        else:
            self.proj_D = self.D

    def prepare_data(self) -> None:
        if (self.D is None) and (self.period is None):
            self.get_period()
        if self.D is None:
            self.signal_to_matrix()

        self.initial_D = self.D.copy()
        self.projection_observation()

    def k_choice(self, th: float = 0.95) -> None:
        """
        Compute de dimension k for decomposition

        Args:
            D (np.Array): observation matrix
            th (float): threshold

        Returns:
            int: dimension for the decomposition, via threshold
        """
        _, s, _ = np.linalg.svd(self.proj_D, full_matrices=True)
        nuclear = np.sum(s)
        cum_sum = np.cumsum([i / nuclear for i in s])
        k = np.argwhere(cum_sum > th)[0][0] + 1
        self.rank = k

    @staticmethod
    def get_Hmatrix(T: int, dimension: int) -> np.ndarray:
        """
        Create a matrix H to take into account temporal correlation via HX

        Args:
            T (int): period
            dimension (int): second dimension of H = first dimension of X

        Returns:
            H (np.ndarray): temporal matrix
        """

        H = np.eye(dimension - T, dimension)
        H[: dimension - T, T:] = H[: dimension - T, T:] - np.eye(
            dimension - T, dimension - T
        )
        return H

    def compute_improve_rpca(self) -> Tuple[np.ndarray, np.ndarray]:

        self.projection_observation()

        if self.rank is None:
            self.k_choice()

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
            H[str(i)] = improve_rpca.get_Hmatrix(self.list_periods[i], m)

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

        return X, A

    def plot_matrices(self) -> None:

        matrices = [self.initial_D, self.X, self.A]
        titles = ["Observations", "Low-rank", "Sparse"]

        fig, ax = plt.subplots(1, 3, figsize=(10, 3))

        for i, (m, t) in enumerate(zip(matrices, titles)):
            if i != 20:
                im = ax[i].imshow(
                    m,
                    aspect="auto",
                    vmin=min(np.min(self.proj_D), np.min(self.X)),
                    vmax=max(np.max(self.proj_D), np.max(self.X)),
                )
            else:
                m = ax[i].imshow(m, aspect="auto")
            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")
            ax[i].set_title(t, fontsize=16)

        plt.tight_layout()
        plt.show()

    def plot_signal(self) -> None:

        x_index = list(range(len(self.signal) - self.ret))
        res = [
            self.signal,
            self.X.flatten().tolist(),
            self.A.flatten().tolist(),
        ]
        titles = ["original signal", "clean signal", "anomalies"]
        colors = ["black", "darkblue", "crimson"]

        fig = sp.make_subplots(rows=3, cols=1)

        for i, (r, c, t) in enumerate(zip(res, colors, titles)):
            if self.ret == 0:
                fig.add_trace(
                    go.Scatter(x=x_index, y=r, line=dict(color=c), name=t),
                    row=i + 1,
                    col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=x_index,
                        y=r[: -self.ret],
                        line=dict(color=c),
                        name=t,
                    ),
                    row=i + 1,
                    col=1,
                )

        fig.show()


class improve_rpca_hyperparams(improve_rpca):
    # def __init__(self, params):
    #     lam = params.pop("lam")
    #     etas = params.pop("etas")
    #     improve_rpca.__init__(params)
    #     self.add_hyperparams(lam, etas)
        
    def add_hyperparams(
        self,
        hyperparams_lam: Optional[List[float]] = [],
        hyperparams_etas: Optional[List[List[float]]] = [[]],
    ) -> None:

        self.search_space = []
        if len(hyperparams_lam) > 0:
            self.search_space.append(
                skopt.space.Real(
                    low=hyperparams_lam[0], high=hyperparams_lam[1], name="lam"
                )
            )
        if len(hyperparams_etas) > 0:  # TO DO: more cases
            for i in range(len(hyperparams_etas)):
                self.search_space.append(
                    skopt.space.Real(
                        low=hyperparams_etas[i][0],
                        high=hyperparams_etas[i][1],
                        name=f"eta_{i}",
                    )
                )

    def objective(self, args):
        self.lam = args[0]
        self.list_etas = [args[i + 1] for i in range(len(self.list_periods))]

        n1, n2 = self.initial_D.shape
        nb_missing = int(n1 * n2 * 0.05)

        errors = []
        for iter_obj in range(2):
            indices_x = np.random.choice(n1, nb_missing)
            indices_y = np.random.choice(n2, nb_missing)
            data_missing = self.initial_D.copy().astype("float")
            data_missing[indices_x, indices_y] = np.nan

            self.D = data_missing

            X, _ = self.compute_improve_rpca()

            error = (
                np.linalg.norm(
                    self.initial_D[indices_x, indices_y]
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

    def compute_improve_rpca_hyperparams(self) -> Tuple[np.ndarray, np.ndarray]:
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
        X, A = self.compute_improve_rpca()
