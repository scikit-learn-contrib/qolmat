from __future__ import annotations
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

import utils


class rpca_PCP:
    """Decompose a matrix into low rank and sparse components.
    Computes the RPCA decomposition using Alternating Lagrangian Multipliers.
    Returns L,S the low rank and sparse components respectively
    """

    def __init__(
        self,
        signal: Optional[List[float]] = [],
        period: Optional[int] = 0,
        M: Optional[np.ndarray] = None,
        mu: Optional[float] = None,
        lam: Optional[float] = None,
        maxIter: Optional[int] = int(1e4),
        tol: Optional[float] = 1e-6,
        verbose: Optional[str] = False,
    ) -> None:

        if (signal == []) and (M is None):
            raise Exception(
                "You have to provide either a time series (signal) or a matrix (M)"
            )

        self.signal = signal
        self.period = period
        self.M = M
        self.maxIter = maxIter
        self.tol = tol
        self.verbose = verbose
        self.mu = mu
        self.lam = lam

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

        self.M = np.array(self.signal).reshape(-1, self.period)
        self.ret = ret

    def prepare_data(self) -> None:
        if (self.M is None) and (self.period is None):
            self.get_period()
        if self.M is None:
            self.signal_to_matrix()

        self.initial_M = self.M.copy()

    def compute_rpca_PCP(self) -> Tuple[np.ndarray, np.ndarray]:

        if np.isnan(np.sum(self.M)):
            self.proj_M = utils.impute_nans(self.M, method="median")
        else:
            self.proj_M = self.M

        if self.mu is None:
            self.mu = np.prod(self.proj_M.shape) / (
                4.0 * utils.l1_norm(self.proj_M)
            )

        if self.lam is None:
            self.lam = 1 / np.sqrt(np.max(self.proj_M.shape))

        M_norm = np.linalg.norm(self.proj_M, "fro")

        n, m = self.M.shape
        S = np.zeros((n, m))
        Y = np.zeros((n, m))

        errors = []
        for iteration in range(self.maxIter):
            L = utils.svd_thresholding(
                self.proj_M - S + Y / self.mu, 1 / self.mu
            )
            S = utils.soft_thresholding(
                self.proj_M - L + Y / self.mu, self.lam / self.mu
            )
            Y += self.mu * (self.proj_M - L - S)

            errors.append(np.linalg.norm(self.proj_M - L - S, "fro") / M_norm)
            if errors[-1] <= self.tol:
                if self.verbose:
                    print(f"Converged in {iteration} iterations")
                break

        self.L = L
        self.S = S
        self.errors = errors

        return L, S

    def plot_matrices(self) -> None:

        matrices = [self.initial_M, self.L, self.S]
        titles = ["Observations", "Low-rank", "Sparse"]

        fig, ax = plt.subplots(1, 3, figsize=(10, 3))

        for i, (m, t) in enumerate(zip(matrices, titles)):
            if i != 20:
                im = ax[i].imshow(
                    m,
                    aspect="auto",
                    vmin=min(np.min(self.proj_M), np.min(self.L)),
                    vmax=max(np.max(self.proj_M), np.max(self.L)),
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
            self.L.flatten().tolist(),
            self.S.flatten().tolist(),
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
