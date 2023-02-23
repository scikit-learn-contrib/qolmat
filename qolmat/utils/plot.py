"""
Useful drawing functions
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

sns.set_context("paper")
sns.set_style("whitegrid", {"axes.grid": False})
sns.set_theme(style="ticks")

plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 1
plt.rcParams["grid.color"] = "#cccccc"

tab10 = plt.get_cmap("tab10")


def plot_matrices(list_matrices: List[np.ndarray], title: Optional[str] = None) -> None:
    """Plot RPCA matrices

    Parameters
    ----------
    list_matrices : List[np.ndarray]
        List containing, in the right order, the observations matrix, the low-rank matrix and the
        sparse matrix
    title : Optional[str], optional
        if present, title of the saved figure, by default None
    """

    suptitles = ["Observations", "Low-rank", "Sparse"]

    fig, ax = plt.subplots(1, 3, figsize=(10, 3))

    for i, (m, t) in enumerate(zip(list_matrices, suptitles)):
        if i != 2:
            im = ax[i].imshow(
                m,
                aspect="auto",
                # vmin=min(np.min(list_matrices[0]), np.min(list_matrices[1])),
                # vmax=max(np.max(list_matrices[0]), np.max(list_matrices[1])),
            )
        else:
            m = ax[i].imshow(m, aspect="auto")
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")
        ax[i].set_title(t, fontsize=16)

    plt.tight_layout()
    if title:
        plt.savefig(f"../figures/{title}.png", transparent=True)

    plt.show()


def plot_signal(
    list_signals: List[List],
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    dates: Optional[List] = None,
) -> None:
    """Plot RPCA results for time series

    Parameters
    ----------
    list_signals : List[List]
        List containing, in the right order, the  observed time series, the cleaned signal and
        the anomalies
    title : Optional[str], optional
        if present, title of the saved figure, by default None
    ylabel : Optional[str], optional
        ylabel, by default None
    dates : Optional[List], optional
        dates of the time series (xlabel), by default None
    """

    suptitles = ["Observations", "Cleaned", "Anomalies"]
    colors = ["black", "darkblue", "crimson"]
    fontsize = 15

    if dates is None:
        dates = list(range(len(list_signals[0])))

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(15, 6))
    for i, (r, c, t) in enumerate(zip(list_signals, colors, suptitles)):
        ax[i].plot(dates, r, color=c)
        ax[i].set_title(t, fontsize=fontsize)
        if ylabel:
            ax[i].set_ylabel(ylabel, fontsize=fontsize)

    fig.align_ylabels()
    plt.tight_layout()
    if title:
        plt.savefig(f"../figures/{title}.png", transparent=True)
    plt.show()


def plot_images(
    M: np.ndarray,
    A: np.ndarray,
    E: np.ndarray,
    index_array: List[int],
    dims: Tuple[int, int],
    filename: Optional[str] = None,
) -> None:
    """Plot multiple images in 3 columns for original, background and "foreground"

    Parameters
    ----------
    M : np.ndarray
        orginal array
    A : np.ndarray
        background array
    E : np.ndarray
        foreground/moving object array
    index_array : List[int]
        indices of the plotted frames
    dims : Tuple[int, int]
        dimensions of the reduction
    filename : Optional[str], optional
        filename for saving figure, by default None
    """

    f = plt.figure(figsize=(15, 10))
    r = len(index_array)

    for k, i in enumerate(index_array):
        for j, mat in enumerate([M, A, E]):
            sp = f.add_subplot(r, 3, 3 * k + j + 1)
            sp.set_xticks([])
            sp.set_yticks([])
            pixels = mat[:, i]
            if isinstance(pixels, scipy.sparse.csr_matrix):
                pixels = pixels.todense()
            sp.imshow(np.reshape(pixels, dims), cmap="gray")

            if j == 0:
                sp.set_ylabel(f"Frame {i}", fontsize=25)

            if k == 0:
                if j == 0:
                    sp.set_title("Original", fontsize=25)
                elif j == 1:
                    sp.set_title("Background", fontsize=25)
                else:
                    sp.set_title("Moving objects", fontsize=25)

    if filename:
        plt.savefig(f"../figures/{filename}.png", transparent=True)

    plt.tight_layout()
    plt.show()


def make_ellipses(
    X: np.ndarray,
    ax: any,
    color: Union[str, Tuple[float, float, float]],
):
    """Draw ellipses on a figure

    Parameters
    ----------
    X : np.ndarray
        array for the ellipse
    ax : matplotlib.axes._subplots.AxesSubplot
        matplotlib ax handles
    color : Union[str, Tuple[float, float, float]]
        ellipse's color
    """
    covariances = X.cov()  # gmm.covariances_[0] # [n][:2, :2]
    v, w = np.linalg.eigh(covariances)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    center = X.mean()  # .means_[0]
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    ell = mpl.patches.Ellipse(center, v[0], v[1], 180 + angle, color=color)
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.5)
    ax.add_artist(ell)
    ax.set_aspect("equal", "datalim")


def compare_covariances(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    col_x: str,
    col_y: str,
    ax: any,
    color=None,
):
    """
    Covariance plot: scatter plot with ellipses

    Parameters
    ----------
    df1 : pd.DataFrame
        dataframe with raw data
    df2 : pd.DataFrame
        dataframe with imputations
    col_x : str
        variable x, column's name of dataframe df1 to compare with
    col_y : str
        variable y, column's name of dataframe df2 to compare with
    ax : matplotlib.axes._subplots.AxesSubplot
        matplotlib ax handles
    """
    if color is None:
        color = tab10(0)
    ax.scatter(df2[col_x], df2[col_y], marker=".", color=color)
    ax.scatter(df1[col_x], df1[col_y], marker=".", color="black")
    make_ellipses(df1[[col_x, col_y]], ax, "black")
    make_ellipses(df2[[col_x, col_y]], ax, color)
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)


def display_bar_table(data: pd.DataFrame, ylabel: Optional[str] = "", path: Optional[str] = None):
    """Displaying barplot and table with the associated data side by side

    Parameters
    ----------
    data : pd.DataFrame
        dataframe containing the data to display. Indices = groups
    ylabel : Optional[str], optional
        ylabel of the plot, by default ""
    path : Optional[str], optional
        entire path for saving, by default None
    """
    colors = plt.cm.YlGnBu(np.linspace(0.2, 0.75, len(data)))

    data.T.plot(x=data.T.index.name, kind="bar", stacked=False, color=colors)
    sns.despine()

    plt.table(
        cellText=np.around(data.values, 4),
        rowLabels=data.index,
        rowColours=colors,
        colLabels=data.columns,
        bbox=[1.5, 0, 1.6, 1],
    )

    plt.xticks(fontsize=14)
    plt.ylabel(f"{ylabel}", fontsize=14)
    sns.despine()

    if path:
        plt.savefig(f"{path}.png", transparent=True)
    plt.show()
