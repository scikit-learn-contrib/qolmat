"""
Useful drawing functions
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
            im = ax[i].imshow(m, aspect="auto")
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
    x: NDArray,
    y: NDArray,
    ax: mpl.axes.Axes,
    n_std: float = 2,
    color: Union[str, Any, Tuple[float, float, float]] = "None",
):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    color : Optional[str]
        facecolor

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson) * 2.5
    ell_radius_y = np.sqrt(1 - pearson) * 2.5
    ell = mpl.patches.Ellipse((0, 0), width=ell_radius_x, height=ell_radius_y, facecolor=color)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = (
        mpl.transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    )
    ell.set_transform(transf + ax.transData)
    ax.add_patch(ell)
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.4)
    ax.set_aspect("equal", "datalim")


def compare_covariances(
    df_1: pd.DataFrame,
    df_2: pd.DataFrame,
    col_x: str,
    col_y: str,
    ax: mpl.axes.Axes,
    label: str = "",
    color: Union[None, str, Tuple[float, float, float], Tuple[float, float, float, float]] = None,
):
    """
    Covariance plot: scatter plot with ellipses

    Parameters
    ----------
    df_1 : pd.DataFrame
        dataframe with raw data
    df_2 : pd.DataFrame
        dataframe with imputations
    col_x : str
        variable x, column's name of dataframe df1 to compare with
    col_y : str
        variable y, column's name of dataframe df2 to compare with
    ax : matplotlib.axes._subplots.AxesSubplot
        matplotlib ax handles
    """
    df1 = df_1.dropna()
    df2 = df_2.dropna()
    if color is None:
        color = tab10(0)
    ax.scatter(df2[col_x], df2[col_y], marker=".", color=color, s=2, alpha=0.7, label="imputed")
    ax.scatter(df1[col_x], df1[col_y], marker=".", color="black", s=2, alpha=0.7, label="original")
    make_ellipses(df1[col_x], df1[col_y], ax, color="black")
    make_ellipses(df2[col_x], df2[col_y], ax, color=color)
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)


def multibar(
    df: pd.DataFrame,
    ax: Optional[mpl.axes.Axes] = None,
    orientation: str = "vertical",
    colors: Any = None,
    decimals: float = 0,
):
    """Create a multi-bar graph to represent the values of the different dataframe columns.

    Parameters
    ----------
    df : pd.DataFrame
        contain the dataframe
    ax : Any, optional
        matplotlib ax handles, by default None
    orientation : str, optional
        orientation of plot, by default "vertical"
    colors : str, optional
        color in multibar plot, by default None
    decimals : float, optional
        the decimals numbers, by default 0
    """

    if ax is None:
        ax = plt.gca()
        if colors is None:
            colors = tab10
    x = np.arange(len(df))  # the label locations
    n_columns = len(df.columns)
    width_tot = 0.8
    width_col = width_tot / n_columns  # the width of the bars

    for i_column, column in enumerate(df.columns):
        color_col = colors(i_column % 10)
        dx = width_tot * (-0.5 + float(i_column) / n_columns)
        if orientation == "horizontal":
            rect = ax.barh(
                x + dx,
                df[column],
                width_col,
                label=column,
                align="edge",
                color=color_col,
            )
            plt.yticks(x, df.index)
        else:
            rect = ax.bar(
                x + dx,
                df[column],
                width_col,
                label=column,
                align="edge",
                color=color_col,
            )
            plt.xticks(x, df.index)
        ax.bar_label(rect, padding=3, fmt=f"%.{decimals}f")

    plt.legend(loc=(1, 0))


def plot_imputations(df: pd.DataFrame, dict_df_imputed: Dict[str, pd.DataFrame]):
    """Plot original and imputed dataframes for each imputers

    Parameters
    ----------
    df : pd.DataFrame
        original dataframe
    dict_df_imputed : Dict[str, pd.DataFrame]
        dictionnary of imputed dataframe for each imputers
    """
    n_columns = len(df.columns)
    n_imputers = len(dict_df_imputed)

    fig = plt.figure(figsize=(8 * n_columns, 6 * n_imputers))
    i_plot = 1
    for name_imputer, df_imputed in dict_df_imputed.items():
        for col in df:
            ax = fig.add_subplot(n_imputers, n_columns, i_plot)
            values_orig = df[col]

            plt.plot(values_orig, ".", color="black", label="original")
            values_imp = df_imputed[col].copy()
            values_imp[values_orig.notna()] = np.nan
            plt.plot(values_imp, ".", color=tab10(0), label=name_imputer, alpha=1)
            plt.ylabel(col, fontsize=16)
            if i_plot % n_columns == 0:
                plt.legend(loc=[1, 0], fontsize=18)
            loc = plticker.MultipleLocator(base=2 * 365)
            ax.xaxis.set_major_locator(loc)
            ax.tick_params(axis="both", which="major", labelsize=17)
            i_plot += 1
    plt.show()
