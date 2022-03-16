"""
Useful drawing functions
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Type

import numpy as np
import scipy
import matplotlib.pyplot as plt
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 1
plt.rcParams["grid.color"] = "#cccccc"
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.graph_objects as go
import plotly.subplots as sp

def plot_matrices(
    list_matrices: List[np.ndarray],
    title: Optional[str]=None
) -> None:
    """Plot RPCA matrices

    Parameters
    ----------
    list_matrices : List[np.ndarray]
        List containing, in the right order, the observations matrix, the low-rank matrix and the sparse matrix
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
                #vmin=min(np.min(list_matrices[0]), np.min(list_matrices[1])),
                #vmax=max(np.max(list_matrices[0]), np.max(list_matrices[1])),
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
    title: Optional[str]=None, 
    ylabel: Optional[str]=None,
    dates: Optional[List]=None,
    style: Optional[str]="plotly",
) -> None:
    """Plot RPCA results for time series

    Parameters
    ----------
    list_signals : List[List]
        List containing, in the right order, the  observed time series, the cleaned signal and the anomalies
    title : Optional[str], optional
        if present, title of the saved figure, by default None
    ylabel : Optional[str], optional
        ylabel, by default None
    dates : Optional[List], optional
        dates of the time series (xlabel), by default None
    style : Optional[str], optional
        "matplotlib" or "plotly", by default "plotly"
    """

    suptitles = ["Observations", "Cleaned", "Anomalies"]
    colors = ["black", "darkblue", "crimson"]
    fontsize = 15

    if dates is None:
        dates = list(range(len(list_signals[0])))

    if style == "matplotlib":
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(15,6))
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
            
    if style == "plotly":
        fig = sp.make_subplots(rows=3, cols=1)
        for i, (r, c, t) in enumerate(zip(list_signals, colors, suptitles)):
            fig.add_trace(
                    go.Scatter(x=dates, y=r, line=dict(color=c), name=t),
                    row=i + 1,
                    col=1,
                )
        if title:
            plt.savefig(f"../figures/{title}.png", transparent=True)
        fig.show()

def plot_images(
    M: np.ndarray, 
    A: np.ndarray, 
    E: np.ndarray, 
    index_array: List[int], 
    dims: Tuple[int, int], 
    filename: Optional[str]=None
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
            pixels = mat[:,i]
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