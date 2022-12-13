from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb


def compute_distri(sample: np.ndarray) -> np.ndarray:
    """Compute the distribution from an array of float

    Parameters
    ----------
    sample : np.ndarray
        array of floats

    Returns
    -------
    np.ndarray
        distribution
    """
    return np.unique(sample, return_counts=True)[1] / len(sample)


def KL(P: pd.Series, Q: pd.Series) -> float:
    """
    Compute the Kullback-Leibler divergence between distributions P and Q
    Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0.

    Parameters
    ----------
    P : pd.Series
        "true" distribution
    Q : pd.Series
        sugggesetd distribution

    Return
    ------
    float
        KL(P,Q)
    """
    epsilon = 0.00001

    P = P.copy() + epsilon
    Q = Q.copy() + epsilon

    return np.sum(P * np.log(P / Q))


def display_bar_table(
    data: pd.DataFrame, ylabel: Optional[str] = "", path: Optional[str] = None
):
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
    sb.despine()

    plt.table(
        cellText=np.around(data.values, 4),
        rowLabels=data.index,
        rowColours=colors,
        colLabels=data.columns,
        bbox=[1.5, 0, 1.6, 1],
    )

    plt.xticks(fontsize=14)
    plt.ylabel(f"{ylabel}", fontsize=14)
    sb.despine()

    if path:
        plt.savefig(f"{path}.png", transparent=True)
    plt.show()
