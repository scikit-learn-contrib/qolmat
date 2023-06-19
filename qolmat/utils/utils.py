from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    plt.table(
        cellText=np.around(data.values, 4),
        rowLabels=data.index,
        rowColours=colors,
        colLabels=data.columns,
        bbox=[1.5, 0, 1.6, 1],
    )

    plt.xticks(fontsize=14)
    plt.ylabel(f"{ylabel}", fontsize=14)

    if path:
        plt.savefig(f"{path}.png", transparent=True)
    plt.show()


def progress_bar(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 100,
    fill: str = "█",
):
    """Call in a loop to create terminal progress bar

    Parameters
    ----------
    iteration : int
        current iteration
    total : int
        total iterations
    prefix : str
        prefix string, by default ""
    suffix : str
        suffix string, by default ""
    decimals : int
        positive number of decimals in percent complete, by default 1
    length : int
        character length of bar, by default 100
    fill : str
        bar fill character, by default "█"
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="\r")
    if iteration == total:
        print()


def acf(values: pd.Series, lag_max: int = 30) -> pd.Series:
    """Correlation series of dataseries

    Parameters
    ----------
    values : pd.Series
        dataseries
    lag_max : int, optional
        the maximum lag, by default 30

    Returns
    -------
    pd.Series
        correlation series of value
    """
    acf = pd.Series(0, index=range(lag_max))
    for lag in range(lag_max):
        acf[lag] = values.corr(values.shift(lag))
    return acf
