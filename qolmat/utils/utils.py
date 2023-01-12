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


def progress_bar(iteration, total, prefix="", suffix="", decimals=1, length=100, fill="█"):
    """Call in a loop to create terminal progress bar

    Parameters
    ----------
    iteration : int
        current iteration
    total : int
        total iterations
    prefix : str, optional
        prefix string, by default ""
    suffix : str, optional
        suffix string, by default ""
    decimals : int, optional
        positive number of decimals in percent complete, by default 1
    length : int, optional
        character length of bar, by default 100
    fill : str, optional
        bar fill character, by default "█"
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="\r")
    # Print New Line on Complete
    if iteration == total:
        print()
