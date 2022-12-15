from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
