"""
=======================================================================================
Tutorial for comparison between mean and median imputations with uniform hole geneation
=======================================================================================

In this tutorial, we show how to use the Qolmat comparator
(:class:`~qolmat.benchmark.comparator`) to choose
the best imputation between imputation by the mean
(:class:`~qolmat.imputations.imputers.ImputerMean`) or the median
(:class:`~qolmat.imputations.imputers.ImputerMedian`).
The dataset used is the the numerical `superconduct` dataset and
contains information on 21263 superconductors.
We generate holes uniformly at random via
:class:`~qolmat.benchmark.missing_patterns.UniformHoleGenerator`
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd

from qolmat.benchmark import comparator, missing_patterns
from qolmat.imputations import imputers
from qolmat.utils import data, plot


#################################################################
# 1. Data
# ---------------------------------------------------------------
# The data contains information on 21263 superconductors.
# Originally, the first 81 columns contain extracted features and
# the 82nd column contains the critical temperature which is used as the
# target variable. The original data from which the features were extracted
# comes from http://supercon.nims.go.jp/index_en.html, which is public.
# The data does not contain missing values; so for the purpose of this notebook,
# we corrupt the data, with the `~qolmat.utils.data.add_holes` function.
# In this way, each column has missing values.

csv_url = (
    "https://huggingface.co/datasets/polinaeterna/"
    "tabular-benchmark/resolve/main/reg_num/superconduct.csv"
)
df_data = pd.read_csv(csv_url, index_col=0)
df = data.add_holes(df_data, ratio_masked=0.2, mean_size=120)

#################################################################
# The dataset contains 82 columns. For simplicity,
# we only consider some.

columns = [
    "criticaltemp",
    "mean_atomic_mass",
    "mean_Density",
    "mean_FusionHeat",
    "mean_ThermalConductivity",
    "mean_Valence",
]
df = df[columns]
cols_to_impute = df.columns

#################################################################
# Let's take a look at variables to impute.

fig, axs = plt.subplots(len(cols_to_impute), 1, figsize=(13, 3 * len(cols_to_impute)))
for ax, col in zip(axs.flatten(), cols_to_impute):
    ax.plot(df[col], "o", ms=2)
    ax.set_ylabel(col.replace("_", " "))
plt.show()

#################################################################
# 2. Imputation
# ---------------------------------------------------------------
# This part is devoted to the imputation methods.
# In this tutorial, we only focus on mean and median imputation.
# In order to use the comparator, we have to define a dictionary of imputers,
# a way to generate holes (additional missing values on which the
# imputers will be evaluated) and a list of metrics.

imputer_mean = imputers.ImputerMean()
imputer_median = imputers.ImputerMedian()
dict_imputers = {"mean": imputer_mean, "median": imputer_median}

generator_holes = missing_patterns.UniformHoleGenerator(
    n_splits=2, subset=cols_to_impute, ratio_masked=0.1
)

metrics = ["mae", "wmape", "KL_columnwise"]

#################################################################
# Concretely, the comparator takes as input a dataframe to impute,
# a proportion of nan to create, a dictionary of imputers
# (those previously mentioned),
# a list with the columns names to impute,
# a generator of holes specifying the type of holes to create.

comparison = comparator.Comparator(
    dict_imputers,
    cols_to_impute,
    generator_holes=generator_holes,
    metrics=metrics,
    max_evals=5,
)

#################################################################
# On the basis of the results, we can see that imputation by the median provides
# lower reconstruction errors than those obtained by imputation by the mean,
# whatever the metric and for all the columns to be imputed.

results = comparison.compare(df)
results.style.highlight_min(color="lightgreen", axis=1)

#################################################################
# Let's visualize this dataframe.

n_metrics = len(metrics)
fig = plt.figure(figsize=(14, 3 * n_metrics))
for i, metric in enumerate(metrics):
    fig.add_subplot(n_metrics, 1, i + 1)
    plot.multibar(results.loc[metric], decimals=2)
    plt.ylabel(metric)
plt.show()
