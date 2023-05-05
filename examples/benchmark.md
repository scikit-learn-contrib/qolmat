---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: env_qolmat_dev
    language: python
    name: env_qolmat_dev
---

**This notebook aims to present the Qolmat repo through an example of a multivariate time series.
In Qolmat, a few data imputation methods are implemented as well as a way to evaluate their performance.**


First, import some useful librairies

```python
import warnings
# warnings.filterwarnings('error')
```

```python
%reload_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
import scipy
np.random.seed(1234)
import pprint
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as plticker

tab10 = plt.get_cmap("tab10")
plt.rcParams.update({'font.size': 18})

from typing import Optional

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor


import sys
from qolmat.benchmark import comparator, missing_patterns
from qolmat.benchmark.utils import kl_divergence
from qolmat.imputations import imputers
from qolmat.utils import data, utils, plot
# from qolmat.drawing import display_bar_table

```

### **I. Load data**


The dataset `Beijing` is the Beijing Multi-Site Air-Quality Data Set. It consists in hourly air pollutants data from 12 chinese nationally-controlled air-quality monitoring sites and is available at https://archive.ics.uci.edu/ml/machine-learning-databases/00501/.
This dataset only contains numerical vairables.

```python
df_data = data.get_data_corrupted("Beijing", ratio_masked=.2, mean_size=120)

# cols_to_impute = ["TEMP", "PRES", "DEWP", "NO2", "CO", "O3", "WSPM"]
# cols_to_impute = df_data.columns[df_data.isna().any()]
cols_to_impute = ["TEMP", "PRES"]

```

The dataset `Artificial` is designed to have a sum of a periodical signal, a white noise and some outliers.

```python
df_data
```

```python
# df_data = data.get_data_corrupted("Artificial", ratio_masked=.2, mean_size=10)
# cols_to_impute = ["signal"]
```

Let's take a look at variables to impute. We only consider a station, Aotizhongxin.
Time series display seasonalities (roughly 12 months).

```python
n_stations = len(df_data.groupby("station").size())
n_cols = len(cols_to_impute)
```

```python
fig = plt.figure(figsize=(10 * n_stations, 3 * n_cols))
for i_station, (station, df) in enumerate(df_data.groupby("station")):
    df_station = df_data.loc[station]
    for i_col, col in enumerate(cols_to_impute):
        fig.add_subplot(n_cols, n_stations, i_col * n_stations + i_station + 1)
        plt.plot(df_station[col], '.', label=station)
        # break
        plt.ylabel(col)
        plt.xticks(rotation=15)
        if i_col == 0:
            plt.title(station)
        if i_col != n_cols - 1:
            plt.xticks([], [])
plt.show()
```

### **II. Imputation methods**


This part is devoted to the imputation methods. The idea is to try different algorithms and compare them.

<u>**Methods**</u>:
All presented methods are group-wise: here each station is imputed independently. For example ImputerMean computes the mean of each variable in each station and uses the result for imputation; ImputerInterpolation interpolates termporal signals corresponding to each variable on each station.

<u>**Hyperparameters' search**</u>:
Some methods require hyperparameters. The user can directly specify them, or rather determine them through an optimization step using the `search_params` dictionary. The keys are the imputation method's name and the values are a dictionary specifying the minimum, maximum or list of categories and type of values (Integer, Real, Category or a dictionary indexed by the variable names) to search.
In pratice, we rely on a cross validation to find the best hyperparams values minimizing an error reconstruction.

```python
imputer_mean = imputers.ImputerMean(groups=["station"])
imputer_median = imputers.ImputerMedian(groups=["station"])
imputer_mode = imputers.ImputerMode(groups=["station"])
imputer_locf = imputers.ImputerLOCF(groups=["station"])
imputer_nocb = imputers.ImputerNOCB(groups=["station"])
imputer_interpol = imputers.ImputerInterpolation(groups=["station"], method="linear")
imputer_spline = imputers.ImputerInterpolation(groups=["station"], method="spline", order=2)
imputer_shuffle = imputers.ImputerShuffle(groups=["station"])
imputer_residuals = imputers.ImputerResiduals(groups=["station"], period=7, model_tsa="additive", extrapolate_trend="freq", method_interpolation="linear")

imputer_rpca = imputers.ImputerRPCA(groups=["station"], columnwise=True, period=365, max_iter=200, tau=2, lam=.3)
imputer_rpca_opti = imputers.ImputerRPCA(groups=["station"], columnwise=True, period=365, max_iter=100)

imputer_ou = imputers.ImputerEM(groups=["station"], method="multinormal", strategy="ou", max_iter_em=34, n_iter_ou=15, dt=1e-3)
imputer_tsou = imputers.ImputerEM(groups=["station"], method="VAR1", strategy="ou", max_iter_em=34, n_iter_ou=15, dt=1e-3)
imputer_tsmle = imputers.ImputerEM(groups=["station"], method="VAR1", strategy="mle", max_iter_em=34, n_iter_ou=15, dt=1e-3)


imputer_knn = imputers.ImputerKNN(groups=["station"], k=10)
imputer_mice = imputers.ImputerMICE(groups=["station"], estimator=LinearRegression(), sample_posterior=False, max_iter=100, missing_values=np.nan)
imputer_regressor = imputers.ImputerRegressor(groups=["station"], estimator=LinearRegression())

dict_imputers = {
    "mean": imputer_mean,
    "median": imputer_median,
    # "mode": imputer_mode,
    "interpolation": imputer_interpol,
    # "spline": imputer_spline,
    # "shuffle": imputer_shuffle,
    # "residuals": imputer_residuals,
    "OU": imputer_ou,
    "TSOU": imputer_tsou,
    "TSMLE": imputer_tsmle,
    # "RPCA": imputer_rpca,
    # "RPCA_opti": imputer_rpca_opti,
    # "locf": imputer_locf,
    # "nocb": imputer_nocb,
    # "knn": imputer_knn,
    "iterative": imputer_regressor,
    "regressor": imputer_mice,
}
n_imputers = len(dict_imputers)

search_params = {
    "RPCA_opti": {
        "tau": {"min": .5, "max": 5, "type":"Real"},
        "lam": {"min": .1, "max": 1, "type":"Real"},
    }
}

ratio_masked = 0.1
```

In order to compare the methods, we $i)$ artificially create missing data (for missing data mechanisms, see the docs); $ii)$ then impute it using the different methods chosen and $iii)$ calculate the reconstruction error. These three steps are repeated a number of times equal to `n_splits`. For each method, we calculate the average error and compare the final errors.

<p align="center">
    <img src="../../docs/images/comparator.png"  width=50% height=50%>
</p>



Concretely, the comparator takes as input a dataframe to impute, a proportion of nan to create, a dictionary of imputers (those previously mentioned), a list with the columns names to impute, a generator of holes specifying the type of holes to create and the search dictionary search_params for hyperparameter optimization.

Note these metrics compute reconstruction errors; it tells nothing about the distances between the "true" and "imputed" distributions.

```python
generator_holes = missing_patterns.EmpiricalHoleGenerator(n_splits=2, groups=["station"], ratio_masked=ratio_masked)

comparison = comparator.Comparator(
    dict_imputers,
    cols_to_impute,
    generator_holes = generator_holes,
    n_calls_opt=10,
    search_params=search_params,
)
results = comparison.compare(df_data)
results
```

```python
fig = plt.figure(figsize=(24, 8))
fig.add_subplot(2, 1, 1)
plot.multibar(results.loc["mae"], decimals=1)
plt.ylabel("mae")

fig.add_subplot(2, 1, 2)
plot.multibar(results.loc["energy"], decimals=1)
plt.ylabel("energy")
plt.show()
```

### **III. Comparison of methods**


We now run just one time each algorithm on the initial corrupted dataframe and compare the different performances through multiple analysis.

```python
df_plot = df_data[cols_to_impute]
```

```python
dfs_imputed = {name: imp.fit_transform(df_plot) for name, imp in dict_imputers.items()}
```

```python
station = df_plot.index.get_level_values("station")[0]
df_station = df_plot.loc[station]
dfs_imputed_station = {name: df_plot.loc[station] for name, df_plot in dfs_imputed.items()}
```

Let's look at the imputations.
When the data is missing at random, imputation is easier. Missing block are more challenging.

```python
for col in cols_to_impute:
    fig, ax = plt.subplots(figsize=(10, 3))
    values_orig = df_station[col]

    plt.plot(values_orig, ".", color='black', label="original")

    for ind, (name, model) in enumerate(list(dict_imputers.items())):
        values_imp = dfs_imputed_station[name][col].copy()
        values_imp[values_orig.notna()] = np.nan
        plt.plot(values_imp, ".", color=tab10(ind), label=name, alpha=1)
    plt.ylabel(col, fontsize=16)
    plt.legend(loc=[1, 0], fontsize=18)
    loc = plticker.MultipleLocator(base=2*365)
    ax.xaxis.set_major_locator(loc)
    ax.tick_params(axis='both', which='major', labelsize=17)
    plt.show()

```

```python
# plot.plot_imputations(df_station, dfs_imputed_station)

n_columns = len(df_plot.columns)
n_imputers = len(dict_imputers)

fig = plt.figure(figsize=(8 * n_imputers, 6 * n_columns))
i_plot = 1
for i_col, col in enumerate(df_plot):
    for name_imputer, df_imp in dfs_imputed_station.items():

        fig.add_subplot(n_columns, n_imputers, i_plot)
        values_orig = df_station[col]

        plt.plot(values_orig, ".", color='black', label="original")
        #plt.plot(df.iloc[870:1000][col], markers[0], color='k', linestyle='-' , ms=3)

        values_imp = df_imp[col].copy()
        values_imp[values_orig.notna()] = np.nan
        plt.plot(values_imp, ".", color=tab10(0), label=name_imputer, alpha=1)
        plt.ylabel(col, fontsize=16)
        if i_plot % n_columns == 1:
            plt.legend(loc=[1, 0], fontsize=18)
        plt.xticks(rotation=15)
        if i_col == 0:
            plt.title(name_imputer)
        if i_col != n_columns - 1:
            plt.xticks([], [])
        loc = plticker.MultipleLocator(base=2*365)
        ax.xaxis.set_major_locator(loc)
        ax.tick_params(axis='both', which='major')
        i_plot += 1
plt.savefig("figures/imputations_benchmark.png")
plt.show()

```

## (Optional) Neuronal Network Model


In this section, we present an MLP model of data imputation using Keras, which can be installed using a "pip install tensorflow".

```python
from qolmat.imputations import imputers_keras
import tensorflow as tf
```

We will train the MLP model on a time series, so we encode the temporality.

```python
time = np.concatenate([np.cos(2*np.pi*np.arange(60,366)/365), np.cos(2*np.pi*np.arange(1,366)/365), np.cos(2*np.pi*np.arange(1,366)/365), np.cos(2*np.pi*np.arange(1,367)/366),np.cos(2*np.pi*np.arange(1,60)/365)  ])
for i_station, (station, df) in enumerate(df_data.groupby("station")):
    df_data.loc[station, "Time"] = time
    # df_data.loc[station, "Time"] = np.array(df_data.loc[station].index.dayofyear)
```

To illustrate with an example the use of an MLP imputor, we will set up a network with 3 layers of neurons by adding it to the previous dictionary dict_imputer.

```python
estimator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)])
estimator.compile(optimizer='adam', loss='mse', metrics=['mae'])
dict_imputers["MLP"] = imputer_mlp = imputers_keras.ImputerRegressorKeras(groups=["station"], estimator=estimator, handler_nan = "column")
```

We can re-run the imputation model benchmark as before.

```python
generator_holes = missing_patterns.EmpiricalHoleGenerator(n_splits=2, subset = cols_to_impute, groups=["station"], ratio_masked=ratio_masked)

comparison = comparator.Comparator(
    dict_imputers,
    df_data.columns,
    generator_holes = generator_holes,
    n_calls_opt=10,
    search_params=search_params,
)
results = comparison.compare(df_data)
results
```

```python
fig = plt.figure(figsize=(24, 4))
plot.multibar(results.loc["mae"], decimals=1)
plt.ylabel("mae")
plt.show()
```

```python
df_plot = df_data
dfs_imputed = {name: imp.fit_transform(df_plot) for name, imp in dict_imputers.items()}
station = df_plot.index.get_level_values("station")[0]
df_station = df_plot.loc[station]
dfs_imputed_station = {name: df_plot.loc[station] for name, df_plot in dfs_imputed.items()}
```

```python
for col in cols_to_impute:
    fig, ax = plt.subplots(figsize=(10, 3))
    values_orig = df_station[col]

    plt.plot(values_orig, ".", color='black', label="original")

    for ind, (name, model) in enumerate(list(dict_imputers.items())):
        values_imp = dfs_imputed_station[name][col].copy()
        values_imp[values_orig.notna()] = np.nan
        plt.plot(values_imp, ".", color=tab10(ind), label=name, alpha=1)
    plt.ylabel(col, fontsize=16)
    plt.legend(loc=[1, 0], fontsize=18)
    loc = plticker.MultipleLocator(base=2*365)
    ax.xaxis.set_major_locator(loc)
    ax.tick_params(axis='both', which='major', labelsize=17)
    plt.show()
```

```python
n_columns = len(df_plot.columns)
n_imputers = len(dict_imputers)

fig = plt.figure(figsize=(8 * n_imputers, 6 * n_columns))
i_plot = 1
for i_col, col in enumerate(df_plot):
    for name_imputer, df_imp in dfs_imputed_station.items():

        fig.add_subplot(n_columns, n_imputers, i_plot)
        values_orig = df_station[col]

        plt.plot(values_orig, ".", color='black', label="original")

        values_imp = df_imp[col].copy()
        values_imp[values_orig.notna()] = np.nan
        plt.plot(values_imp, ".", color=tab10(0), label=name_imputer, alpha=1)
        plt.ylabel(col, fontsize=16)
        if i_plot % n_columns == 1:
            plt.legend(loc=[1, 0], fontsize=18)
        plt.xticks(rotation=15)
        if i_col == 0:
            plt.title(name_imputer)
        if i_col != n_columns - 1:
            plt.xticks([], [])
        loc = plticker.MultipleLocator(base=2*365)
        ax.xaxis.set_major_locator(loc)
        ax.tick_params(axis='both', which='major')
        i_plot += 1
plt.savefig("figures/imputations_benchmark.png")
plt.show()
```

## Covariance


We first check the covariance. We simply plot one variable versus one another.
One observes the methods provide similar visual resuls: it's difficult to compare them based on this criterion.

```python
fig = plt.figure(figsize=(6 * n_imputers, 6 * n_columns))
i_plot = 1
for i, col in enumerate(cols_to_impute[:-1]):
    for i_imputer, (name_imputer, df_imp) in enumerate(dfs_imputed.items()):
        ax = fig.add_subplot(n_columns, n_imputers, i_plot)
        plot.compare_covariances(df_plot, df_imp, col, cols_to_impute[i+1], ax, color=tab10(i_imputer), label=name_imputer)
        ax.set_title(f"imputation method: {name_imputer}", fontsize=20)
        i_plot += 1
        ax.legend()
plt.show()
```

## Auto-correlation


We are now interested in th eauto-correlation function (ACF). As seen before, time series display seaonal patterns.
[Autocorrelation](https://en.wikipedia.org/wiki/Autocorrelation) is the correlation of a signal with a delayed copy of itself as a function of delay. Informally, it is the similarity between observations of a random variable as a function of the time lag between them.

The idea is the AFC to be similar between the original dataset and the imputed one.
Fot the TEMP variable, one sees the good reconstruction for all the algorithms.
On th econtrary, for the PRES variable, all methods overestimates the autocorrelation of the variables, especially the RPCA one.
Finally, for the DEWP variable, the methods cannot impute to obtain a behavior close to the original: the autocorrelation decreases to linearly.

```python
n_columns = len(df_plot.columns)
n_imputers = len(dict_imputers)

fig = plt.figure(figsize=(6 * n_columns, 6))
for i_col, col in enumerate(df_plot):
    ax = fig.add_subplot(1, n_columns, i_col + 1)
    for name_imputer, df_imp in dfs_imputed_station.items():

        acf = utils.acf(df_imp[col])
        plt.plot(acf, label=name_imputer)
    values_orig = df_station[col]
    acf = utils.acf(values_orig)
    plt.plot(acf, color="black", lw=2, ls="--", label="original")
    plt.legend()

plt.savefig("figures/acf.png")
plt.show()

```

```python

```
