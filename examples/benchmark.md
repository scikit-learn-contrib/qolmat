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

```python tags=[]
import warnings
# warnings.filterwarnings('error')
```

```python tags=[]
%reload_ext autoreload
%autoreload 2

from IPython.display import Image

import pandas as pd
from datetime import datetime
import numpy as np
import hyperopt as ho
np.random.seed(1234)
from matplotlib import pyplot as plt
import matplotlib.ticker as plticker

tab10 = plt.get_cmap("tab10")
plt.rcParams.update({'font.size': 18})


from sklearn.linear_model import LinearRegression

from qolmat.benchmark import comparator, missing_patterns
from qolmat.imputations import imputers
from qolmat.utils import data, utils, plot

```

### **I. Load data**


The dataset `Beijing` is the Beijing Multi-Site Air-Quality Data Set. It consists in hourly air pollutants data from 12 chinese nationally-controlled air-quality monitoring sites and is available at https://archive.ics.uci.edu/ml/machine-learning-databases/00501/.
This dataset only contains numerical vairables.

```python tags=[]
df_data = data.get_data_corrupted("Beijing", ratio_masked=.2, mean_size=120)
cols_to_impute = ["TEMP", "PRES"]
```

The dataset `Artificial` is designed to have a sum of a periodical signal, a white noise and some outliers.

```python tags=[]
df_data
```

Let's take a look at variables to impute. We only consider a station, Aotizhongxin.
Time series display seasonalities (roughly 12 months).

```python tags=[]
n_stations = len(df_data.groupby("station").size())
n_cols = len(cols_to_impute)
```

```python tags=[]
fig = plt.figure(figsize=(20 * n_stations, 6 * n_cols))
for i_station, (station, df) in enumerate(df_data.groupby("station")):
    df_station = df_data.loc[station]
    for i_col, col in enumerate(cols_to_impute):
        fig.add_subplot(n_cols, n_stations, i_col * n_stations + i_station + 1)
        plt.plot(df_station[col], label=station)
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

```python tags=[]
ratio_masked = 0.1
```

```python tags=[]
dict_config_opti = {}

imputer_mean = imputers.ImputerSimple(groups=("station",), strategy="mean")
imputer_median = imputers.ImputerSimple(groups=("station",), strategy="median")
imputer_mode = imputers.ImputerSimple(groups=("station",), strategy="most_frequent")
imputer_locf = imputers.ImputerLOCF(groups=("station",))
imputer_nocb = imputers.ImputerNOCB(groups=("station",))
imputer_interpol = imputers.ImputerInterpolation(groups=("station",), method="linear")
imputer_spline = imputers.ImputerInterpolation(groups=("station",), method="spline", order=2)
imputer_shuffle = imputers.ImputerShuffle(groups=("station",))
imputer_residuals = imputers.ImputerResiduals(groups=("station",), period=365, model_tsa="additive", extrapolate_trend="freq", method_interpolation="linear")

imputer_rpca = imputers.ImputerRpcaNoisy(groups=("station",), columnwise=False, max_iterations=500, tau=.01, lam=5, rank=1)
imputer_rpca_opti = imputers.ImputerRpcaNoisy(groups=("station",), columnwise=False, max_iterations=256)
dict_config_opti["RPCA_opti"] = {
    "tau": ho.hp.uniform("tau", low=.5, high=5),
    "lam": ho.hp.uniform("lam", low=.1, high=1),
}
imputer_rpca_opticw = imputers.ImputerRpcaNoisy(groups=("station",), columnwise=False, max_iterations=256)
dict_config_opti["RPCA_opticw"] = {
    "tau/TEMP": ho.hp.uniform("tau/TEMP", low=.5, high=5),
    "tau/PRES": ho.hp.uniform("tau/PRES", low=.5, high=5),
    "lam/TEMP": ho.hp.uniform("lam/TEMP", low=.1, high=1),
    "lam/PRES": ho.hp.uniform("lam/PRES", low=.1, high=1),
}

imputer_normal_sample = imputers.ImputerEM(groups=("station",), model="multinormal", method="sample", max_iter_em=8, n_iter_ou=128, dt=4e-2)
imputer_var_sample = imputers.ImputerEM(groups=("station",), model="VAR", method="sample", max_iter_em=8, n_iter_ou=128, dt=4e-2, p=1)
imputer_var_max = imputers.ImputerEM(groups=("station",), model="VAR", method="mle", max_iter_em=32, n_iter_ou=128, dt=4e-2, p=1)

imputer_knn = imputers.ImputerKNN(groups=("station",), n_neighbors=10)
imputer_mice = imputers.ImputerMICE(groups=("station",), estimator=LinearRegression(), sample_posterior=False, max_iter=100)
imputer_regressor = imputers.ImputerRegressor(groups=("station",), estimator=LinearRegression())
```

```python tags=[]
generator_holes = missing_patterns.EmpiricalHoleGenerator(n_splits=1, groups=("station",), subset=cols_to_impute, ratio_masked=ratio_masked)
```

```python tags=[]
dict_imputers = {
    "mean": imputer_mean,
    # "median": imputer_median,
    # "mode": imputer_mode,
    # "interpolation": imputer_interpol,
    # "spline": imputer_spline,
    # "shuffle": imputer_shuffle,
    "residuals": imputer_residuals,
    "Normal_sample": imputer_normal_sample,
    "VAR_sample": imputer_var_sample,
    "VAR_max": imputer_var_max,
    "RPCA": imputer_rpca,
    # "RPCA_opti": imputer_rpca,
    # "RPCA_opticw": imputer_rpca_opti2,
    # "locf": imputer_locf,
    # "nocb": imputer_nocb,
    # "knn": imputer_knn,
    "OLS": imputer_regressor,
    "MICE_OLS": imputer_mice,
}
n_imputers = len(dict_imputers)
```

In order to compare the methods, we $i)$ artificially create missing data (for missing data mechanisms, see the docs); $ii)$ then impute it using the different methods chosen and $iii)$ calculate the reconstruction error. These three steps are repeated a number of times equal to `n_splits`. For each method, we calculate the average error and compare the final errors.

<p align="center">
    <img src="https://raw.githubusercontent.com/Quantmetry/qolmat/main/docs/images/schema_qolmat.png"  width=50% height=50%>
</p>



Concretely, the comparator takes as input a dataframe to impute, a proportion of nan to create, a dictionary of imputers (those previously mentioned), a list with the columns names to impute, a generator of holes specifying the type of holes to create and the search dictionary search_params for hyperparameter optimization.

Note these metrics compute reconstruction errors; it tells nothing about the distances between the "true" and "imputed" distributions.

```python tags=[]
metrics = ["mae", "wmape", "kl_columnwise", "frechet"]
comparison = comparator.Comparator(
    dict_imputers,
    cols_to_impute,
    generator_holes = generator_holes,
    metrics=metrics,
    max_evals=2,
    dict_config_opti=dict_config_opti,
)
results = comparison.compare(df_data)
results.style.highlight_min(color="lightgreen", axis=1)
```

```python
n_metrics = len(metrics)
fig = plt.figure(figsize=(24, 4 * n_metrics))
for i, metric in enumerate(metrics):
    fig.add_subplot(n_metrics, 1, i + 1)
    df = results.loc[metric]
    plot.multibar(df, decimals=2)
    plt.ylabel(metric)

#plt.savefig("figures/imputations_benchmark_errors.png")
plt.show()
```

### **III. Comparison of methods**


We now run just one time each algorithm on the initial corrupted dataframe and visualize the different imputations.

```python tags=[]
df_plot = df_data[cols_to_impute]
```

```python
df_plot = data.add_datetime_features(df_plot, col_time="date")
```

```python tags=[]
dfs_imputed = {name: imp.fit_transform(df_plot) for name, imp in dict_imputers.items()}
```

```python tags=[]
station = df_plot.index.get_level_values("station")[0]
# station = "Huairou"
df_station = df_plot.loc[station]
dfs_imputed_station = {name: df_plot.loc[station] for name, df_plot in dfs_imputed.items()}
```

```python tags=[]
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

```python tags=[]
n_columns = len(cols_to_impute)
n_imputers = len(dict_imputers)

fig = plt.figure(figsize=(12 * n_imputers, 4 * n_columns))
i_plot = 1
for i_col, col in enumerate(cols_to_impute):
    for name_imputer, df_imp in dfs_imputed_station.items():

        ax = fig.add_subplot(n_columns, n_imputers, i_plot)
        values_orig = df_station[col]

        values_imp = df_imp[col].copy()
        values_imp[values_orig.notna()] = np.nan
        plt.plot(values_imp, marker="o", color=tab10(0), label="imputation", alpha=1)
        plt.plot(values_orig, color='black', marker="o", label="original")
        plt.ylabel(col, fontsize=16)
        if i_plot % n_imputers == 0:
            plt.legend(loc="lower right", fontsize=18)
        plt.xticks(rotation=15)
        if i_col == 0:
            plt.title(name_imputer)
        if i_col != n_columns - 1:
            ax.set_xticklabels([])
        loc = plticker.MultipleLocator(base=2*365)
        ax.xaxis.set_major_locator(loc)
        ax.tick_params(axis='both', which='major')
        i_plot += 1

plt.show()

```

## (Optional) Deep Learning Model


In this section, we present an MLP model of data imputation using PyTorch, which can be installed using a "pip install qolmat[pytorch]".

```python
from qolmat.imputations import imputers_pytorch
from qolmat.imputations.diffusions.ddpms import TabDDPM
try:
    import torch.nn as nn
except ModuleNotFoundError:
    raise PyTorchExtraNotInstalled
```

For the example, we use a simple MLP model with 3 layers of neurons.
Then we train the model without taking a group on the stations

```python
import numpy as np
from qolmat.imputations.imputers_pytorch import ImputerDiffusion
from qolmat.imputations.diffusions.ddpms import TabDDPM

X = np.array([[1, 1, 1, 1], [np.nan, np.nan, 3, 2], [1, 2, 2, 1], [2, 2, 2, 2]])
imputer = ImputerDiffusion(model=TabDDPM(random_state=11), epochs=50, batch_size=1)

imputer.fit_transform(X)
```

```python
import numpy as np
from qolmat.imputations.imputers_pytorch import ImputerDiffusion
from qolmat.imputations.diffusions.ddpms import TabDDPM

X = np.array([[1, 1, 1, 1], [np.nan, np.nan, 3, 2], [1, 2, 2, 1], [2, 2, 2, 2]])
imputer = ImputerDiffusion(model=TabDDPM(random_state=11), epochs=50, batch_size=1)

imputer.fit_transform(X)
```

```python
1.33573675, 1.40472937
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

```python
n_variables = len(cols_to_impute)

estimator = imputers_pytorch.build_mlp(input_dim=n_variables-1, list_num_neurons=[256,128,64])
encoder, decoder  = imputers_pytorch.build_autoencoder(input_dim=n_variables,latent_dim=4, output_dim=n_variables, list_num_neurons=[4*4, 2*4])
```

```python
dict_imputers["MLP"] = imputer_mlp = imputers_pytorch.ImputerRegressorPyTorch(estimator=estimator, groups=('station',), epochs=500)
dict_imputers["Autoencoder"] = imputer_autoencoder = imputers_pytorch.ImputerAutoencoder(encoder, decoder, max_iterations=100, epochs=100)
dict_imputers["Diffusion"] = imputer_diffusion = imputers_pytorch.ImputerDiffusion(model=TabDDPM(num_sampling=5), epochs=100, batch_size=100)
```

We can re-run the imputation model benchmark as before.
```python
comparison = comparator.Comparator(
    dict_imputers,
    cols_to_impute,
    generator_holes = generator_holes,
    metrics=metrics,
    max_evals=2,
    dict_config_opti=dict_config_opti,
)
```

```python tags=[]
generator_holes = missing_patterns.EmpiricalHoleGenerator(n_splits=3, groups=('station',), subset=cols_to_impute, ratio_masked=ratio_masked)

comparison = comparator.Comparator(
    dict_imputers,
    cols_to_impute,
    generator_holes = generator_holes,
    metrics=metrics,
    max_evals=2,
    dict_config_opti=dict_config_opti,
)
results = comparison.compare(df_data)
results.style.highlight_min(color="green", axis=1)
```
```python
n_metrics = len(metrics)
fig = plt.figure(figsize=(24, 4 * n_metrics))
for i, metric in enumerate(metrics):
    fig.add_subplot(n_metrics, 1, i + 1)
    df = results.loc[metric]
    plot.multibar(df, decimals=2)
    plt.ylabel(metric)

#plt.savefig("figures/imputations_benchmark_errors.png")
plt.show()
```

```python tags=[]
df_plot = df_data[cols_to_impute]
```

```python tags=[]
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

n_columns = len(cols_to_impute)
n_imputers = len(dict_imputers)

fig = plt.figure(figsize=(12 * n_imputers, 4 * n_columns))
i_plot = 1
for i_col, col in enumerate(cols_to_impute):
    for name_imputer, df_imp in dfs_imputed_station.items():

        ax = fig.add_subplot(n_columns, n_imputers, i_plot)
        values_orig = df_station[col]

        values_imp = df_imp[col].copy()
        values_imp[values_orig.notna()] = np.nan
        plt.plot(values_imp, marker="o", color=tab10(0), label="imputation", alpha=1)
        plt.plot(values_orig, color='black', marker="o", label="original")
        plt.ylabel(col, fontsize=16)
        if i_plot % n_imputers == 0:
            plt.legend(loc="lower right", fontsize=18)
        plt.xticks(rotation=15)
        if i_col == 0:
            plt.title(name_imputer)
        if i_col != n_columns - 1:
            ax.set_xticklabels([])
        loc = plticker.MultipleLocator(base=2*365)
        ax.xaxis.set_major_locator(loc)
        ax.tick_params(axis='both', which='major')
        i_plot += 1

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
        ax.set_title(f"{name_imputer}", fontsize=20)
        i_plot += 1
        ax.legend()
plt.show()
```

## Auto-correlation


We are now interested in the auto-correlation function (ACF). As seen before, time series display seaonal patterns.
[Autocorrelation](https://en.wikipedia.org/wiki/Autocorrelation) is the correlation of a signal with a delayed copy of itself as a function of delay. It measures the similarity between observations of a random variable as a function of the time lag between them. The objective is to have an ACF to be similar between the original dataset and the imputed one.

```python
n_columns = len(df_plot.columns)
n_imputers = len(dict_imputers)

fig = plt.figure(figsize=(9 * n_columns, 6))
for i_col, col in enumerate(df_plot):
    ax = fig.add_subplot(1, n_columns, i_col + 1)
    for name_imputer, df_imp in dfs_imputed_station.items():

        acf = utils.acf(df_imp[col])
        plt.plot(acf, label=name_imputer)
    values_orig = df_station[col]
    acf = utils.acf(values_orig)
    plt.plot(acf, color="black", lw=2, ls="--", label="original")
    ax.set_title(f"{col}", fontsize=20)
    plt.legend()

plt.savefig("figures/acf.png")
plt.show()

```

```python

```
