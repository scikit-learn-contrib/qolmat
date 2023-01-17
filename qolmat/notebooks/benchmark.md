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
    display_name: env_qolmat
    language: python
    name: env_qolmat
---

**This notebook aims to present the Qolmat repo through an example of a multivariate time series.
In Qolmat, a few data imputation methods are implemented as well as a way to evaluate their performance.**


First, import some useful librairies

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

import seaborn as sns

sns.set_context("paper")
sns.set_style("whitegrid", {'axes.grid' : False})
sns.set_theme(style="ticks")

tab10 = plt.get_cmap("tab10")

from typing import Optional

# models for imputations
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor


import sys
# sys.path.append("../../")
from qolmat.benchmark import comparator, missing_patterns
from qolmat.benchmark.utils import kl_divergence
from qolmat.imputations import models
from qolmat.utils import data, utils, plot
from qolmat.imputations.em_sampler import ImputeEM
# from qolmat.drawing import display_bar_table

```

### **I. Load data**


The data used in this example is the Beijing Multi-Site Air-Quality Data Set. It consists in hourly air pollutants data from 12 chinese nationally-controlled air-quality monitoring sites and is available at https://archive.ics.uci.edu/ml/machine-learning-databases/00501/.
This dataset only contains numerical vairables.


df = pd.read_csv("/Users/hlbotterman/Downloads/m5-daily-sales.csv")
df.head()
df["Date"] = pd.to_datetime(df["Date"])
df = df.rename(columns={"Date": "datetime"})
df.set_index("datetime", inplace = True)
df["Sales"] = df['Sales'].astype(float)
cols_to_impute = ["Sales"]

```python
download = True
df_data = data.get_data_corrupted(download=download, ratio_masked=.2, mean_size=120 , groups=["station"])

# cols_to_impute = ["TEMP", "PRES", "DEWP", "NO2", "CO", "O3", "WSPM"]
# cols_to_impute = df_data.columns[df_data.isna().any()]
cols_to_impute = ["TEMP", "PRES"]

```

Let's take a look at variables to impute. We only consider a station, Aotizhongxin.
Time series display seasonalities (roughly 12 months).

```python
n_stations = len(df_data.groupby("station").size())
n_cols = len(cols_to_impute)
```

```python
fig = plt.figure(figsize=(10 * n_stations, 2 * n_cols))
for i_station, (station, df) in enumerate(df_data.groupby("station")):
    for i_col, col in enumerate(cols_to_impute):
        fig.add_subplot(n_cols, n_stations, i_col * n_stations + i_station + 1)
        plt.plot(df.reset_index().datetime, df[col], '.', label=station)
        # break
        plt.ylabel(col, fontsize=12)
        if i_col == 0:
            plt.title(station)
plt.show()
```

### **II. Imputation methods**


This part is devoted to the imputation methods. The idea is to try different algorithms and compare them.

<u>**Methods**</u>:
There are two kinds of methods. The first one is not specific to multivariate time series, as for instance ImputeByMean: columns with missing values are imputed separetaly, i.e. possible correlations are not taken into account. The second one is specific to multivariate time series, where other columns are needed to impute another.

For the ImputeByMean or ImputeByMedian, the user is allow to specify a list of variables indicating how the groupby will be made, to impute the data. More precisely, data are first grouped and then the mean or median of each group is computed. The missign values are then imputed by the corresponding mean or median. If nothing is passed, then the mean or median of each column is used as value to impute.

The ImputeOnResiduals method procedes in 3 steps. First, time series are decomposed (seasonality, trend and residuals). Then the residuals are imputed thanks to a interpolation method. And finally, time series are recomposed.

For more information about the methods, we invite you to read the docs.

<u>**Hyperparameters' search**</u>:
Some methods require hyperparameters. The user can directly specify them, or he can procede to a search. To do so, he has to define a search_params dictionary, where keys are the imputation method's name and the values are a dictionary specifyign the minimum, maximum or list of categories and type of values (Integer, Real or Category) to search.
In pratice, we rely on a cross validation to find the best hyperparams values minimising an error reconstruction.

```python
imputer_mean = models.ImputeByMean(["datetime.dt.month"]) #["datetime.dt.month", "datetime.dt.dayofweek"]
imputer_median = models.ImputeByMedian(["datetime.dt.month"]) # ["datetime.dt.month", "datetime.dt.dayofweek"] #, "datetime.dt.round('10min')"])
imputer_interpol = models.ImputeByInterpolation(method="linear")
imputer_spline = models.ImputeByInterpolation(method="spline")
imputer_random = models.ImputeRandom()
imputer_residuals = models.ImputeOnResiduals("additive", 7, "freq", "linear")
imputer_rpca = models.ImputeRPCA(
  method="temporal", multivariate=False, **{"n_rows":7*4, "maxIter":1000, "tau":1, "lam":0.7}
  )
imputer_em = ImputeEM(n_iter_em=34, n_iter_ou=15, verbose=0, strategy="ou", temporal=False)
imputer_locf = models.ImputeLOCF()
imputer_nocb = models.ImputeNOCB()
imputer_knn = models.ImputeKNN(k=10)
imputer_iterative = models.ImputeMICE(
  **{"estimator": LinearRegression(), "sample_posterior": False, "max_iter": 100, "missing_values": np.nan}
  )
impute_regressor = models.ImputeRegressor(
  HistGradientBoostingRegressor(), cols_to_impute=cols_to_impute
)
impute_stochastic_regressor = models.ImputeStochasticRegressor(
  HistGradientBoostingRegressor(), cols_to_impute=cols_to_impute
)
impute_mfe = models.ImputeMissForest()

dict_models = {
    #"mean": imputer_mean,
    #"median": imputer_median,
    "interpolation": imputer_interpol,
    #"residuals": imputer_residuals,
    #"iterative": imputer_iterative,
    "EM": imputer_em,
    #"RPCA": imputer_rpca,
}
n_models = len(dict_models)


search_params = {
  # "ImputeRPCA": {
  #   "lam": {"min": 0.5, "max": 1, "type":"Real"},
  #   "tau": {"min": 1, "max": 1.5, "type":"Real"},
  # }
}

prop_nan = 0.05
filter_value_nan = 0.01
```

In order to compare the methods, we $i)$ artificially create missing data (for missing data mechanisms, see the docs); $ii)$ then impute it using the different methods chosen and $iii)$ calculate the reconstruction error. These three steps are repeated a cv number of times. For each method, we calculate the average error and compare the final errors.

<p align="center">
    <img src="../../docs/images/comparator.png"  width=50% height=50%>
</p>



Concretely, the comparator takes as input a dataframe to impute, a proportion of nan to create, a dictionary of models (those previously mentioned), a list with the columns names to impute, the number of articially corrupted dataframe to create: n_samples, the search dictionary search_params, and possibly a threshold for filter the nan value.

Then, it suffices to use the compare function to obtain the results: a dataframe with different metrics.
This allows an easy comparison of the different imputations.

Note these metrics compute reconstruction errors; it tells nothing about the distances between the "true" and "imputed" distributions.

```python
doy = pd.Series(df_data.reset_index().datetime.dt.isocalendar().week.values, index=df_data.index)

generator_holes = missing_patterns.EmpiricalHoleGenerator(n_splits=2, groups=["station"], ratio_masked=0.1)
# generator_holes = missing_patterns.GeometricHoleGenerator(n_splits=10, groups=["station"], ratio_masked=0.1)
# generator_holes = missing_patterns.UniformHoleGenerator(n_splits=2, ratio_masked=0.4)
# generator_holes = missing_patterns.GroupedHoleGenerator(n_splits=2, groups=["station", doy], ratio_masked=0.4)
# generator_holes = missing_patterns.MultiMarkovHoleGenerator(n_splits=2, groups=["station"], ratio_masked=0.1)

comparison = comparator.Comparator(
    dict_models,
    cols_to_impute,
    generator_holes = generator_holes,
    n_cv_calls=2,
    search_params=search_params,
)
results = comparison.compare(df_data)
results
```

### **IV. Comparison of methods**


We now run just one time each algorithm on the initial corrupted dataframe and compare the different performances through multiple analysis.

```python
dfs_imputed = {name: imp.fit_transform(df_data) for name, imp in dict_models.items()}
```

```python
station = "Aotizhongxin"
df_station = df_data.loc[station]
dfs_imputed_station = {name: df.loc[station] for name, df in dfs_imputed.items()}
```

Let's look at the imputations.
When the data is missing at random, imputation is easier. Missing block are more challenging.
Note here we didn't fit the hyperparams of the RPCA... results might be of poor quality...

```python
palette = sns.color_palette("icefire", n_colors=len(dict_models))
#palette = sns.color_palette("husl", 8)
sns.set_palette(palette)
markers = ["o", "s", "D", "+", "P", ">", "^", "d"]
colors = ["tab:red", "tab:blue", "tab:blue"]


for col in cols_to_impute:
    fig, ax = plt.subplots(figsize=(10, 3))
    values_orig = df_station[col]

    plt.plot(values_orig, ".", color='black', label="original")
    #plt.plot(df.iloc[870:1000][col], markers[0], color='k', linestyle='-' , ms=3)

    for ind, (name, model) in enumerate(list(dict_models.items())):
        values_imp = dfs_imputed_station[name][col].copy()
        values_imp[values_orig.notna()] = np.nan
        plt.plot(values_imp, ".", color=colors[ind], label=name, alpha=1)
    plt.ylabel(col, fontsize=16)
    plt.legend(loc=[1, 0], fontsize=18)
    loc = plticker.MultipleLocator(base=2*365)
    ax.xaxis.set_major_locator(loc)
    ax.tick_params(axis='both', which='major', labelsize=17)
    sns.despine()
    plt.show()
```

**IV.a. Covariance**


We first check the covariance. We simply plot one variable versus one another.
One observes the methods provide similar visual resuls: it's difficult to compare them based on this criterion.

```python
for i_model, model in enumerate(dict_models.keys()):
    fig, axs = plt.subplots(1, len(cols_to_impute)-1, figsize=(4 * (len(cols_to_impute)-1), 4))
    df_imp = dfs_imputed_station[model]
    for i in range(len(cols_to_impute)-1):
        plot.compare_covariances(df_station, df_imp, cols_to_impute[i], cols_to_impute[i+1], axs, color=tab10(i_model))
        axs.set_title(f"imputation method: {model}", fontsize=20)
        sns.despine()
    plt.show()
```

**IV.b. Auto-correlation**


We are now interested in th eauto-correlation function (ACF). As seen before, time series display seaonal patterns.
[Autocorrelation](https://en.wikipedia.org/wiki/Autocorrelation) is the correlation of a signal with a delayed copy of itself as a function of delay. Informally, it is the similarity between observations of a random variable as a function of the time lag between them.

The idea is the AFC to be similar between the original dataset and the imputed one.
Fot the TEMP variable, one sees the good reconstruction for all the algorithms.
On th econtrary, for the PRES variable, all methods overestimates the autocorrelation of the variables, especially the RPCA one.
Finally, for the DEWP variable, the methods cannot impute to obtain a behavior close to the original: the autocorrelation decreases to linearly.

```python
from statsmodels.tsa.stattools import acf

palette = sns.dark_palette("b", n_colors=len(dict_models), reverse=False)
sns.set_palette(palette)
markers = ["o", "s", "*", "D", "P", ">", "^", "d"]

fig, axs = plt.subplots(1, len(cols_to_impute), figsize=(16, 2))
for i, col in enumerate(cols_to_impute):
    axs[i].plot(acf(df_station[col].dropna()), color="k", marker=markers[0], lw=0.8)
    for j, (name, df) in enumerate(dfs_imputed_station.items()):
        axs[i].plot(acf(df[col]), marker=markers[j+1], lw=0.8)
    axs[i].set_xlabel("Lags [days]", fontsize=15)
    axs[i].set_ylabel("Correlation", fontsize=15)
    axs[i].set_ylim([0.5, 1])
    axs[i].set_title(col, fontsize=15)
axs[-1].legend(["Original dataset"] +  list(dfs_imputed.keys()), loc=[1, 0])
sns.despine()
```

**IV.b. Distances between distributions**


We are now interested in a way for quantifying the distance between two distributions.
Until now, we look at the reconstruction error, whatever the distributions.

There is a plethora of methods to quantify the distance between distributions $P$ and $Q$.
For instance, those based on the information theory as for instance, the well-known [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence). A simple interpretation of the KL divergence of $P$ from $Q$ is the expected excess surprise from using $Q$ as a model when the actual distribution is $P$.

A drawback with this divergence is it ignores the underlying geometry of the space (the KL divergence is somewhat difficult to intuitively interpret).
As a remedy, we consider a second metric, the [Wasserstein distance](https://en.wikipedia.org/wiki/Wasserstein_metric), a distance function defined between probability distributions on a given metric space $M$.

To understand one of the differences between these two quantities, let us look at this simple example.
The KL between the 2 distributions on the left is the same as that of the 2 distributions on the right: the KL divergence does not take into account the underlying metric space. Conversely, the Wasserstein metric is larger for those on the left since the "transport" is greater than for those on the right.

<p align="center">
    <img src="../../docs/images/KL_wasser.png"  width=50% height=50%>
</p>


```python
df_kl = pd.DataFrame(np.nan, index=dfs_imputed_station.keys(), columns=cols_to_impute)
for model, df_imputed in dfs_imputed_station.items():
    for col in cols_to_impute:
        kl = kl_divergence(df_station[[col]].dropna(how="all"), df_imputed[[col]]).iloc[0]
        df_kl.loc[model, col] = kl

plot.display_bar_table(df_kl, ylabel="KL divergence")
```

```python
df_wasserstein = pd.DataFrame(np.nan, index=dfs_imputed_station.keys(), columns=cols_to_impute)
for model, df_imputed in dfs_imputed_station.items():
    for col in cols_to_impute:
        wasserstein = scipy.stats.wasserstein_distance(df_station[col].dropna(how="all"), df_imputed[col])
        df_wasserstein.loc[model, col] = wasserstein

plot.display_bar_table(df_wasserstein, ylabel="Wasserstein distance")
```

```python

```
