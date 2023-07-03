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
import sklearn
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
from qolmat.benchmark.metrics import kl_divergence
from qolmat.benchmark import metrics as qmetrics
from qolmat.imputations import imputers
from qolmat.utils import data, utils, plot

```

<!-- #region tags=[] -->
### **I. Load data**
<!-- #endregion -->

The dataset `Beijing` is the Beijing Multi-Site Air-Quality Data Set. It consists in hourly air pollutants data from 12 chinese nationally-controlled air-quality monitoring sites and is available at https://archive.ics.uci.edu/ml/machine-learning-databases/00501/.
This dataset only contains numerical vairables.

```python
df_data = data.get_data_corrupted("Beijing", ratio_masked=.2, mean_size=120)

# cols_to_impute = ["TEMP", "PRES", "DEWP", "NO2", "CO", "O3", "WSPM"]
# cols_to_impute = df_data.columns[df_data.isna().any()]
cols_to_impute = ["TEMP", "PRES"]

```

The dataset `Artificial` is designed to have a sum of a periodical signal, a white noise and some outliers.

```python tags=[]
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
df = df_data.loc["Aotizhongxin"].fillna(df_data.median())
```

```python
df2 = df.copy()
df2 += 1
```

```python

```

# Density estimation

```python
n_samples = 1000
n_variables = 1
mu1 = np.array([0] * n_variables)
S1 = .1 * np.ones(n_variables) + .9 * np.eye(n_variables)
df1 = pd.DataFrame(np.random.multivariate_normal(mu1, S1, n_samples))

mu2 = np.array([0] * n_variables) + 1
S2 = .1 * np.ones(n_variables) + .9 * np.eye(n_variables)
df2 = pd.DataFrame(np.random.multivariate_normal(mu2, S2, n_samples))
```

```python
n_estimators = 100
estimator = sklearn.ensemble.RandomTreesEmbedding(n_estimators=n_estimators, max_depth=6)
# y = pd.concat([pd.Series([False] * len(df1)), pd.Series([True] * len(df2))])
estimator.fit(df1)
counts1 = qmetrics.density_from_rf(df1, estimator)

counts_bis = qmetrics.density_from_rf(df1, estimator, df_est=df2)
```

```python
df1
```

```python
counts_bis
```

```python
plt.plot(df1, counts1, ".")
```

```python
from scipy.stats import norm


plt.plot(df1, counts1, ".")
plt.gca().twinx()
plt.plot(df2, counts_bis, ".", color=tab10(1))

plt.gca().twinx()
x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
plt.plot(x, norm.pdf(x), 'r-', lw=5, alpha=0.6, label='norm pdf')
plt.show()
```

# KL div estimation

```python
n_samples = 10000
n_variables = 1
mu1 = np.array([0] * n_variables)
S1 = .1 * np.ones(n_variables) + .9 * np.eye(n_variables)
df1 = pd.DataFrame(np.random.multivariate_normal(mu1, S1, n_samples))

mu2 = np.array([0] * n_variables)
mu2[0] = 1
S2 = .1 * np.ones(n_variables) + .9 * np.eye(n_variables)
df2 = pd.DataFrame(np.random.multivariate_normal(mu2, S2, n_samples))
```

```python
qmetrics.kl_divergence(df1, df2, df1.notna(), method="random_forest2")
```

```python
qmetrics.kl_divergence(df1, df2, df1.notna(), method="gaussian")
```

```python

```
