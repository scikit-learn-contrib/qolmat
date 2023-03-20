---
jupyter:
  jupytext:
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

```python
import warnings
# warnings.filterwarnings('error')
```

```python tags=[]
%reload_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
import scipy
import pprint
import sys

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as plticker

from typing import Optional

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
import tensorflow as tf

from qolmat.benchmark import comparator, missing_patterns
from qolmat.benchmark.utils import kl_divergence
from qolmat.imputations import imputers
from qolmat.utils import data, utils, plot

np.random.seed(1234)
tab10 = plt.get_cmap("tab10")
```

# Imputation de données par MLP avec peu de données manquantes

```python
df = data.get_data("Beijing")
df_data = df.copy()
cols_to_impute = ["PRES","TEMP"]

# cols_to_impute = ["TEMP", "PRES", "DEWP", "NO2", "CO", "O3", "WSPM"]
df_data[cols_to_impute] = data.add_holes(pd.DataFrame(df_data[cols_to_impute]), ratio_masked=.2, mean_size=100)
#df_data[cols_to_impute] = data.add_holes(pd.DataFrame(df_data[cols_to_impute]), ratio_masked=.2, mean_size=120)£
```

```python
np.sum(df_data.isna())
```

## Encodage de la temporalité

```python
# Encodage d'une temporalité
time = np.concatenate([np.cos(2*np.pi*np.arange(60,366)/365), np.cos(2*np.pi*np.arange(1,366)/365), np.cos(2*np.pi*np.arange(1,366)/365), np.cos(2*np.pi*np.arange(1,367)/366),np.cos(2*np.pi*np.arange(1,60)/365)  ])
for i_station, (station, df) in enumerate(df_data.groupby("station")):
    df_data.loc[station, "Time"] = time
```

## On retire les lignes contenant un seul NaN dans les features

```python
estimator = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)])

estimator.compile(optimizer='adam',
             loss='mse',
             metrics=['mae'])
handler_nan = "row"
col_imp = cols_to_impute
```

```python tags=[]
df_imputed = df_data.apply(pd.DataFrame.median, result_type="broadcast", axis=0)

for col in col_imp:
    es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, mode='min')

    X = df_data.drop(columns=col, errors="ignore")
    y = df_data[col]

    is_valid = pd.Series(True, index=df_data.index)
    if handler_nan == "fit":
        pass
    elif handler_nan == "row":
        is_valid = (~X.isna().any(axis=1))
    elif handler_nan == "column":
        X = X.dropna(how="any", axis=1)
    else:
        raise ValueError(f"Value '{self.handler_nan}' is not correct for argument `handler_nan'")

    is_na = y.isna()
    if X.empty:
        y_imputed = pd.Series(y.mean(), index=y.index)
    else:
        estimator.fit(X[(~is_na) & is_valid], y[(~is_na) & is_valid], epochs=100, callbacks=[es], verbose =0)
        y_imputed = estimator.predict(X[is_na & is_valid])
    df_imputed.loc[~is_na, col] = y[~is_na]
    df_imputed.loc[is_na & is_valid, col] = y_imputed
```

```python
n_stations = len(df_data.groupby("station").size())
n_cols = len(cols_to_impute)
fig = plt.figure(figsize=(10 * n_stations, 2 * n_cols))
for i_station, (station, df) in enumerate(df_data.groupby("station")):
    df_station = df_data.loc[station]
    df_station_imputed = df_imputed.loc[station]
    for i_col, col in enumerate(cols_to_impute):
        fig.add_subplot(2, n_stations, i_col * n_stations + i_station + 1)
        plt.plot
        plt.plot(df_station_imputed[col], '.r', label=station)
        plt.plot(df_station[col], '.b', label=station)
        plt.ylabel(col, fontsize=12)
        if i_col == 0:
            plt.title(station)
plt.show()
```

## On retire les features contenant un seul NaN

```python
estimator = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)])

estimator.compile(optimizer='adam',
             loss='mse',
             metrics=['mae'])
handler_nan = "column"
col_imp = cols_to_impute
```

```python tags=[]
df_imputed = df_data.apply(pd.DataFrame.median, result_type="broadcast", axis=0)

for col in col_imp:
    es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, mode='min')

    X = df_data.drop(columns=col, errors="ignore")
    y = df_data[col]

    is_valid = pd.Series(True, index=df_data.index)
    if handler_nan == "fit":
        pass
    elif handler_nan == "row":
        is_valid = (~X.isna().any(axis=1))
    elif handler_nan == "column":
        X = X.dropna(how="any", axis=1)
    else:
        raise ValueError(f"Value '{self.handler_nan}' is not correct for argument `handler_nan'")

    is_na = y.isna()
    if X.empty:
        y_imputed = pd.Series(y.mean(), index=y.index)
    else:
        estimator.fit(X[(~is_na) & is_valid], y[(~is_na) & is_valid], epochs=100, callbacks=[es])
        y_imputed = estimator.predict(X[is_na & is_valid])
    df_imputed.loc[~is_na, col] = y[~is_na]
    df_imputed.loc[is_na & is_valid, col] = y_imputed
```

```python
n_stations = len(df_data.groupby("station").size())
n_cols = len(cols_to_impute)
fig = plt.figure(figsize=(10 * n_stations, 2 * n_cols))
for i_station, (station, df) in enumerate(df_data.groupby("station")):
    df_station = df_data.loc[station]
    df_station_imputed = df_imputed.loc[station]
    for i_col, col in enumerate(cols_to_impute):
        fig.add_subplot(2, n_stations, i_col * n_stations + i_station + 1)
        plt.plot
        plt.plot(df_station_imputed[col], '.r', label=station)
        plt.plot(df_station[col], '.b', label=station)
        plt.ylabel(col, fontsize=12)
        if i_col == 0:
            plt.title(station)
plt.show()
```

## Covariance

```python
n_imputers = 1
n_columns = len(cols_to_impute)
fig = plt.figure(figsize=(6 * n_imputers, 6 * n_columns))
i_plot = 1
for i, col in enumerate(cols_to_impute[:-1]):
    ax = fig.add_subplot(n_columns, n_imputers, i_plot)
    plot.compare_covariances(df_data, df_imputed, col, cols_to_impute[i+1], ax, color=tab10(1), label='MLP')
    ax.set_title("Imputation method: MLP", fontsize=20)
    i_plot += 1
    ax.legend()
plt.show()
```

```python

```
