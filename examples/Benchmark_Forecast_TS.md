---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: env_qolmat_dev
    language: python
    name: python3
---

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from qolmat.utils import data
from qolmat.benchmark.comparator_forcast_ts import iter_compare_forecast
from sklearn.linear_model import LinearRegression
from qolmat.imputations import imputers
from qolmat.imputations import imputers_pytorch
from qolmat.imputations.diffusions.ddpms import TabDDPM
try:
    import torch.nn as nn
except ModuleNotFoundError:
    raise PyTorchExtraNotInstalled
```

```python
df_data = pd.read_csv('data/beijing+pm2+5+data/PRSA_data_2010.1.1-2014.12.31.csv')
df_data["datetime"] = pd.to_datetime(df_data[["year", "month", "day", "hour"]])
df_data["station"] = "Beijing"
df_data.set_index(["station", "datetime"], inplace=True)
df_data.drop(
    columns=["year", "month", "day", "hour", "No", "cbwd", "Iws", "Is", "Ir"], inplace=True
)
df_data.sort_index(inplace=True)
df_data = df_data.groupby(
    ["station", df_data.index.get_level_values("datetime").floor("d")], group_keys=False
).mean()
df_data = data.add_datetime_features(df_data)
df_data = df_data.loc['Beijing']
df_data = df_data.drop(['pm2.5'], axis=1)

context_length = 365*4
horizon_length = 365

df_train = df_data.loc[df_data.index[0]:df_data.index[0]+pd.DateOffset(days=context_length)]
df_test = df_data.loc[df_train.index[-1]+pd.DateOffset(days=1):df_train.index[-1]+pd.DateOffset(days=horizon_length)]
df_test = df_test.drop(['DEWP', 'PRES', 'time_cos'], axis=1)

for date in df_test.index:
    day_month = date.strftime("%m-%d")
    date_mean = df_train[df_train.index.strftime("%m-%d") == day_month].mean()
    df_test.loc[date, 'DEWP_mean'] = date_mean['DEWP']
    df_test.loc[date, 'TEMP_mean'] = date_mean['TEMP']
    df_test.loc[date, 'PRES_mean'] = date_mean['PRES']
    df_test.loc[date, 'time_cos_mean'] = date_mean['time_cos']

imputer_median = imputers.ImputerMedian()
imputer_residuals = imputers.ImputerResiduals(period=365, model_tsa="additive", extrapolate_trend="freq", method_interpolation="linear")
imputer_tsou = imputers.ImputerEM(model="VAR", method="sample", max_iter_em=34, n_iter_ou=15, dt=1e-3, p=1, random_state=42, verbose=False)
imputer_rpca = imputers.ImputerRPCA(columnwise=False, max_iterations=500, tau=2, lam=0.05, random_state=42, verbose=False)
imputer_knn = imputers.ImputerKNN(n_neighbors=10)
imputer_regressor = imputers.ImputerRegressor(estimator=LinearRegression(), random_state=42)
imputer_mice = imputers.ImputerMICE(estimator=LinearRegression(), sample_posterior=False, max_iter=100, random_state=42)
imputer_diffusion = imputers_pytorch.ImputerDiffusion(model=TabDDPM(num_sampling=5), epochs=100, batch_size=100)

dict_imputers = {
    "median": imputer_median,
    "residuals": imputer_residuals,
    "TSOU": imputer_tsou,
    "RPCA": imputer_rpca,
    "ols": imputer_regressor,
    "mice_ols": imputer_mice,
    "diffusion" : imputer_diffusion,
}

df_forecast_metric = iter_compare_forecast(df_train, df_test, dict_imputers, ration_masked=0.3, mean_size=80, nb_iteration=10)
df_forecast_metric.style.highlight_min(color="green", axis=1)
```

```python

```
