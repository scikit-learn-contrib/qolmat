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
    name: python3
---

# Ablation study for diffusion models

```python
%reload_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from qolmat.imputations import imputers
from qolmat.utils import data

```

```python
import qolmat.benchmark.metrics as mtr

dict_metrics = {
    "wasser": mtr.wasser_distance,
    "KL": mtr.kl_divergence,
    "ks_test": mtr.kolmogorov_smirnov_test, 
}

def plot_errors(df_original, dfs_imputed, dict_metrics):
    dict_errors_df = {}
    for ind, (name, df_imputed) in enumerate(list(dfs_imputed.items())): 
        dict_errors_mtr = {}
        for name_metric in dict_metrics:
            dict_errors_mtr[name_metric] = dict_metrics[name_metric](df_original, df_imputed)

        dict_errors_df[name] = pd.concat(dict_errors_mtr.values(), keys=dict_errors_mtr.keys())

    return pd.DataFrame(dict_errors_df)
```

## **I. Load data**

```python
df_data = data.get_data_corrupted("Beijing", ratio_masked=.2, mean_size=120)

# cols_to_impute = ["TEMP", "PRES", "DEWP", "NO2", "CO", "O3", "WSPM"]
# cols_to_impute = df_data.columns[df_data.isna().any()]
cols_to_impute = ["TEMP", "PRES"]

n_stations = len(df_data.groupby("station").size())
n_cols = len(cols_to_impute)

```

## Baseline imputers

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
    # "mean": imputer_mean,
    # "median": imputer_median,
    # "mode": imputer_mode,
    # "interpolation": imputer_interpol,
    # "spline": imputer_spline,
    # "shuffle": imputer_shuffle,
    # "residuals": imputer_residuals,
    "OU": imputer_ou,
    "TSOU": imputer_tsou,
    # "TSMLE": imputer_tsmle,
    # "RPCA": imputer_rpca,
    # "RPCA_opti": imputer_rpca_opti,
    # "locf": imputer_locf,
    # "nocb": imputer_nocb,
    # "knn": imputer_knn,
    "mice": imputer_mice,
    # "regressor": imputer_regressor,
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

```python
dfs_imputed = {name: imp.fit_transform(df_data) for name, imp in dict_imputers.items()}
```

## Diffusion models


### ImputerRegressor: feed-forward NN regressor

```python
from qolmat.imputations import imputers_pytorch
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing

class feedforward_regressor:
    def __init__(self, input_size):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.estimator = torch.nn.Sequential(
            torch.nn.Linear(input_size, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        ).to(self.device)
        self.loss_func = torch.nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.estimator.parameters(), lr = 0.0001) 

    def fit(self, x, y, epochs=10, batch_size=100):
        x = x.fillna(x.mean())
        self.normalizer_x = preprocessing.MinMaxScaler()
        x_normalized = self.normalizer_x.fit_transform(x.values)

        self.normalizer_y = preprocessing.MinMaxScaler()
        y_normalized = self.normalizer_y.fit_transform(np.expand_dims(y.values, axis=1))

        self.estimator.train()
        x_tensor = torch.from_numpy(x_normalized).float()
        y_tensor = torch.from_numpy(y_normalized).float()
        dataloader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=True)
        dataset_size = len(dataloader.dataset)
        for epoch in range(epochs):
            loss_epoch = 0.
            for id_batch, (x_batch, y_batch) in enumerate(dataloader):
                self.optimiser.zero_grad()
                outputs = self.estimator.forward(x_batch.to(self.device))
                loss = self.loss_func(outputs, y_batch.to(self.device))
                loss.backward()
                self.optimiser.step()
                loss_epoch += loss.item()
            # if epoch%20==0:
            #     print(f"Epoch {epoch}, Loss = {loss_epoch/dataset_size}")
    
    def predict(self, x):
        x = x.fillna(x.mean())
        x_normalized = self.normalizer_x.transform(x.values)
        inputs = torch.from_numpy(x_normalized).float().to(self.device)
        outputs_normalized = self.estimator.forward(inputs).detach().cpu().numpy()
        outputs = self.normalizer_y.inverse_transform(outputs_normalized)
        return outputs
```

```python
%%time
df_data_MLP = df_data.copy()

time = np.concatenate([np.cos(2*np.pi*np.arange(60,366)/365), np.cos(2*np.pi*np.arange(1,366)/365), np.cos(2*np.pi*np.arange(1,366)/365), np.cos(2*np.pi*np.arange(1,367)/366),np.cos(2*np.pi*np.arange(1,60)/365)  ])
for i_station, (station, df) in enumerate(df_data_MLP.groupby("station")):
    df_data_MLP.loc[station, "Time"] = time

dict_imputers["regressor_column"] = imputers.ImputerRegressor(groups=["station"], estimator=LinearRegression(), handler_nan = "column")
dfs_imputed["regressor_column"] = dict_imputers["regressor_column"].fit_transform(df_data_MLP)

# dict_imputers["regressor_row"] = imputers.ImputerRegressor(groups=["station"], estimator=LinearRegression(), handler_nan = "row")
# dfs_imputed["regressor_row"] = dict_imputers["regressor_row"].fit_transform(df_data)
# There is no row for predict()

dict_imputers["MLP_column"] = imputers_pytorch.ImputerRegressorPytorch(groups=["station"], estimator=feedforward_regressor(input_size=1), handler_nan = "column", batch_size=1000, epochs=100)
dfs_imputed["MLP_column"] = dict_imputers["MLP_column"].fit_transform(df_data_MLP)

dict_imputers["MLP_fit"] = imputers_pytorch.ImputerRegressorPytorch(groups=["station"], estimator=feedforward_regressor(input_size=10), handler_nan = "fit", batch_size=1000, epochs=100)
dfs_imputed["MLP_fit"] = dict_imputers["MLP_fit"].fit_transform(df_data)
```

```python
df_error = plot_errors(df_data, dfs_imputed, dict_metrics)
df_error.loc[pd.IndexSlice[:, cols_to_impute], :]
```

### ImputerGenerativeModel: simple diffusion model

```python
from qolmat.imputations import imputers_pytorch
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing

class simpleNN:
    def __init__(self, input_size):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.estimator = torch.nn.Sequential(
            torch.nn.Linear(input_size, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        ).to(self.device)
        self.loss_func = torch.nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.estimator.parameters(), lr = 0.0001) 

    def fit(self, x, y, epochs=10, batch_size=100):
        x = x.fillna(x.mean())
        self.normalizer_x = preprocessing.MinMaxScaler()
        x_normalized = self.normalizer_x.fit_transform(x.values)

        self.normalizer_y = preprocessing.MinMaxScaler()
        y_normalized = self.normalizer_y.fit_transform(np.expand_dims(y.values, axis=1))

        self.estimator.train()
        x_tensor = torch.from_numpy(x_normalized).float()
        y_tensor = torch.from_numpy(y_normalized).float()
        dataloader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=True)
        dataset_size = len(dataloader.dataset)
        for epoch in range(epochs):
            loss_epoch = 0.
            for id_batch, (x_batch, y_batch) in enumerate(dataloader):
                self.optimiser.zero_grad()
                outputs = self.estimator.forward(x_batch.to(self.device))
                loss = self.loss_func(outputs, y_batch.to(self.device))
                loss.backward()
                self.optimiser.step()
                loss_epoch += loss.item()
            # if epoch%20==0:
            #     print(f"Epoch {epoch}, Loss = {loss_epoch/dataset_size}")
    
    def predict(self, x):
        x = x.fillna(x.mean())
        x_normalized = self.normalizer_x.transform(x.values)
        inputs = torch.from_numpy(x_normalized).float().to(self.device)
        outputs_normalized = self.estimator.forward(inputs).detach().cpu().numpy()
        outputs = self.normalizer_y.inverse_transform(outputs_normalized)
        return outputs
```

```python
%%time

dict_imputers["MLP_column"] = imputers_pytorch.ImputerPytorch(groups=["station"], estimator=simpleNN(input_size=1), handler_nan = "column", batch_size=100, epochs=100)
dfs_imputed["MLP_column"] = dict_imputers["MLP_column"].fit_transform(df_data_MLP)

dict_imputers["MLP_fit"] = imputers_pytorch.ImputerPytorch(groups=["station"], estimator=simpleNN(input_size=10), handler_nan = "fit", batch_size=100, epochs=100)
dfs_imputed["MLP_fit"] = dict_imputers["MLP_fit"].fit_transform(df_data)
```

```python
df_error = plot_errors(df_data, dfs_imputed, dict_metrics)
df_error.loc[pd.IndexSlice[:, cols_to_impute], :]
```

## Evaluation

```python
import plotly.graph_objects as go

station = df_data.index.get_level_values("station")[0]
df_station = df_data.loc[station]
dfs_imputed_station = {name: _df.loc[station] for name, _df in dfs_imputed.items()}

for col in cols_to_impute:
    fig = go.Figure()
    values_orig = df_station[col]
    fig.add_trace(go.Scatter(x=values_orig.index, y=values_orig, mode='markers', name='original'))

    for ind, (name, model) in enumerate(list(dict_imputers.items())):
        values_imp = dfs_imputed_station[name][col].copy()
        values_imp[values_orig.notna()] = np.nan

        fig.add_trace(go.Scatter(x=values_imp.index, y=values_imp, mode='markers', name=name))

    fig.update_layout(xaxis_title='Datetime',
                      yaxis_title=col, height=300)
    fig.show()
```

```python
%%time
from qolmat.benchmark import comparator, missing_patterns
from matplotlib import pyplot as plt
from qolmat.utils import plot

generator_holes = missing_patterns.EmpiricalHoleGenerator(n_splits=2, groups=["station"], subset = cols_to_impute, ratio_masked=ratio_masked)

comparison = comparator.Comparator(
    dict_imputers,
    df_data.columns,
    generator_holes = generator_holes,
    n_calls_opt=10,
    search_params=search_params,
)

metrics = ['mae', 'wasser', 'KL']
results = comparison.compare(df_data_MLP, True, metrics)

results
```

```python
fig = plt.figure(figsize=(24, 4))
plot.multibar(results.loc["mae"], decimals=4)
plt.ylabel("mae")
plt.show()
```

```python

```
