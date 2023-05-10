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

```python
df_data_MLP = df_data.copy()

time = np.concatenate([np.cos(2*np.pi*np.arange(60,366)/365), np.cos(2*np.pi*np.arange(1,366)/365), np.cos(2*np.pi*np.arange(1,366)/365), np.cos(2*np.pi*np.arange(1,367)/366),np.cos(2*np.pi*np.arange(1,60)/365)  ])
for i_station, (station, df) in enumerate(df_data_MLP.groupby("station")):
    df_data_MLP.loc[station, "Time"] = time
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
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
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
        self.estimator.eval()
        x = x.fillna(x.mean())
        x_normalized = self.normalizer_x.transform(x.values)
        inputs = torch.from_numpy(x_normalized).float().to(self.device)
        outputs_normalized = self.estimator.forward(inputs).detach().cpu().numpy()
        outputs = self.normalizer_y.inverse_transform(outputs_normalized)
        return outputs
```

```python
# %%time

dict_imputers["regressor_column_ts"] = imputers.ImputerRegressor(groups=["station"], estimator=LinearRegression(), handler_nan = "column")
dfs_imputed["regressor_column_ts"] = dict_imputers["regressor_column_ts"].fit_transform(df_data_MLP)

# # dict_imputers["regressor_row"] = imputers.ImputerRegressor(groups=["station"], estimator=LinearRegression(), handler_nan = "row")
# # dfs_imputed["regressor_row"] = dict_imputers["regressor_row"].fit_transform(df_data)
# # There is no row for predict()

# dict_imputers["MLP_column_ts"] = imputers_pytorch.ImputerRegressorPytorch(groups=["station"], estimator=feedforward_regressor(input_size=1), handler_nan = "column", batch_size=100, epochs=100)
# dfs_imputed["MLP_column_ts"] = dict_imputers["MLP_column_ts"].fit_transform(df_data_MLP)

# dict_imputers["MLP_fit"] = imputers_pytorch.ImputerRegressorPytorch(groups=["station"], estimator=feedforward_regressor(input_size=10), handler_nan = "fit", batch_size=100, epochs=100)
# dfs_imputed["MLP_fit"] = dict_imputers["MLP_fit"].fit_transform(df_data)

dict_imputers["MLP_fit_ts"] = imputers_pytorch.ImputerRegressorPytorch(groups=["station"], estimator=feedforward_regressor(input_size=11), handler_nan = "fit", batch_size=100, epochs=100)
dfs_imputed["MLP_fit_ts"] = dict_imputers["MLP_fit_ts"].fit_transform(df_data_MLP)
```

### ImputerGenerativeModel: Denoising Diffusion Probabilistic Models - DDPM

```python
from qolmat.imputations import imputers_pytorch
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
from typing import Tuple

class AutoEncoder(torch.nn.Module):
    def __init__(self, input_size):
        super(AutoEncoder, self).__init__()
        self.layer_x_1 = torch.nn.Linear(input_size, 256)
        self.layer_x_2 = torch.nn.Linear(256, 256)

        self.layer_t_1 = torch.nn.Linear(1, 256)
        self.layer_t_2 = torch.nn.Linear(256, 256)

        self.layer_out_1 = torch.nn.Linear(512, 512)
        self.layer_out_2 = torch.nn.Linear(512, input_size)

    def forward(self, x: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        x_1 = torch.nn.functional.relu(self.layer_x_1(x))
        x_2 = torch.nn.functional.relu(self.layer_x_2(x_1))

        t_1 = torch.nn.functional.relu(self.layer_t_1(t))
        t_2 = torch.nn.functional.relu(self.layer_t_2(t_1))

        cat_x_t = torch.cat([x_2, t_2], dim=1)

        out_1 = torch.nn.functional.relu(self.layer_out_1(cat_x_t))
        out_2 = self.layer_out_2(out_1)
        return out_2

class TabDDPM:
    def __init__(self, input_size, noise_steps, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.eps_model = AutoEncoder(input_size).to(self.device)
        self.loss_func = torch.nn.SmoothL1Loss()
        self.optimiser = torch.optim.Adam(self.eps_model.parameters(), lr = 0.0001)

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Section 2, equation 4 and near explation for alpha, alpha hat, beta.
        self.beta = torch.linspace(start=self.beta_start, end=self.beta_end, steps=self.noise_steps, device=self.device) # Linear noise schedule
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # Section 3.2, algorithm 1 formula implementation. Generate values early reuse later.
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)

        # Section 3.2, equation 2 precalculation values.
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.std_beta = torch.sqrt(self.beta)

    def q_sample(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Section 3.2, algorithm 1 formula implementation. Forward process, defined by `q`.
        Found in section 2. `q` gradually adds gaussian noise according to variance schedule. Also,
        can be seen on figure 2.
        """
        sqrt_alpha_hat = self.sqrt_alpha_hat[t].view(-1, 1)
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat[t].view(-1, 1)
        epsilon = torch.randn_like(x, device=self.device)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def fit(self, x, epochs=10, batch_size=100):
        x = x.fillna(x.mean())
        self.normalizer_x = preprocessing.MinMaxScaler()
        x_normalized = self.normalizer_x.fit_transform(x.values)

        self.eps_model.train()
        x_tensor = torch.from_numpy(x_normalized).float()
        dataloader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=True)
        size_dataset = len(dataloader.dataset)
        for epoch in range(epochs):
            loss_epoch = 0.
            for id_batch, x_batch in enumerate(dataloader):
                x_batch = x_batch[0].to(self.device)
                self.optimiser.zero_grad()
                t = torch.randint(low=1, high=self.noise_steps, size=(x_batch.size(dim=0), 1), device=self.device)
                x_batch_t, noise = self.q_sample(x=x_batch, t=t)
                predicted_noise = self.eps_model(x=x_batch_t, t=t.float())
                loss = self.loss_func(predicted_noise, noise)
                loss.backward()
                self.optimiser.step()
                loss_epoch += loss.item()
            # if epoch%20==0:
            #     print(f"Epoch {epoch}, Loss = {loss_epoch/size_dataset}")

    def predict(self, x):
        self.eps_model.eval()
        n_samples = len(x)
        n_features = x.columns.size

        with torch.no_grad():
            noise = torch.randn((n_samples, n_features), device=self.device)

            for i in reversed(range(1, self.noise_steps)):
                t = torch.ones((n_samples, 1), dtype=torch.long, device=self.device) * i

                sqrt_alpha_t = self.sqrt_alpha[t].view(-1, 1)
                beta_t = self.beta[t].view(-1, 1)
                sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1)
                epsilon_t = self.std_beta[t].view(-1, 1)

                random_noise = torch.randn_like(noise) if i > 1 else torch.zeros_like(noise)

                noise = ((1 / sqrt_alpha_t) * (noise - ((beta_t / sqrt_one_minus_alpha_hat_t) * self.eps_model(noise, t.float())))) + (epsilon_t * random_noise)

        outputs_normalized = noise.detach().cpu().numpy()
        outputs_real = self.normalizer_x.inverse_transform(outputs_normalized)
        outputs = pd.DataFrame(outputs_real, columns=x.columns, index=x.index)
        x = x.fillna(outputs)
        return x

class TabDDPMMask(TabDDPM):
    def __init__(self, input_size, noise_steps, beta_start: float = 1e-4, beta_end: float = 0.02):
        super(TabDDPMMask, self).__init__(input_size, noise_steps, beta_start, beta_end)

        self.loss_func_KL = torch.nn.SmoothL1Loss(reduction='none')
        self.loss_func_MSE = torch.nn.MSELoss(reduction='none')

    def fit(self, x, epochs=10, batch_size=100):
        mask_x = ~x.isna().to_numpy()
        x = x.fillna(x.mean())
        self.normalizer_x = preprocessing.MinMaxScaler()
        x_normalized = self.normalizer_x.fit_transform(x.values)

        self.eps_model.train()
        x_tensor = torch.from_numpy(x_normalized).float()
        mask_x_tensor = torch.from_numpy(mask_x)
        dataloader = DataLoader(TensorDataset(x_tensor, mask_x_tensor), batch_size=batch_size, shuffle=True)
        size_dataset = len(dataloader.dataset)
        for epoch in range(epochs):
            loss_epoch = 0.
            for id_batch, (x_batch, mask_x_batch) in enumerate(dataloader):
                x_batch = x_batch.to(self.device)
                mask_x_batch = mask_x_batch.to(self.device).bool().float()

                self.optimiser.zero_grad()
                t = torch.randint(low=1, high=self.noise_steps, size=(x_batch.size(dim=0), 1), device=self.device)
                x_batch_t, noise = self.q_sample(x=x_batch, t=t)
                predicted_noise = self.eps_model(x=x_batch_t, t=t.float())
                loss = (self.loss_func(predicted_noise, noise) * mask_x_batch).mean()
                loss.backward()
                self.optimiser.step()
                loss_epoch += loss.item()
            # if epoch%20==0:
            #     print(f"Epoch {epoch}, Loss = {loss_epoch/size_dataset}")


```

```python
%%time

# dict_imputers["TabDDPM"] = imputers_pytorch.ImputerGenerativeModelPytorch(groups=["station"], model=TabDDPM(input_size=11, noise_steps=50), batch_size=100, epochs=100)
# dfs_imputed["TabDDPM"] = dict_imputers["TabDDPM"].fit_transform(df_data)

dict_imputers["TabDDPM_ts"] = imputers_pytorch.ImputerGenerativeModelPytorch(groups=["station"], model=TabDDPM(input_size=12, noise_steps=50), batch_size=100, epochs=100)
dfs_imputed["TabDDPM_ts"] = dict_imputers["TabDDPM_ts"].fit_transform(df_data_MLP)

dict_imputers["TabDDPM_ts_mask"] = imputers_pytorch.ImputerGenerativeModelPytorch(groups=["station"], model=TabDDPMMask(input_size=12, noise_steps=50), batch_size=100, epochs=100)
dfs_imputed["TabDDPM_ts_mask"] = dict_imputers["TabDDPM_ts_mask"].fit_transform(df_data_MLP)
```

```python
df_error = plot_errors(df_data, dfs_imputed, dict_metrics)
df_error.loc[pd.IndexSlice[:, :], :]
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

search_params = {
    "RPCA_opti": {
        "tau": {"min": .5, "max": 5, "type":"Real"},
        "lam": {"min": .1, "max": 1, "type":"Real"},
    }
}

ratio_masked = 0.1

generator_holes = missing_patterns.EmpiricalHoleGenerator(n_splits=2, groups=["station"], subset = cols_to_impute, ratio_masked=ratio_masked)

comparison = comparator.Comparator(
    dict_imputers,
    df_data_MLP.columns, #Number of columns
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
