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

# Benchmark for diffusion models


**ImputerGenerativeModel: Denoising Diffusion Probabilistic Models - [DDPM](https://arxiv.org/abs/2006.11239)**

- Forward: $x_0 \rightarrow x_1 \rightarrow \dots \rightarrow x_{T-1} \rightarrow x_T$
    - $q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)$
    - $x_t = \bar{\alpha}_t \times x_0 + \sqrt{1-\bar{\alpha}_t} \times \epsilon$ where
        - $\epsilon \sim \mathcal{N}(0,I)$
        - $\bar{\alpha}_t = \sum^t_{t=0} \alpha_t$
        - $\alpha$: noise scheduler

- Reserve: $x_T \rightarrow x_{t-1} \rightarrow \dots \rightarrow x_1 \rightarrow x_0$
    - $p_\theta (x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta (x_t, t), \Sigma_\theta (x_t, t))$
    - $x_{t-1} = \frac{1}{\sqrt{\alpha_t}} (x_t - \frac{1 - \alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)) + \sigma_t z$ where
        - $\epsilon$: our model to predict noise at t
        - $z \sim \mathcal{N}(0,I)$

- Objective function:
    - $E_{t \sim \mathcal{U} [[1,T]], x_0 \sim q(x_0), \epsilon \sim \mathcal{N}(0,I)} [|| \epsilon - \epsilon_\theta(x_t, t)||^2]$

**TabDDPM**

- Training:
    - Fill real nan with mean
    - Self-training: compute only loss values from observed data (CSDI)
    - More complex autoencoder based on ResNet [Gorishniy et al., 2021](https://arxiv.org/abs/2106.11959) ([code](https://github.com/Yura52/rtdl))
    - Embedding of noise steps (CSDI)
- Inference:
    - $\epsilon \rightarrow \hat{x}_t \rightarrow \hat{x}_0$ where
        - $\hat{x}_t = mask * x_0 + (1 - mask) * \hat{x}_t$
        - $mask$: 1 = observed values
    - Fill nan with $\hat{x}_0$

**TabDDPMTS**
- Sliding window method: obtain a list of data chunks
- Apply Transformer Encoder to encode the relationship between times in a chunk


```python
%reload_ext autoreload
%autoreload 2

import os
import sys
sys.path.append('/home/ec2-ngo/qolmat/')

# import warnings
# warnings.filterwarnings('error')

import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
import inspect
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from qolmat.benchmark import comparator, missing_patterns, metrics
from qolmat.imputations import imputers, imputers_pytorch
from qolmat.imputations.diffusions import diffusions
from qolmat.utils import data, plot
```

```python
dict_metrics = {
    "mae": metrics.mean_absolute_error,
    "wasser": metrics.wasserstein_distance,
}

def plot_errors(df_original, dfs_imputed, dfs_mask, dict_metrics, cols_to_impute, **kwargs):
    dict_errors_df = {}
    for ind, (name, df_imputed) in enumerate(list(dfs_imputed.items())):
        dict_errors_mtr = {}
        for name_metric in dict_metrics:
            metric_args = list(inspect.signature(dict_metrics[name_metric]).parameters)
            args_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in metric_args}
            if "wasser" in name_metric:
                dict_errors_mtr[name_metric] = dict_metrics[name_metric](df_original[cols_to_impute], df_imputed[cols_to_impute], dfs_mask[cols_to_impute])
            else:
                dict_errors_mtr[name_metric] = dict_metrics[name_metric](df_original[cols_to_impute], df_imputed[cols_to_impute], dfs_mask[cols_to_impute], **args_dict)
        dict_errors_df[name] = pd.concat(dict_errors_mtr.values(), keys=dict_errors_mtr.keys())

    return pd.DataFrame(dict_errors_df)

def plot_summaries(summaries, display='epoch_loss', xaxis_title='epoch', height=500):
    fig = go.Figure()

    if display == 'num_params':
        values_selected = []
        for ind, (name, values) in enumerate(list(summaries.items())):
            values_selected.append(values[display])
        fig.add_trace(go.Bar(x=list(summaries.keys()), y=np.squeeze(values_selected)))
        return fig

    for ind, (name, values) in enumerate(list(summaries.items())):
        values_selected = values[display]
        fig.add_trace(go.Scatter(x=list(range(len(values_selected))), y=values_selected, mode='lines', name=name))

    fig.update_layout(xaxis_title=xaxis_title,
                      yaxis_title=display, height=height)

    fig.update_yaxes(type="log")

    return fig
```

## **I. Load data**

```python
df_data_raw = data.get_data("Beijing_offline", datapath='../data')
df_data = data.add_holes(df_data_raw, ratio_masked=.2, mean_size=120)

# cols_to_impute = ["TEMP", "PRES", "DEWP", "NO2", "CO", "O3", "WSPM"]
# cols_to_impute = df_data.columns[df_data.isna().any()]
cols_to_impute = ["TEMP", "PRES"]

n_stations = len(df_data.groupby("station").size())
n_cols = len(cols_to_impute)

df_mask = df_data.isna()
df_mask[df_data_raw.isna()] = False

```

```python
display(df_data.describe())
display(df_data.isna().sum())
```

## II. Baseline imputers

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

imputer_ou = imputers.ImputerEM(groups=["station"], model="multinormal", method="sample", max_iter_em=34, n_iter_ou=15, dt=1e-3)
imputer_tsou = imputers.ImputerEM(groups=["station"], model="VAR1", method="sample", max_iter_em=34, n_iter_ou=15, dt=1e-3)
imputer_tsmle = imputers.ImputerEM(groups=["station"], model="VAR1", method="mle", max_iter_em=34, n_iter_ou=15, dt=1e-3)

imputer_knn = imputers.ImputerKNN(groups=["station"], k=10)
imputer_mice = imputers.ImputerMICE(groups=["station"], estimator=LinearRegression(), sample_posterior=False, max_iter=100, missing_values=np.nan)
imputer_regressor = imputers.ImputerRegressor(groups=["station"], estimator=LinearRegression())

dict_imputers_baseline = {
    # "mean": imputer_mean,
    # "median": imputer_median,
    # "mode": imputer_mode,
    "interpolation": imputer_interpol,
    # "spline": imputer_spline,
    # "shuffle": imputer_shuffle,
    # "residuals": imputer_residuals,
    "OU": imputer_ou,
    "TSOU": imputer_tsou,
    "TSMLE": imputer_tsmle,
    # "RPCA": imputer_rpca,
    "RPCA_opti": imputer_rpca_opti,
    "locf": imputer_locf,
    "nocb": imputer_nocb,
    "knn": imputer_knn,
    "mice": imputer_mice,
    # "regressor": imputer_regressor,
}

n_imputers = len(dict_imputers_baseline)
```

## III. Hyperparameter tuning

```python
station = df_data_raw.index.get_level_values("station").unique()[0]
df_valid = df_data_raw.loc[station].dropna()
df_valid_mask = df_mask.loc[station].loc[df_valid.index]

df_train = df_data.loc[station]

print(f"Train: {len(df_train)}, Valid: {len(df_valid)} ")

summaries = {}
```

```python
%%time

hyperparams_tuning = {
    'num_noise_steps': [50, 100, 200, 300],
    'dim_embedding': [128, 256, 512],
}

for name_hyperparam, hyperparams in hyperparams_tuning.items():
    for hyperparam in hyperparams:
        imputer = diffusions.TabDDPM(dim_input=11, **{name_hyperparam: hyperparam})
        imputer.fit(df_train, batch_size=500, epochs=100, x_valid=df_valid, x_valid_mask=df_valid_mask, print_valid=False, metrics_valid=dict_metrics)
        summaries[f"{name_hyperparam}={hyperparam}"] = imputer.summary
        imputer.cuda_empty_cache()

```

```python
%%time

hyperparams_tuning = {
    'size_window': [10, 30, 60, 180, 200],
}

for name_hyperparam, hyperparams in hyperparams_tuning.items():
    for hyperparam in hyperparams:
        imputer = diffusions.TabDDPMTS(dim_input=11, num_noise_steps=100, dim_embedding=256, **{name_hyperparam: hyperparam})
        imputer.fit(df_train, batch_size=500, epochs=100, x_valid=df_valid, x_valid_mask=df_valid_mask, print_valid=False, metrics_valid=dict_metrics)
        summaries[f"TS {name_hyperparam}={hyperparam}"] = imputer.summary
        imputer.cuda_empty_cache()
```

```python
with open('figures/summaries_tuning.pkl', 'wb') as handle:
    pickle.dump(summaries, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

```python
with open('figures/summaries_tuning.pkl', 'rb') as handle:
    summaries = pickle.load(handle)

plot_summaries(summaries, display='epoch_loss', height=300).show()
plot_summaries(summaries, display='mae', height=300).show()
plot_summaries(summaries, display='wasser', height=300).show()
plot_summaries(summaries, display='num_params', height=300).show()
```

## III. Evaluation


### One-shot training

```python
df_data_st = df_data.loc[['Aotizhongxin']]
df_data_raw_st = df_data_raw.loc[['Aotizhongxin']]
df_mask_st = df_mask.loc[['Aotizhongxin']]

df_data_ft = data.add_datetime_features(df_data)
# df_data_ft = data.add_station_features(df_data_ft)
df_data_ft_st = df_data_ft.loc[['Aotizhongxin']]

df_data_raw_eval = df_data_raw
df_data_eval = df_data
df_mask_eval = df_mask
dim_input = 11

dict_imputers = {}
dfs_imputed = {}
```

```python
dfs_imputed_baseline = {name: imp.fit_transform(df_data_eval) for name, imp in dict_imputers_baseline.items()}
```

```python
%%time

dict_imputers["TabDDPM"] = imputers_pytorch.ImputerGenerativeModelPytorch(groups=['station'], model=diffusions.TabDDPM(dim_input=dim_input, num_noise_steps=500, num_blocks=1, dim_embedding=512), batch_size=500, epochs=500, print_valid=False)
dfs_imputed["TabDDPM"] = dict_imputers["TabDDPM"].fit_transform(df_data_eval)
```

```python
%%time

dict_imputers["TabDDPMTS"] = imputers_pytorch.ImputerGenerativeModelPytorch(groups=['station'], model=diffusions.TabDDPMTS(dim_input=dim_input, num_noise_steps=500, num_blocks=1, size_window=180, dim_embedding=512), batch_size=500, epochs=500, print_valid=False)
dfs_imputed["TabDDPMTS"] = dict_imputers["TabDDPMTS"].fit_transform(df_data_eval)
```

```python
with open('figures/df_data_raw_eval.pkl', 'wb') as handle:
    pickle.dump(df_data_raw_eval, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('figures/df_mask_eval.pkl', 'wb') as handle:
    pickle.dump(df_mask_eval, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('figures/dfs_imputed_baseline.pkl', 'wb') as handle:
    pickle.dump(dfs_imputed_baseline, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('figures/dfs_imputed.pkl', 'wb') as handle:
    pickle.dump(dfs_imputed, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

```python
with open('figures/df_data_raw_eval.pkl', 'rb') as handle:
    df_data_raw_eval = pickle.load(handle)

with open('figures/df_mask_eval.pkl', 'rb') as handle:
    df_mask_eval = pickle.load(handle)

with open('figures/dfs_imputed_baseline.pkl', 'rb') as handle:
    dfs_imputed_baseline = pickle.load(handle)

with open('figures/dfs_imputed.pkl', 'rb') as handle:
    dfs_imputed = pickle.load(handle)
```

```python
dict_metrics = {
    "mae": metrics.mean_absolute_error,
    "wasser": metrics.wasserstein_distance,
    # "kl": metrics.kl_divergence,
    # "corr": metrics.mean_difference_correlation_matrix_numerical_features,
}

df_error = plot_errors(df_data_raw_eval, {**dfs_imputed_baseline, **dfs_imputed}, df_mask_eval, dict_metrics, df_data_raw.columns.to_list()).sort_index()
```

```python
cols_min_value = df_error.idxmin(axis=1).unique().tolist() + list(dict_imputers.keys())

print("Metrics: ", df_error.columns)
df_error.style\
.apply(lambda x: ["background: green" if v == x.min() else "" for v in x], axis = 1)\
# .hide([col for col in df_error.columns.to_list() if col not in cols_min_value], axis=1)\

display(df_error.loc[ ['wasser'], [col for col in df_error.columns.to_list() if col in cols_min_value]]\
.style.apply(lambda x: ["background: green" if v == x.min() else "" for v in x], axis = 1))

display(df_error.loc[ ['mae'], [col for col in df_error.columns.to_list() if col in cols_min_value]]\
.style.apply(lambda x: ["background: green" if v == x.min() else "" for v in x], axis = 1))
```

```python
station = df_data_eval.index.get_level_values("station").unique()[5]
print(station)
df_station = df_data_eval.loc[station]
df_raw_station = df_data_raw_eval.loc[station]
dfs_imputed_station = {name: df_plot.loc[station] for name, df_plot in {**dfs_imputed_baseline, **dfs_imputed}.items()}

for col in cols_to_impute:
    fig = go.Figure()
    df_target = df_raw_station.copy()
    df_target[df_station.notna()] = np.nan

    fig.add_trace(go.Scatter(x=df_station.index, y=df_station[col], mode='markers', name='obs', marker=dict(color='black')))
    fig.add_trace(go.Scatter(x=df_target.index, y=df_target[col], mode='markers', name='true', marker=dict(color='grey')))

    for ind, (name, model) in enumerate(list(dfs_imputed_station.items())):
        values_imp = dfs_imputed_station[name].copy()
        values_imp[df_station.notna()] = np.nan
        fig.add_trace(go.Scatter(x=values_imp.index, y=values_imp[col], mode='markers', name=name))
    fig.update_layout(title=f'{station}: {col}', xaxis_title="datetime", yaxis_title=col, legend_title="Models")
    fig.show()
```

### Noise ratio

```python
%%time
ratios_masked = [0.2, 0.4, 0.6, 0.8]

dfs_imputed_ratio = {}
df_mask_ratio = {}
for ratio_masked in ratios_masked:
    df_data_ = data.add_holes(df_data_raw, ratio_masked=ratio_masked, mean_size=120)
    df_mask_ = df_data_.isna()
    df_mask_[df_data_raw.isna()] = False

    dfs_imputed_ = {name: imp.fit_transform(df_data_) for name, imp in dict_imputers_baseline.items()}

    imputer = imputers_pytorch.ImputerGenerativeModelPytorch(groups=['station'], model=diffusions.TabDDPM(dim_input=11, num_noise_steps=200, num_blocks=1, dim_embedding=512), batch_size=500, epochs=100, print_valid=False)
    dfs_imputed_["TabDDPM"] = imputer.fit_transform(df_data_)
    imputer.model.cuda_empty_cache()

    imputer = imputers_pytorch.ImputerGenerativeModelPytorch(groups=['station'], model=diffusions.TabDDPMTS(dim_input=11, num_noise_steps=200, num_blocks=1, dim_embedding=512), batch_size=500, epochs=100, print_valid=False)
    dfs_imputed_["TabDDPMTS"] = imputer.fit_transform(df_data_)
    imputer.model.cuda_empty_cache()

    dfs_imputed_ratio[ratio_masked] = dfs_imputed_
    df_mask_ratio[ratio_masked] = df_mask_

columns_ratio = list(dfs_imputed_ratio[list(dfs_imputed_ratio.keys())[0]].keys())
df_error_ratio = pd.DataFrame()
scaler = MinMaxScaler()
df_data_raw_scaled = pd.DataFrame(scaler.fit_transform(df_data_raw.values), columns=df_data_raw.columns, index=df_data_raw.index)
for ratio, dfs_imputed_ in dfs_imputed_ratio.items():
    dfs_imputed_scaled = {}
    for method, df_imputed_ in dfs_imputed_.items():
        dfs_imputed_scaled[method] = pd.DataFrame(scaler.transform(df_imputed_.values), columns=df_data_raw.columns, index=df_data_raw.index)

    df_error_ = plot_errors(df_data_raw_scaled, dfs_imputed_scaled, df_mask_ratio[ratio], dict_metrics, df_data_raw.columns.to_list()).sort_index()
    df_error_.columns=pd.MultiIndex.from_tuples([(ratio, col) for col in columns_ratio])
    df_error_ratio = pd.concat([df_error_ratio, df_error_], axis=1)

with open('figures/df_error_ratio.pkl', 'wb') as handle:
    pickle.dump(df_error_ratio, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

```python
with open('figures/df_error_ratio.pkl', 'rb') as handle:
    df_error_ratio = pickle.load(handle)

for mtr in dict_metrics:
    df_plot = df_error_ratio.groupby(level=0).mean().loc[mtr]
    fig = px.line(x=df_plot.index.get_level_values(0), y=df_plot.values, color=df_plot.index.get_level_values(1), markers=True)
    fig.update_layout(title=mtr, xaxis_title="Ratio masked", yaxis_title=mtr, legend_title="Models")
    fig.show()
```

```python

```
