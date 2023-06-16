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

```python
cd ../
```

```python
%reload_ext autoreload
%autoreload 2

# import warnings
# warnings.filterwarnings('error')

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import inspect

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
    "kl": metrics.kl_divergence
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
df_data_raw = data.get_data_corrupted("Beijing", ratio_masked=0., mean_size=120)
df_data = data.get_data_corrupted("Beijing", ratio_masked=.2, mean_size=120)

# cols_to_impute = ["TEMP", "PRES", "DEWP", "NO2", "CO", "O3", "WSPM"]
# cols_to_impute = df_data.columns[df_data.isna().any()]
cols_to_impute = ["TEMP", "PRES"]

n_stations = len(df_data.groupby("station").size())
n_cols = len(cols_to_impute)

df_mask = df_data.isna()
df_mask[df_data_raw.isna()] = False

```

```python
df_data.describe()
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
    # "OU": imputer_ou,
    # "TSOU": imputer_tsou,
    "TSMLE": imputer_tsmle,
    # "RPCA": imputer_rpca,
    "RPCA_opti": imputer_rpca_opti,
    # "locf": imputer_locf,
    # "nocb": imputer_nocb,
    "knn": imputer_knn,
    "mice": imputer_mice,
    # "regressor": imputer_regressor,
}

n_imputers = len(dict_imputers_baseline)
```

## III. Hyperparameter tuning

```python
# station = df_mask.index.get_level_values("station").unique()[:2]
# df_valid = df_data_raw.loc[station].dropna()
# df_valid_mask = df_mask.loc[station].loc[df_valid.index]
# df_train = df_data.loc[station]

# print(f"Train: {len(df_train)}, Valid: {len(df_valid)} ")

# summaries = {}
```

```python
# imputer = diffusions.TabDDPM(dim_input=11, num_noise_steps=50, num_blocks=1, dim_embedding=256)
# imputer.fit(df_train, batch_size=500, epochs=10, x_valid=df_valid, x_valid_mask=df_valid_mask, print_valid=True, metrics_valid=dict_metrics)
# summaries["model"] = imputer.summary
# imputer.cuda_empty_cache()
```

```python
# imputer = diffusions.TabDDPMTS(dim_input=11, num_noise_steps=50, num_blocks=1, size_window=10, dim_embedding=256)
# imputer.fit(df_train, batch_size=1000, epochs=10, x_valid=df_valid, x_valid_mask=df_valid_mask, print_valid=True, metrics_valid=dict_metrics)
# summaries["model_ts"] = imputer.summary
# imputer.cuda_empty_cache()
```

### Hyperparams

```python
# %%time
# name_hyperparam = 'size_window'
# hyperparams = [30, 60, 180]
# for hyperparam in hyperparams:
#     imputer = imputers_pytorch.TabDDPMTS(dim_input=11, num_noise_steps=50, num_blocks=1, size_window=hyperparam, lr=0.001, dim_embedding=256)
#     imputer.fit(df_train, batch_size=500, epochs=50, x_valid=df_valid, x_valid_mask=df_valid_mask, print_valid=True)
#     summaries[f"{name_hyperparam}={hyperparam}"] = imputer.summary
#     imputer.cuda_empty_cache()
```

### Plot

```python
plot_summaries(summaries, display='epoch_loss', height=300).show()
plot_summaries(summaries, display='mae', height=300).show()
plot_summaries(summaries, display='kl', height=300).show()
#plot_summaries(summaries, display='num_params', height=300).show()
```

## III. Evaluation

```python
df_data_st = df_data.loc[['Aotizhongxin']]
df_data_raw_st = df_data_raw.loc[['Aotizhongxin']]
df_mask_st = df_mask.loc[['Aotizhongxin']]

df_data_ft_dt = data.add_datetime_features(df_data)
# df_data_ft_st = data.add_station_features(df_data)

df_data_dt_st = df_data_ft_dt.loc[['Aotizhongxin']]
```

```python
dfs_imputed_baseline = {name: imp.fit_transform(df_data) for name, imp in dict_imputers_baseline.items()}
```

```python
dict_imputers = {}
dfs_imputed = {}
```

```python
%%time

dict_imputers["TabDDPM"] = imputers_pytorch.ImputerGenerativeModelPytorch(groups=['station'], model=diffusions.TabDDPM(dim_input=11, num_noise_steps=200, num_blocks=1, dim_embedding=512), batch_size=500, epochs=500, print_valid=True)
dfs_imputed["TabDDPM"] = dict_imputers["TabDDPM"].fit_transform(df_data)
```

```python
%%time

dict_imputers["TabDDPMTS"] = imputers_pytorch.ImputerGenerativeModelPytorch(groups=['station'], model=diffusions.TabDDPMTS(dim_input=11, num_noise_steps=200, num_blocks=1, size_window=60, dim_embedding=512), batch_size=300, epochs=500, print_valid=True)
dfs_imputed["TabDDPMTS"] = dict_imputers["TabDDPMTS"].fit_transform(df_data)
```

```python
dict_metrics = {
    "mae": metrics.mean_absolute_error,
    "wasser": metrics.wasserstein_distance,
    "kl": metrics.kl_divergence,
    "corr": metrics.mean_difference_correlation_matrix_numerical_features,
}

df_error = plot_errors(df_data_raw, {**dfs_imputed_baseline, **dfs_imputed}, df_mask, dict_metrics, df_data_raw.columns.to_list(), use_p_value=False, method="gaussian").sort_index()
```

```python
cols_min_value = df_error.idxmin(axis=1).unique().tolist() + list(dict_imputers.keys())

print("Metrics: ", df_error.columns)
df_error.style\
.apply(lambda x: ["background: green" if v == x.min() else "" for v in x], axis = 1)\
# .hide([col for col in df_error.columns.to_list() if col not in cols_min_value], axis=1)\

display(df_error.loc[ ['kl', 'wasser'], [col for col in df_error.columns.to_list() if col in cols_min_value]]\
.style.apply(lambda x: ["background: green" if v == x.min() else "" for v in x], axis = 1))

display(df_error.loc[ ['mae'], [col for col in df_error.columns.to_list() if col in cols_min_value]]\
.style.apply(lambda x: ["background: green" if v == x.min() else "" for v in x], axis = 1))

display(df_error.loc[ ['corr'], [col for col in df_error.columns.to_list() if col in cols_min_value]]\
.style.apply(lambda x: ["background: green" if v == x.min() else "" for v in x], axis = 1))
```

```python
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as plticker

tab10 = plt.get_cmap("tab10")
plt.rcParams.update({'font.size': 18})

station = df_data.index.get_level_values("station")[0]
print(station)
df_station = df_data.loc[station]
df_station_raw = df_data_raw.loc[station]
dfs_imputed_station = {name: df_plot.loc[station] for name, df_plot in {**dfs_imputed_baseline, **dfs_imputed}.items()}

for col in cols_to_impute:
    fig, ax = plt.subplots(figsize=(10, 3))
    values_orig_raw = df_station_raw[col]
    values_orig = df_station[col]

    plt.plot(values_orig, ".", color='black', label="obs")
    values_orig_raw[values_orig.notna()] = np.nan
    plt.plot(values_orig_raw, ".", color='yellow', label="true")

    for ind, (name, model) in enumerate(list(dfs_imputed_station.items())):
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
