---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
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

import sys
sys.path.append('/home/ec2-ngo/qolmat/')

# import warnings
# warnings.filterwarnings('error')

import pandas as pd
import numpy as np
import hyperopt as ho
import multiprocessing
import scipy
import plotly.graph_objects as go
import plotly.express as px
import inspect
import pickle

from functools import partial
from sklearn.linear_model import LinearRegression

from qolmat.benchmark import comparator, missing_patterns, metrics
from qolmat.imputations import imputers, imputers_pytorch
from qolmat.imputations.diffusions import diffusions
from qolmat.utils import data, plot
```

## **I. Load data**

```python
df_data_raw = data.get_data("Beijing_offline", datapath='data/')
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
dict_config_opti = {}

imputer_mean = imputers.ImputerMean(groups=("station",))
imputer_median = imputers.ImputerMedian(groups=("station",))
imputer_mode = imputers.ImputerMode(groups=("station",))
imputer_locf = imputers.ImputerLOCF(groups=("station",))
imputer_nocb = imputers.ImputerNOCB(groups=("station",))
imputer_interpol = imputers.ImputerInterpolation(groups=("station",), method="linear")
imputer_spline = imputers.ImputerInterpolation(groups=("station",), method="spline", order=2)
imputer_shuffle = imputers.ImputerShuffle(groups=("station",))
imputer_residuals = imputers.ImputerResiduals(groups=("station",), period=365, model_tsa="additive", extrapolate_trend="freq", method_interpolation="linear")

imputer_rpca = imputers.ImputerRPCA(groups=("station",), columnwise=False, max_iterations=500, tau=2, lam=0.05)
imputer_rpca_opti = imputers.ImputerRPCA(groups=("station",), columnwise=False, max_iterations=256)
dict_config_opti["RPCA_opti"] = {
    "tau": ho.hp.uniform("tau", low=.5, high=5),
    "lam": ho.hp.uniform("lam", low=.1, high=1),
}
imputer_rpca_opticw = imputers.ImputerRPCA(groups=("station",), columnwise=False, max_iterations=256)
dict_config_opti["RPCA_opticw"] = {
    "tau/TEMP": ho.hp.uniform("tau/TEMP", low=.5, high=5),
    "tau/PRES": ho.hp.uniform("tau/PRES", low=.5, high=5),
    "lam/TEMP": ho.hp.uniform("lam/TEMP", low=.1, high=1),
    "lam/PRES": ho.hp.uniform("lam/PRES", low=.1, high=1),
}

imputer_ou = imputers.ImputerEM(groups=("station",), model="multinormal", method="sample", max_iter_em=34, n_iter_ou=15, dt=1e-3)
imputer_tsou = imputers.ImputerEM(groups=("station",), model="VAR", method="sample", max_iter_em=34, n_iter_ou=15, dt=1e-3, p=1)
imputer_tsmle = imputers.ImputerEM(groups=("station",), model="VAR", method="mle", max_iter_em=100, n_iter_ou=15, dt=1e-3, p=1)

imputer_knn = imputers.ImputerKNN(groups=("station",), n_neighbors=10)
imputer_mice = imputers.ImputerMICE(groups=("station",), estimator=LinearRegression(), sample_posterior=False, max_iter=100)
imputer_regressor = imputers.ImputerRegressor(groups=("station",), estimator=LinearRegression())

dict_imputers_baseline = {
    "mean": imputer_mean,
    "median": imputer_median,
    "mode": imputer_mode,
    "interpolation": imputer_interpol,
    "spline": imputer_spline,
    "shuffle": imputer_shuffle,
    "residuals": imputer_residuals,
    "OU": imputer_ou,
    "TSOU": imputer_tsou,
    "TSMLE": imputer_tsmle,
    "locf": imputer_locf,
    "nocb": imputer_nocb,
    "knn": imputer_knn,
    "mice": imputer_mice,
}

n_imputers = len(dict_imputers_baseline)
```

## III. Hyperparameter tuning

```python
# station = df_data_raw.index.get_level_values("station").unique()[0]
# df_valid = df_data_raw.loc[station].dropna()
# df_valid_mask = df_mask.loc[station].loc[df_valid.index]
# df_train = df_data.loc[station]

# df_valid = df_data_raw.dropna()
# df_valid_mask = df_mask.loc[df_valid.index]
# df_train = df_data

# print(f"Train: {len(df_train)}, Valid: {len(df_valid)} ")

# summaries = {}
```

```python
# %%time

# hyperparams_tuning = {
#     'num_noise_steps': [50, 100, 200, 300],
#     'dim_embedding': [128, 256, 512],
#     'num_sampling': [1, 10, 20, 30, 40, 50, 100, 150, 200],
# }

# for name_hyperparam, hyperparams in hyperparams_tuning.items():
#     for hyperparam in hyperparams:
#         imputer = diffusions.TabDDPM(dim_input=11, **{name_hyperparam: hyperparam})
#         imputer.fit(df_train, batch_size=500, epochs=100, x_valid=df_valid, x_valid_mask=df_valid_mask, print_valid=False, metrics_valid=list(dict_metrics.items()))
#         summaries[f"{name_hyperparam}={hyperparam}"] = imputer.summary
#         imputer._cuda_empty_cache()

```

```python
# %%time

# hyperparams_tuning = {
#     'size_window': [10, 30, 60, 180, 200],
# }

# for name_hyperparam, hyperparams in hyperparams_tuning.items():
#     for hyperparam in hyperparams:
#         imputer = diffusions.TabDDPMTS(dim_input=11, num_noise_steps=100, dim_embedding=256, **{name_hyperparam: hyperparam})
#         imputer.fit(df_train, batch_size=500, epochs=100, x_valid=df_valid, x_valid_mask=df_valid_mask, print_valid=False, metrics_valid=list(dict_metrics.items()))
#         summaries[f"TS {name_hyperparam}={hyperparam}"] = imputer.summary
#         imputer._cuda_empty_cache()
```

```python
# with open('figures/summaries_tuning.pkl', 'wb') as handle:
#     pickle.dump(summaries, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

```python
# with open('figures/summaries_tuning.pkl', 'rb') as handle:
#     summaries = pickle.load(handle)

# plot_summaries(summaries, display='epoch_loss', height=300).show()
# plot_summaries(summaries, display='mae', height=300).show()
# plot_summaries(summaries, display='wasser', height=300).show()
# plot_summaries(summaries, display='num_params', height=300).show()
```

## III. Evaluation


### One-shot training

```python
# df_data_st = df_data.loc[['Aotizhongxin']]
# df_data_raw_st = df_data_raw.loc[['Aotizhongxin']]
# df_mask_st = df_mask.loc[['Aotizhongxin']]

# df_data_ft = data.add_datetime_features(df_data)
# df_data_ft = data.add_station_features(df_data_ft)
# df_data_ft_st = df_data_ft.loc[['Aotizhongxin']]

df_data_raw_eval = df_data_raw
df_data_eval = df_data
df_mask_eval = df_mask
dim_input = 11

dict_imputers = {}
dfs_imputed = {}
```

```python
# dfs_imputed_baseline = {name: imp.fit_transform(df_data_eval) for name, imp in dict_imputers_baseline.items()}

manager = multiprocessing.Manager()
dfs_imputed_baseline = manager.dict()

def impute_df(imputer_name, imputer, dict):
    dict[imputer_name] = imputer.fit_transform(df_data_eval)

for name, imp in dict_imputers_baseline.items():
    p = multiprocessing.Process(target=impute_df, args=[name, imp, dfs_imputed_baseline])
    p.start()
    p.join()

```

```python
%%time

dict_imputers["TabDDPM"] = imputers_pytorch.ImputerGenerativeModelPytorch(model=diffusions.TabDDPM(dim_input=dim_input, num_noise_steps=500, num_blocks=1, dim_embedding=512), batch_size=500, epochs=100, print_valid=True)
dict_imputers["TabDDPM"] = dict_imputers["TabDDPM"].fit(df_data_eval)
```

```python
%%time

dict_imputers["TabDDPMTS"] = imputers_pytorch.ImputerGenerativeModelPytorch(model=diffusions.TabDDPMTS(dim_input=dim_input, num_noise_steps=500, num_blocks=1, size_window=180, dim_embedding=512), batch_size=500, epochs=100, print_valid=True)
dict_imputers["TabDDPMTS"] = dict_imputers["TabDDPMTS"].fit(df_data_eval)
```

```python
dict_imputers["TabDDPMTS"].model.set_hyperparams_predict(num_sampling=1, batch_size_predict=800)
dfs_imputed["TabDDPMTS"] = dict_imputers["TabDDPMTS"].transform(df_data_eval)

dict_imputers["TabDDPMTS"].model.set_hyperparams_predict(num_sampling=50, batch_size_predict=800)
dfs_imputed["TabDDPMTS"] = dict_imputers["TabDDPMTS"].transform(df_data_eval)
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
# with open('figures/df_data_raw_eval.pkl', 'rb') as handle:
#     df_data_raw_eval = pickle.load(handle)

# with open('figures/df_mask_eval.pkl', 'rb') as handle:
#     df_mask_eval = pickle.load(handle)

# with open('figures/dfs_imputed_baseline.pkl', 'rb') as handle:
#     dfs_imputed_baseline = pickle.load(handle)

# with open('figures/dfs_imputed.pkl', 'rb') as handle:
#     dfs_imputed = pickle.load(handle)
```

```python
df_error = plot_errors(df_data_raw_eval, {**dfs_imputed_baseline, **dfs_imputed}, df_mask_eval, dict_metrics, df_data_raw.columns.to_list()).sort_index()
```

```python
df_error.loc[:, ['TSOU', 'TSMLE', 'knn', 'mice', 'TabDDPM']]\
.groupby(level=0).mean().transpose()\
.style.apply(lambda x: ["background: green" if v == x.min() else "" for v in x], axis = 0)
```

```python
cols_min_value = df_error.idxmin(axis=1).unique().tolist() + list(dict_imputers.keys())

print("Metrics: ", df_error.columns)
df_error.style\
.apply(lambda x: ["background: green" if v == x.min() else "" for v in x], axis = 1)\
# .hide([col for col in df_error.columns.to_list() if col not in cols_min_value], axis=1)\

display(df_error.loc[ ['wasser', 'KL_gaussian'], [col for col in df_error.columns.to_list() if col in cols_min_value]]\
.style.apply(lambda x: ["background: green" if v == x.min() else "" for v in x], axis = 1))

display(df_error.loc[ ['mae', 'wmape'], [col for col in df_error.columns.to_list() if col in cols_min_value]]\
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

### Evaluation with Comparator

```python
# generator_holes = missing_patterns.EmpiricalHoleGenerator(n_splits=2, groups=["station"], subset=cols_to_impute, ratio_masked=0.2)
# dict_imputers_comparator = {**dict_imputers_baseline,**dict_imputers}

# comparison = comparator.Comparator(
#     dict_imputers_comparator,
#     cols_to_impute,
#     generator_holes = generator_holes,
#     metrics=["mae", "wmape", "wasserstein_columnwise", "KL_gaussian"],
#     max_evals=10,
#     dict_config_opti=dict_config_opti,
# )
# results = comparison.compare(df_data)
# results
```

```python
results = pd.read_pickle('/home/ec2-ngo/qolmat/examples/figures/results_comparator_test.pkl')
```

```python
results.groupby(axis=1, level=0).mean().groupby(axis=0, level=0).mean()
```

```python
metrics = results_mean_cols.columns.unique().to_list()
methods = results_mean_cols.index.get_level_values(level=0).unique().to_list()
```

```python
alternative='less'
metric = 'mae'

matrix_statistic = np.zeros((len(methods), len(methods)))
matrix_pvalue = np.zeros((len(methods), len(methods)))
for idx1, m1 in enumerate(methods):
    for idx2, m2 in enumerate(methods):
        score_m1 = results_1.loc[m1, metric]
        score_m2 = results_1.loc[m2, metric]

        test_result = scipy.stats.ttest_rel(score_m1, score_m2, alternative=alternative)
        matrix_statistic[idx1, idx2] = test_result.statistic
        matrix_pvalue[idx1, idx2] = test_result.pvalue
        # print(m1, m2, test_result.statistic)

df_statistic = pd.DataFrame(matrix_statistic, index=methods, columns=methods)
df_pvalue = pd.DataFrame(matrix_pvalue, index=methods, columns=methods)

metrics_plot = ['mice', 'TabDDPM', 'TabDDPM_sampling']
print('Paired t-test for MAE')
print('''The alternative hypothesis: the mean of the distribution underlying the imputer in index is less than the
mean of the distribution underlying the imputer in column.''')
print('Table of statistic')
display(df_statistic.loc[metrics_plot, metrics_plot])
print('Table of pvalue')
display(df_pvalue.loc[metrics_plot, metrics_plot])
```

```python
alternative='less'
metric = 'wasserstein_columnwise'

matrix_statistic = np.zeros((len(methods), len(methods)))
matrix_pvalue = np.zeros((len(methods), len(methods)))
for idx1, m1 in enumerate(methods):
    for idx2, m2 in enumerate(methods):
        score_m1 = results_1.loc[m1, metric]
        score_m2 = results_1.loc[m2, metric]

        test_result = scipy.stats.ttest_rel(score_m1, score_m2, alternative=alternative)
        matrix_statistic[idx1, idx2] = test_result.statistic
        matrix_pvalue[idx1, idx2] = test_result.pvalue
        # print(m1, m2, test_result.statistic)

df_statistic = pd.DataFrame(matrix_statistic, index=methods, columns=methods)
df_pvalue = pd.DataFrame(matrix_pvalue, index=methods, columns=methods)

metrics_plot = ['mice', 'TabDDPM', 'TabDDPM_sampling']
print('Paired t-test for Wasserstein distance')
print('''The alternative hypothesis: the mean of the distribution underlying the imputer in index is less than the
mean of the distribution underlying the imputer in column.''')
print('Table of statistic')
display(df_statistic.loc[metrics_plot, metrics_plot])
print('Table of pvalue')
display(df_pvalue.loc[metrics_plot, metrics_plot])
```

```python
fig = px.imshow(matrix_score, text_auto=False, x=methods, y=methods)
fig.show()
```

```python
pd.options.display.float_format = '{:.2f}'.format
results[['TSOU', 'TSMLE', 'knn', 'mice', 'TabDDPM', 'TabDDPM_sampling']].groupby(level=0).mean()
```

### Noise ratio

```python
%%time

ratios_masked = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

dfs_imputed_ratio = {}
df_mask_ratio = {}
for ratio_masked in ratios_masked:
    df_data_ = data.add_holes(df_data_raw, ratio_masked=ratio_masked, mean_size=120)
    df_mask_ = df_data_.isna()
    df_mask_[df_data_raw.isna()] = False

    dfs_imputed_ = {name: imp.fit_transform(df_data_) for name, imp in dict_imputers_baseline.items()}

    imputer = imputers_pytorch.ImputerGenerativeModelPytorch(groups=['station'], model=diffusions.TabDDPM(dim_input=11, num_noise_steps=200, num_blocks=1, dim_embedding=512), batch_size=500, epochs=100, print_valid=False)
    dfs_imputed_["TabDDPM"] = imputer.fit_transform(df_data_)
    imputer.model._cuda_empty_cache()

    imputer = imputers_pytorch.ImputerGenerativeModelPytorch(groups=['station'], model=diffusions.TabDDPMTS(dim_input=11, num_noise_steps=200, num_blocks=1, dim_embedding=512), batch_size=500, epochs=100, print_valid=False)
    dfs_imputed_["TabDDPMTS"] = imputer.fit_transform(df_data_)
    imputer.model._cuda_empty_cache()

    dfs_imputed_ratio[ratio_masked] = dfs_imputed_
    df_mask_ratio[ratio_masked] = df_mask_

columns_ratio = list(dfs_imputed_ratio[list(dfs_imputed_ratio.keys())[0]].keys())
df_error_ratio = pd.DataFrame()
for ratio, dfs_imputed_ in dfs_imputed_ratio.items():
    df_error_ = plot_errors(df_data_raw, dfs_imputed_, df_mask_ratio[ratio], dict_metrics, df_data_raw.columns.to_list()).sort_index()
    df_error_.columns=pd.MultiIndex.from_tuples([(ratio, col) for col in columns_ratio])
    df_error_ratio = pd.concat([df_error_ratio, df_error_], axis=1)

with open('figures/dfs_imputed_ratio.pkl', 'wb') as handle:
    pickle.dump(dfs_imputed_ratio, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('figures/df_mask_ratio.pkl', 'wb') as handle:
    pickle.dump(df_mask_ratio, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('figures/df_error_ratio.pkl', 'wb') as handle:
    pickle.dump(df_error_ratio, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

```python
df_error_ratio = pd.read_pickle('figures/df_error_ratio.pkl')

for mtr in dict_metrics:
    df_plot = df_error_ratio.groupby(level=0).mean().loc[mtr]
    fig = px.line(x=df_plot.index.get_level_values(0), y=df_plot.values, color=df_plot.index.get_level_values(1), markers=True)
    fig.update_layout(title=mtr, xaxis_title="Ratio masked", yaxis_title=mtr, legend_title="Models")
    fig.show()
```
