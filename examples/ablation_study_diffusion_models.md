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

# Ablation study for diffusion models

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
from tqdm import tqdm

from functools import partial
from sklearn.linear_model import LinearRegression

from qolmat.benchmark import comparator, missing_patterns
from qolmat.imputations import imputers, imputers_pytorch
from qolmat.imputations.diffusions import ddpms
from qolmat.utils import data
```

## **I. Load data**

```python
df_data_raw = data.get_data("Beijing_offline", datapath='/home/ec2-ngo/qolmat/examples/data')

cols_to_impute = df_data_raw.columns
n_stations = len(df_data_raw.groupby("station").size())
n_cols = len(cols_to_impute)

display(df_data_raw.describe())
display(df_data_raw.isna().sum())
```

```python
# df_data = data.add_holes(df_data_raw, ratio_masked=.1, mean_size=120)

# df_mask = df_data.isna()
# df_mask[df_data_raw.isna()] = False
```

```python
# with open('/home/ec2-ngo/qolmat/examples/data/benchmark/ablation_df_data.pkl', 'wb') as handle:
#     pickle.dump(df_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('/home/ec2-ngo/qolmat/examples/data/benchmark/ablation_df_mask.pkl', 'wb') as handle:
#     pickle.dump(df_mask, handle, protocol=pickle.HIGHEST_PROTOCOL)

df_data = pd.read_pickle('/home/ec2-ngo/qolmat/examples/data/benchmark/ablation_df_data.pkl')
df_mask = pd.read_pickle('/home/ec2-ngo/qolmat/examples/data/benchmark/ablation_df_mask.pkl')
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

## III. Hyperparams

```python
def plot_summaries(summaries, display='epoch_loss', title='', xaxis_title='epoch', height=300, type="log"):
    fig = go.Figure()

    for ind, (name, values) in enumerate(list(summaries.items())):
        values_selected = values[display]
        fig.add_trace(go.Scatter(x=list(range(len(values_selected))), y=values_selected, mode='lines+markers', name=name))

    fig.update_layout(title=title,
                      xaxis_title=xaxis_title,
                      yaxis_title=display, height=height)

    fig.update_yaxes(type=type)

    return fig
```

### num_noise_steps

```python
# summaries_noise_steps = {}
# for num_noise_steps in tqdm([10, 50, 100, 200, 300, 400, 500]):
#     imputer_TabDDPM = imputers_pytorch.ImputerDiffusion(model=ddpms.TabDDPM(num_noise_steps=num_noise_steps, num_blocks=1, dim_embedding=512, num_sampling=50), batch_size=15000, epochs=50, print_valid=True, x_valid=df_data_raw)
#     imputer_TabDDPM = imputer_TabDDPM.fit(df_data)
#     summaries_noise_steps[f"TabDDPM_ns={num_noise_steps}"] = imputer_TabDDPM.model.summary

# with open('/home/ec2-ngo/qolmat/examples/data/benchmark/ablation_noise_steps.pkl', 'wb') as handle:
#     pickle.dump(summaries_noise_steps, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

```python
summaries_noise_steps = pd.read_pickle('/home/ec2-ngo/qolmat/examples/data/benchmark/ablation_noise_steps.pkl')
```

```python
display(plot_summaries(summaries_noise_steps, display='mean_absolute_error', title= 'Tuning num_noise_steps'))
display(plot_summaries(summaries_noise_steps, display='dist_wasserstein', title= 'Tuning num_noise_steps'))
```

### num_sampling

```python
# summaries_num_sampling = {}
# for num_sampling in tqdm([1, 5, 10, 15, 20, 40, 60, 80, 100]):
#     imputer_TabDDPM = imputers_pytorch.ImputerDiffusion(model=ddpms.TabDDPM(num_noise_steps=100, num_blocks=1, dim_embedding=512, num_sampling=num_sampling), batch_size=15000, epochs=50, print_valid=True, x_valid=df_data_raw)
#     imputer_TabDDPM = imputer_TabDDPM.fit(df_data)
#     summaries_num_sampling[f"TabDDPM_sampling={num_sampling}"] = imputer_TabDDPM.model.summary

# with open('/home/ec2-ngo/qolmat/examples/data/benchmark/ablation_num_sampling.pkl', 'wb') as handle:
#     pickle.dump(summaries_num_sampling, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

```python
summaries_num_sampling = pd.read_pickle('/home/ec2-ngo/qolmat/examples/data/benchmark/ablation_num_sampling.pkl')
```

```python
display(plot_summaries(summaries_num_sampling, display='mean_absolute_error', title= 'Tuning num_noise_steps'))
display(plot_summaries(summaries_num_sampling, display='dist_wasserstein', title= 'Tuning num_noise_steps'))
```

### freq_str

```python
# dict_batch_size = {
#     '1D':500,
#     '15D':50,
#     '1M':20,
#     '2M':10,
#     '6M':5,
#     '1Y':2
# }

# summaries_freq_str = {}
# for freq_str in tqdm(['1D', '15D', '1M', '2M', '6M', '1Y']):
#     imputer_TsDDPM = imputers_pytorch.ImputerDiffusion(model=ddpms.TsDDPM(num_noise_steps=100, num_blocks=1, dim_embedding=512, num_sampling=10), batch_size=dict_batch_size[freq_str], epochs=50, print_valid=False, index_datetime='datetime', freq_str=freq_str, x_valid=df_data_raw)
#     imputer_TsDDPM = imputer_TsDDPM.fit(df_data)
#     summaries_freq_str[f"TabDDPM_freq_str={freq_str}"] = imputer_TsDDPM.model.summary
#     imputer_TsDDPM.model._cuda_empty_cache()

# with open('/home/ec2-ngo/qolmat/examples/data/benchmark/ablation_freq_str.pkl', 'wb') as handle:
#     pickle.dump(summaries_freq_str, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

```python
summaries_freq_str = pd.read_pickle('/home/ec2-ngo/qolmat/examples/data/benchmark/ablation_freq_str.pkl')
```

```python
display(plot_summaries(summaries_freq_str, display='mean_absolute_error', title= 'Tuning freq_str'))
display(plot_summaries(summaries_freq_str, display='dist_wasserstein', title= 'Tuning freq_str'))
```

## IV. Evaluation


### Duration

```python
# import time

# dict_imputers = {
#     'TabDDPM': imputers_pytorch.ImputerDiffusion(model=ddpms.TabDDPM(dim_input=dim_input), print_valid=False)
# }
# dict_duration = {}
# for name, imputer in {**dict_imputers_baseline, **dict_imputers}.items():
#     print(name)
#     start_time = time.time()
#     df_imputed = imputer.fit_transform(df_data)
#     dict_duration[name] = time.time() - start_time

# dict_duration_noise_steps_fit = {}
# dict_duration_noise_steps_transform = {}
# for num_noise_steps in tqdm([10, 50, 100, 200, 300, 400, 500]):
#     imputer_TabDDPM = imputers_pytorch.ImputerDiffusion(model=ddpms.TabDDPM(num_noise_steps=num_noise_steps, num_blocks=1, dim_embedding=512, num_sampling=1), batch_size=10000, epochs=1, print_valid=False)
#     start_time = time.time()
#     imputer_TabDDPM = imputer_TabDDPM.fit(df_data)
#     dict_duration_noise_steps_fit[f'TabDDPM_ns={num_noise_steps}'] = time.time() - start_time
#     start_time = time.time()
#     df_imputed = imputer_TabDDPM.transform(df_data)
#     dict_duration_noise_steps_transform[f'TabDDPM_ns={num_noise_steps}'] = time.time() - start_time

```

### Simple comparison

```python
from typing import Any, Dict
from qolmat.benchmark import hyperparameters

class Comparator_(comparator.Comparator):
    def evaluate_errors_sample(
        self,
        imputer: Any,
        df: pd.DataFrame,
        dict_config_opti_imputer: Dict[str, Any] = {},
        metric_optim: str = "mse",
    ) -> pd.Series:
        list_errors = []
        df_origin = df[self.selected_columns].copy()
        for df_mask in self.generator_holes.split(df_origin):
            df_corrupted = df_origin.copy()
            df_corrupted[df_mask] = np.nan
            imputer_opti = hyperparameters.optimize(
                imputer,
                df,
                self.generator_holes,
                metric_optim,
                dict_config_opti_imputer,
                max_evals=self.max_evals,
                verbose=self.verbose,
            )
            df_imputed = imputer_opti.fit_transform(df_corrupted)
            subset = self.generator_holes.subset
            errors = self.get_errors(df_origin[subset], df_imputed[subset], df_mask[subset])
            list_errors.append(errors)
        df_errors = pd.DataFrame(list_errors)
        return df_errors

    def compare(
            self,
            df: pd.DataFrame,
            file_directory: str
        ):
            dict_errors = {}

            for name, imputer in self.dict_imputers.items():
                dict_config_opti_imputer = self.dict_config_opti.get(name, {})

                try:
                    dict_errors[name] = self.evaluate_errors_sample(
                        imputer, df, dict_config_opti_imputer, self.metric_optim
                    )
                    print(f"Tested model: {type(imputer).__name__}")
                except Exception as excp:
                    print("Error while testing ", type(imputer).__name__)
                    raise excp

                df_errors = pd.concat(dict_errors.values(), join='inner', keys=dict_errors.keys(), axis=0)
                with open(file_directory, 'wb') as handle:
                    pickle.dump(df_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return df_errors
```

```python
# dict_imputers = {}

# dict_imputers['TabDDPM_sampling=1'] = imputers_pytorch.ImputerDiffusion(model=ddpms.TabDDPM(num_noise_steps=100, num_blocks=1, dim_embedding=512, num_sampling=1), batch_size=15000, epochs=100, print_valid=False)

# dict_imputers['TabDDPM_sampling=50'] = imputers_pytorch.ImputerDiffusion(model=ddpms.TabDDPM(num_noise_steps=100, num_blocks=1, dim_embedding=512, num_sampling=50), batch_size=15000, epochs=100, print_valid=False)

# dict_imputers['TsDDPM_sampling=1'] = imputers_pytorch.ImputerDiffusion(model=ddpms.TsDDPM(num_noise_steps=100, num_blocks=1, dim_embedding=512, num_sampling=1, is_rolling=True), batch_size=500, epochs=100, print_valid=False, index_datetime='datetime', freq_str="180D")

# dict_imputers['TsDDPM_sampling=50'] = imputers_pytorch.ImputerDiffusion(model=ddpms.TsDDPM(num_noise_steps=100, num_blocks=1, dim_embedding=512, num_sampling=50, is_rolling=True), batch_size=500, epochs=100, print_valid=False, index_datetime='datetime', freq_str="180D")

# dict_imputers['TabDDPM_sampling=1'].model._set_hyperparams_predict(batch_size_predict=100000)
# dict_imputers['TabDDPM_sampling=50'].model._set_hyperparams_predict(batch_size_predict=100000)
# dict_imputers['TsDDPM_sampling=1'].model._set_hyperparams_predict(batch_size_predict=100000)
# dict_imputers['TsDDPM_sampling=50'].model._set_hyperparams_predict(batch_size_predict=100000)

# generator_holes = missing_patterns.UniformHoleGenerator(n_splits=50, ratio_masked=0.2)
# dict_imputers_comparator = {**dict_imputers_baseline,**dict_imputers}

# comparison = Comparator_(
#     dict_imputers_comparator,
#     cols_to_impute,
#     generator_holes = generator_holes,
#     metrics=["mae", "wasserstein_columnwise"],
#     max_evals=5,
#     dict_config_opti=dict_config_opti,
# )
# results = comparison.compare(df_data_raw, '/home/ec2-ngo/qolmat/examples/data/benchmark/ablation_comparison.pkl')
```

```python
results = pd.read_pickle('/home/ec2-ngo/qolmat/examples/data/benchmark/ablation_comparison.pkl')

results.groupby(axis=0, level=0).mean().groupby(axis=1, level=0).mean()
```

```python
results_mean_cols = results.groupby(axis=1, level=0).mean()

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
        score_m1 = results_mean_cols.loc[m1, metric]
        score_m2 = results_mean_cols.loc[m2, metric]

        test_result = scipy.stats.ttest_rel(score_m1, score_m2, alternative=alternative)
        matrix_statistic[idx1, idx2] = test_result.statistic
        matrix_pvalue[idx1, idx2] = test_result.pvalue
        # print(m1, m2, test_result.statistic)

df_statistic = pd.DataFrame(matrix_statistic, index=methods, columns=methods)
df_pvalue = pd.DataFrame(matrix_pvalue, index=methods, columns=methods)

metrics_plot = ['TabDDPM_sampling=1', 'TabDDPM_sampling=50']
print('Paired t-test for MAE')
print('''The alternative hypothesis: the mean of the distribution underlying the imputer in index is less than the
mean of the distribution underlying the imputer in column.''')
print('Table of statistic')
display(df_statistic.loc[metrics_plot, :])
print('Table of pvalue')
display(df_pvalue.loc[metrics_plot, :])
```

```python
alternative='less'
metric = 'wasserstein_columnwise'

matrix_statistic = np.zeros((len(methods), len(methods)))
matrix_pvalue = np.zeros((len(methods), len(methods)))
for idx1, m1 in enumerate(methods):
    for idx2, m2 in enumerate(methods):
        score_m1 = results_mean_cols.loc[m1, metric]
        score_m2 = results_mean_cols.loc[m2, metric]

        test_result = scipy.stats.ttest_rel(score_m1, score_m2, alternative=alternative)
        matrix_statistic[idx1, idx2] = test_result.statistic
        matrix_pvalue[idx1, idx2] = test_result.pvalue
        # print(m1, m2, test_result.statistic)

df_statistic = pd.DataFrame(matrix_statistic, index=methods, columns=methods)
df_pvalue = pd.DataFrame(matrix_pvalue, index=methods, columns=methods)

metrics_plot = ['TabDDPM_sampling=1', 'TabDDPM_sampling=50']
print('Paired t-test for Wasserstein distance')
print('''The alternative hypothesis: the mean of the distribution underlying the imputer in index is less than the
mean of the distribution underlying the imputer in column.''')
print('Table of statistic')
display(df_statistic.loc[metrics_plot, :])
print('Table of pvalue')
display(df_pvalue.loc[metrics_plot, :])
```

### Confidence Interval

```python
def get_confidence_interval(dfs_imputed, df_ref, df_mask, size, confidence=0.95):
    dof = size - 1

    dfs_imputed_concat = pd.concat(dfs_imputed, keys=[f'{i}' for i in range(size)])

    # t_crit = np.abs(t.ppf((1-confidence)/2,dof))

    # dfs_imputed_max = dfs_imputed_concat.groupby(level=[1,2]).max()
    # dfs_imputed_min = dfs_imputed_concat.groupby(level=[1,2]).min()
    # # dfs_imputed_gap = dfs_imputed_max - dfs_imputed_min

    # dfs_imputed_mean = dfs_imputed_concat.groupby(level=[1,2]).mean()

    # dfs_imputed_mean_concat = pd.concat([dfs_imputed_mean for i in range(size)], keys=[f'{i}' for i in range(50)])
    # dfs_imputed_var = (((dfs_imputed_concat - dfs_imputed_mean_concat).pow(2.))/dof).groupby(level=[1,2]).sum()
    # dfs_imputed_std = dfs_imputed_var.pow(1./2)

    # dfs_imputed_lb = dfs_imputed_mean - (t_crit*dfs_imputed_std/np.sqrt(size))
    # dfs_imputed_ub = dfs_imputed_mean + (t_crit*dfs_imputed_std/np.sqrt(size))

    dfs_imputed_lb = dfs_imputed_concat.groupby(level=[1,2]).quantile(q=(1-confidence)/2)
    dfs_imputed_ub = dfs_imputed_concat.groupby(level=[1,2]).quantile(q=confidence + (1-confidence)/2)

    dfs_imputed_lb_True = dfs_imputed_lb <= df_ref
    dfs_imputed_ub_True = dfs_imputed_ub >= df_ref

    dfs_imputed_correct = dfs_imputed_lb_True * dfs_imputed_ub_True

    dfs_imputed_correct_mask = dfs_imputed_correct.replace(True, 1)
    dfs_imputed_correct_mask = dfs_imputed_correct_mask.replace(False, 0)
    dfs_imputed_correct_mask = dfs_imputed_correct_mask.where(df_mask, 0)

    coverage = dfs_imputed_correct_mask.sum().sum()/df_mask.sum().sum()

    return coverage, dfs_imputed_lb, dfs_imputed_ub
```

```python
# imputer_TabDDPM = imputers_pytorch.ImputerDiffusion(model=ddpms.TabDDPM(num_noise_steps=500, num_blocks=1, dim_embedding=512, num_sampling=1), batch_size=100, epochs=1, print_valid=False)

# imputer_TsDDPM = imputers_pytorch.ImputerDiffusion(model=ddpms.TsDDPM(num_noise_steps=500, num_blocks=1, dim_embedding=512, num_sampling=1), batch_size=20, epochs=1, print_valid=False, index_datetime='datetime', freq_str="30D")

# imputer_TabDDPM = imputer_TabDDPM.fit(df_data)
# imputer_TsDDPM = imputer_TsDDPM.fit(df_data)

# dfs_imputed_sampling = []
# for i in tqdm(range(10)):
#     dfs_imputed_sampling.append(imputer_TabDDPM.transform(df_data))
```

Coverage and Mean width in terms of confidence

```python
# list_ci_conf = []
# size = 50
# for conf in np.arange(0., 1.0, 0.05):
#     dfs_imputed_sampling_ = dfs_imputed_sampling[:size]
#     coverage, dfs_lb, dfs_ub = get_confidence_interval(dfs_imputed_sampling_, df_data_raw, df_mask, size, confidence=conf)
#     width = (dfs_ub - dfs_lb).mean().mean()
#     list_ci_conf.append({'sample num': size, 'confidence': conf, 'coverage':coverage, 'mean width (upper_bound - lower_bound)': width})

# df_ci_conf = pd.DataFrame(list_ci_conf)
# with open('/home/ec2-ngo/qolmat/examples/data/benchmark/ablation_CI_confidence.pkl', 'wb') as handle:
#     pickle.dump(df_ci_conf, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

Coverage and Mean width in terms of number of sampling

```python
# list_ci_size = []
# conf = 0.95
# for size in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
#     dfs_imputed_sampling_ = dfs_imputed_sampling[:size]
#     coverage, dfs_lb, dfs_ub = get_confidence_interval(dfs_imputed_sampling_, df_data_raw, df_mask, size, confidence=conf)
#     width = (dfs_ub - dfs_lb).mean().mean()
#     list_ci_size.append({'sample num': size, 'confidence': conf, 'coverage':coverage, 'mean width (upper_bound - lower_bound)': width})

# df_ci_size = pd.DataFrame(list_ci_size)
# with open('/home/ec2-ngo/qolmat/examples/data/benchmark/ablation_CI_size.pkl', 'wb') as handle:
#     pickle.dump(df_ci_size, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

```python
# size = 100
# confidence = 0.95
# coverage, dfs_plot_lb, dfs_plot_ub = get_confidence_interval(dfs_imputed_sampling[:size], df_data_raw, df_mask, size=size, confidence=confidence)

# station = df_data_raw_eval.index.get_level_values(0).unique()[2]
# col = 'TEMP'

# df_mask_eval_plot = df_mask_eval.loc[station][col]

# df_data_raw_eval_plot = df_data_raw.loc[station][col]
# # df_data_raw_eval_plot[df_mask_eval_plot] = pd.NA

# dfs_plot_lb_plot = dfs_plot_lb.loc[station][col]
# dfs_plot_lb_plot[~df_mask_eval_plot] = pd.NA

# dfs_plot_ub_plot = dfs_plot_ub.loc[station][col]
# dfs_plot_ub_plot[~df_mask_eval_plot] = pd.NA

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=df_data_raw_eval_plot.index, y=df_data_raw_eval_plot, mode='lines', name='Ref'))
# fig.add_trace(go.Scatter(x=dfs_plot_lb_plot.index, y=dfs_plot_lb_plot, mode='lines', name='lb'))
# fig.add_trace(go.Scatter(x=dfs_plot_ub_plot.index, y=dfs_plot_ub_plot, mode='lines', name='ub'))
# fig.update_layout(title=f'Sample num= {size}, confidence={confidence}')
# fig.show()
```

### num_sampling and prediction

```python
class ComparatorPrediction(comparator.Comparator):
    def evaluate_errors_sample(
        self,
        imputer: Any,
        df: pd.DataFrame,
        df_valid: pd.DataFrame = None,
        dict_config_opti_imputer: Dict[str, Any] = {},
        metric_optim: str = "mse",
    ) -> pd.Series:
        list_errors_reconstruction = []
        list_errors_prediction = []
        self.columns_without_nans = df.columns[df.notna().all()]
        df_origin = df[self.selected_columns].copy()
        for df_mask in self.generator_holes.split(df_origin):
            df_corrupted = df_origin.copy()
            df_corrupted[df_mask] = np.nan
            imputer_opti = hyperparameters.optimize(
                imputer,
                df,
                self.generator_holes,
                metric_optim,
                dict_config_opti_imputer,
                max_evals=self.max_evals,
                verbose=self.verbose,
            )
            df_imputed = imputer_opti.fit_transform(df_corrupted)
            subset = self.generator_holes.subset
            errors_reconstruction = self.get_errors(df_origin[subset], df_imputed[subset], df_mask[subset])
            list_errors_reconstruction.append(errors_reconstruction)

            errors_prediction = self.get_errors_prediction(df_imputed, df_valid)
            list_errors_prediction.append(errors_prediction)

        df_errors_reconstruction = pd.DataFrame(list_errors_reconstruction)
        df_errors_prediction = pd.DataFrame(list_errors_prediction)
        return df_errors_reconstruction, df_errors_prediction

    def get_errors_prediction(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.Series:
        dict_errors = []
        for col_target in self.columns_without_nans:
            cols_obs = [col for col in df_train.columns if col is not col_target]
            df_train_X = df_train[cols_obs]
            df_train_y = df_train[col_target]
            df_test_X = df_test[cols_obs]
            df_test_y = df_test[[col_target]]

            reg = LinearRegression().fit(df_train_X, df_train_y)

            df_test_y_hat = pd.DataFrame(reg.predict(df_test_X), columns=[col_target], index=df_test_X.index)

            shape = df_test_y_hat.shape
            df_test_y_mask = pd.DataFrame([[True] * shape[1] for _ in range(shape[0])], columns=[col_target], index=df_test_X.index)
            dict_errors.append(self.get_errors(df_test_y, df_test_y_hat, df_test_y_mask))
        errors = pd.concat(dict_errors).sort_index(level=0)
        return errors

    def compare(
            self,
            df: pd.DataFrame,
            directory: str,
            df_valid: pd.DataFrame = None
        ):
            dict_errors_reconstruction = {}
            dict_errors_prediction = {}

            for name, imputer in self.dict_imputers.items():
                dict_config_opti_imputer = self.dict_config_opti.get(name, {})

                try:
                    dict_errors_reconstruction[name], dict_errors_prediction[name] = self.evaluate_errors_sample(
                        imputer, df, df_valid, dict_config_opti_imputer, self.metric_optim
                    )
                    print(f"Tested model: {type(imputer).__name__}")
                except Exception as excp:
                    print("Error while testing ", type(imputer).__name__)
                    raise excp

                df_errors_reconstruction = pd.concat(dict_errors_reconstruction.values(), join='inner', keys=dict_errors_reconstruction.keys(), axis=0)
                df_errors_prediction = pd.concat(dict_errors_prediction.values(), join='inner', keys=dict_errors_prediction.keys(), axis=0)

                with open(directory + 'ablation_num_sampling_reconstruction.pkl', 'wb') as handle:
                    pickle.dump(df_errors_reconstruction, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open(directory + 'ablation_num_sampling_prediction.pkl', 'wb') as handle:
                    pickle.dump(df_errors_prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return df_errors_reconstruction, df_errors_prediction
```

```python
# dict_imputers = {}
# for num_sampling in tqdm([1, 5, 10, 15, 20, 40, 60, 80, 100]):
#     dict_imputers[f'TabDDPM_sampling={num_sampling}'] = imputers_pytorch.ImputerDiffusion(model=ddpms.TabDDPM(num_noise_steps=100, num_blocks=1, dim_embedding=512, num_sampling=num_sampling), batch_size=15000, epochs=50, print_valid=False)
```

```python
# generator_holes = missing_patterns.UniformHoleGenerator(n_splits=10, ratio_masked=0.2)
# dict_imputers_comparator = {**dict_imputers_baseline}#, **dict_imputers}

# df_data_valid = df_data_raw.dropna().sample(n=7000)
# df_data_train = df_data_raw.loc[~df_data_raw.index.isin(df_data_valid.index)]

# comparison = ComparatorPrediction(
#     dict_imputers_comparator,
#     cols_to_impute,
#     generator_holes = generator_holes,
#     metrics=["mae", "wasserstein_columnwise"],
#     max_evals=5,
#     dict_config_opti=dict_config_opti,
# )
# results_rec, results_pred = comparison.compare(df_data_train, df_valid=df_data_valid, directory='/home/ec2-ngo/qolmat/examples/data/benchmark/')

```

```python
results_rec = pd.read_pickle('/home/ec2-ngo/qolmat/examples/data/benchmark/ablation_num_sampling_reconstruction.pkl')
results_pred = pd.read_pickle('/home/ec2-ngo/qolmat/examples/data/benchmark/ablation_num_sampling_prediction.pkl')
```

```python
list_imputers = []
for num_sampling in tqdm([1, 5, 10, 15, 20, 40, 60, 80, 100]):
    list_imputers.append(f'TabDDPM_sampling={num_sampling}')

list_imputers = list_imputers + list(dict_imputers_baseline.keys())
```

```python
results_pred.groupby(axis=0, level=0).mean().groupby(axis=1, level=0).mean().loc[list_imputers]
```

```python
results_pred_plot = results_pred.groupby(axis=0, level=0).mean().groupby(axis=1, level=0).mean().loc[list_imputers]
results_rec_plot = results_rec.groupby(axis=0, level=0).mean().groupby(axis=1, level=0).mean().loc[list_imputers]

fig = go.Figure()
fig.add_trace(go.Scatter(x=results_pred_plot.index, y=results_pred_plot['mae'], mode='lines+markers', name='mae pred'))
# fig.add_trace(go.Scatter(x=results_pred_plot.index, y=results_pred_plot['wasserstein_columnwise'], mode='lines+markers', name='wass pred'))
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=results_rec_plot.index, y=results_rec_plot['mae'], mode='lines+markers', name='mae rec'))
fig.add_trace(go.Scatter(x=results_rec_plot.index, y=results_rec_plot['wasserstein_columnwise'], mode='lines+markers', name='wass rec'))
fig.show()
```

## TabDDPM vs Hyperimpute, Pypots

```python
from typing import Any, Dict
from qolmat.benchmark import hyperparameters

class Comparator_(comparator.Comparator):
    def evaluate_errors_sample(
        self,
        imputer: Any,
        df: pd.DataFrame,
        dict_config_opti_imputer: Dict[str, Any] = {},
        metric_optim: str = "mse",
    ) -> pd.Series:
        list_errors = []
        df_origin = df[self.selected_columns].copy()
        for df_mask in self.generator_holes.split(df_origin):
            df_corrupted = df_origin.copy()
            df_corrupted[df_mask] = np.nan
            imputer_opti = hyperparameters.optimize(
                imputer,
                df,
                self.generator_holes,
                metric_optim,
                dict_config_opti_imputer,
                max_evals=self.max_evals,
                verbose=self.verbose,
            )
            df_imputed = imputer_opti.fit_transform(df_corrupted)
            df_imputed.columns = df_corrupted.columns
            df_imputed.index = df_corrupted.index
            subset = self.generator_holes.subset
            errors = self.get_errors(df_origin[subset], df_imputed[subset], df_mask[subset])
            list_errors.append(errors)
        df_errors = pd.DataFrame(list_errors)
        return df_errors

    def compare(
            self,
            df: pd.DataFrame,
            file_directory: str
        ):
            dict_errors = {}

            for name, imputer in self.dict_imputers.items():
                dict_config_opti_imputer = self.dict_config_opti.get(name, {})

                try:
                    dict_errors[name] = self.evaluate_errors_sample(
                        imputer, df, dict_config_opti_imputer, self.metric_optim
                    )
                    print(f"Tested model: {type(imputer).__name__}")
                except Exception as excp:
                    print("Error while testing ", type(imputer).__name__)
                    raise excp
                df_errors = pd.concat(dict_errors.values(), join='inner', keys=dict_errors.keys(), axis=0)

                with open(file_directory, 'wb') as handle:
                    pickle.dump(df_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return df_errors
```

```python
import pandas as pd
from pypots.imputation import SAITS, Transformer, BRITS, MRNN
from qolmat.imputations.imputers import _Imputer
from typing import Optional, List
from sklearn import preprocessing

class PypotsWrapper(_Imputer):
    def __init__(
        self,
        n_features: int,
        n_layers: int,
        d_model: int,
        d_inner: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        batch_size: int = 32,
        epochs: int = 100,
        model_name: str = 'SAITS'
    ):
        self.model_name=model_name
        self.n_features=n_features
        self.n_layers=n_layers
        self.d_model=d_model
        self.d_inner=d_inner
        self.n_heads=n_heads
        self.d_k=d_k
        self.d_v=d_v
        self.batch_size=batch_size
        self.epochs=epochs

    def process_data(self, x: pd.DataFrame, freq_str: str, index_datetime: str):
        normalizer_x = preprocessing.StandardScaler()
        normalizer_x = normalizer_x.fit(x.values)

        x_windows: List = []
        x_windows_indices: List = []
        columns_index = [col for col in x.index.names if col != index_datetime]
        columns_index_ = columns_index[0] if len(columns_index) == 1 else columns_index
        for x_group in tqdm(x.groupby(by=columns_index_), disable=True, leave=False):
            for x_w in x_group[1].resample(rule=freq_str, level=index_datetime):
                x_windows.append(x_w[1])
                x_windows_indices.append(x_w[1].index)

        x_windows_processed = []
        size_window = np.max([w.shape[0] for w in x_windows])
        for x_w in x_windows:
            x_w_fillna = x_w.fillna(method="bfill")
            x_w_fillna = x_w_fillna.fillna(x.mean())
            x_w_norm = normalizer_x.transform(x_w_fillna.values)

            x_w_shape = x_w.shape
            if x_w_shape[0] < size_window:
                npad = [(0, size_window - x_w_shape[0]), (0, 0)]
                x_w_norm = np.pad(x_w_norm, pad_width=npad, mode="wrap")

            x_windows_processed.append(x_w_norm)

        return np.array(x_windows_processed), x_windows_indices, normalizer_x

    def process_reversely_data(self, x_imputed, x_indices, x_normalizer, df_ref):
        x_imputed_only = []
        for x_imputed_batch, x_indices_batch in zip(x_imputed, x_indices):
            imputed_index = len(x_indices_batch)
            x_imputed_only += list(x_imputed_batch[:imputed_index])

        x_out_index = pd.MultiIndex.from_tuples(np.concatenate(x_indices), names=df_ref.index.names)
        x_normalized = x_normalizer.inverse_transform(x_imputed_only)
        x_out = pd.DataFrame(
                x_normalized,
                columns=df_ref.columns,
                index=x_out_index,
            )

        return x_out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        arr_preprocessed, list_indices, normalizer = self.process_data(df_data, freq_str='1M', index_datetime='datetime')

        if self.model_name == 'SAITS':
            self.model = SAITS(
                n_steps=arr_preprocessed.shape[1],
                n_features=self.n_features,
                n_layers=self.n_layers,
                d_model=self.d_model,
                d_inner=self.d_inner,
                n_heads=self.n_heads,
                d_k=self.d_k,
                d_v=self.d_v,
                epochs=self.epochs,
                batch_size=self.batch_size
            )
        elif self.model_name == 'Transformer':
            self.model = Transformer(
                n_steps=arr_preprocessed.shape[1],
                n_features=self.n_features,
                n_layers=self.n_layers,
                d_model=self.d_model,
                d_inner=self.d_inner,
                n_heads=self.n_heads,
                d_k=self.d_k,
                d_v=self.d_v,
                epochs=self.epochs,
                batch_size=self.batch_size
            )
        elif self.model_name == 'BRITS':
            self.model = BRITS(
                n_steps=arr_preprocessed.shape[1],
                n_features=self.n_features,
                rnn_hidden_size=self.d_model,
                epochs=self.epochs,
                batch_size=self.batch_size
            )
        elif self.model_name == 'MRNN':
            self.model = MRNN(
                n_steps=arr_preprocessed.shape[1],
                n_features=self.n_features,
                rnn_hidden_size=self.d_model,
                epochs=self.epochs,
                batch_size=self.batch_size
            )
        else:
            self.model = SAITS(
                n_steps=arr_preprocessed.shape[1],
                n_features=self.n_features,
                n_layers=self.n_layers,
                d_model=self.d_model,
                d_inner=self.d_inner,
                n_heads=self.n_heads,
                d_k=self.d_k,
                d_v=self.d_v,
                epochs=self.epochs,
                batch_size=self.batch_size
            )

        self.model.fit({"X": arr_preprocessed})
        arr_imputed = self.model.impute({"X": arr_preprocessed})
        df_imputed = self.process_reversely_data(arr_imputed, list_indices, normalizer, X)

        return df_imputed
```

```python
# from hyperimpute.plugins.imputers.plugin_miracle import MiraclePlugin
# from hyperimpute.plugins.imputers.plugin_gain import GainPlugin
# from hyperimpute.plugins.imputers.plugin_miwae import MIWAEPlugin

# dict_imputers = {}

# dict_imputers['TabDDPM_num_sampling=50'] = imputers_pytorch.ImputerDiffusion(model=ddpms.TabDDPM(num_noise_steps=100, num_blocks=1, dim_embedding=512, num_sampling=50), batch_size=15000, epochs=100, print_valid=False)
# dict_imputers['TabDDPM_num_sampling=1'] = imputers_pytorch.ImputerDiffusion(model=ddpms.TabDDPM(num_noise_steps=100, num_blocks=1, dim_embedding=512, num_sampling=1), batch_size=15000, epochs=100, print_valid=False)

# dict_imputers['SAITS'] = PypotsWrapper(n_features=11, n_layers=2, d_model=256, d_inner=128, n_heads=4, d_k=64, d_v=64, epochs=100, batch_size=100, model_name='SAITS')
# dict_imputers['Transformer'] = PypotsWrapper(n_features=11, n_layers=2, d_model=256, d_inner=128, n_heads=4, d_k=64, d_v=64, epochs=100, batch_size=100, model_name='Transformer')
# dict_imputers['BRITS'] = PypotsWrapper(n_features=11, n_layers=2, d_model=256, d_inner=128, n_heads=4, d_k=64, d_v=64, epochs=100, batch_size=100, model_name='BRITS')
# dict_imputers['MRNN'] = PypotsWrapper(n_features=11, n_layers=2, d_model=256, d_inner=128, n_heads=4, d_k=64, d_v=64, epochs=100, batch_size=100, model_name='MRNN')

# dict_imputers['MIRACLE'] = MiraclePlugin(max_steps=100)
# dict_imputers['GAIN'] = GainPlugin(n_epochs=100)
# dict_imputers['MIWAE'] = MIWAEPlugin(n_epochs=100, batch_size=100)
```

```python
# generator_holes = missing_patterns.UniformHoleGenerator(n_splits=50, ratio_masked=0.2)
# dict_imputers_comparator = {**dict_imputers}

# comparison = Comparator_(
#     dict_imputers_comparator,
#     cols_to_impute,
#     generator_holes = generator_holes,
#     metrics=["mae", "wasserstein_columnwise"],
#     max_evals=5,
#     dict_config_opti=dict_config_opti,
# )
# results = comparison.compare(df_data_raw, '/home/ec2-ngo/qolmat/examples/data/benchmark/ablation_comparison_DL.pkl')
```

```python
results_dl = pd.read_pickle('/home/ec2-ngo/qolmat/examples/data/benchmark/ablation_comparison_DL.pkl')

results_com = pd.read_pickle('/home/ec2-ngo/qolmat/examples/data/benchmark/ablation_comparison.pkl')
```

```python
results_dl.groupby(axis=0, level=0).mean().groupby(axis=1, level=0).mean()
```

```python

```
