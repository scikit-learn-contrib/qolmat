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
import hyperopt as ho

import plotly.graph_objects as go
import plotly.express as px
import inspect
import pickle
from scipy.stats import t

from functools import partial
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from qolmat.benchmark import comparator, missing_patterns, metrics
from qolmat.imputations import imputers, imputers_pytorch
from qolmat.imputations.diffusions import diffusions
from qolmat.utils import data, plot
```

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

## **I. Load data**

```python
df_data_raw = data.get_data("Beijing_offline", datapath='/home/ec2-ngo/qolmat/examples/data')
df_data = data.add_holes(df_data_raw, ratio_masked=.2, mean_size=120)

df_mask = df_data.isna()
df_mask[df_data_raw.isna()] = False
```

```python
dict_imputers = {}
dfs_imputed = {}
```

```python
%%time
dim_input = 11
imputer_TabDDPM = imputers_pytorch.ImputerGenerativeModelPytorch(model=diffusions.TabDDPM(dim_input=dim_input, num_noise_steps=500, num_blocks=1, dim_embedding=512), batch_size=1000, epochs=500, print_valid=True)
imputer_TabDDPM = imputer_TabDDPM.fit(df_data_eval)
```

```python
df_data_raw_eval = df_data_raw
df_data_eval = df_data
df_mask_eval = df_mask

dfs_imputed_sampling = []
dict_imputers["TabDDPM"].model.set_hyperparams_predict(num=1, batch_size_predict=100000)
for i in range(100):
    print(i, end=' ---- ')
    dfs_imputed_sampling.append(dict_imputers["TabDDPM"].transform(df_data_eval))
```

```python
df_comp_conf = []
size = 50
for conf in np.arange(0., 1.0, 0.05):
    dfs_imputed_sampling_ = dfs_imputed_sampling[:size]
    coverage, dfs_lb, dfs_ub = get_confidence_interval(dfs_imputed_sampling_, df_data_raw_eval, df_mask_eval, size, confidence=conf)
    width = (dfs_ub - dfs_lb).mean().mean()
    df_comp_conf.append({'sample num': size, 'confidence': conf, 'coverage':coverage, 'mean width (upper_bound - lower_bound)': width})

display(pd.DataFrame(df_comp_conf))
```

```python
df_comp_size = []
conf = 0.95
for size in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    dfs_imputed_sampling_ = dfs_imputed_sampling[:size]
    coverage, dfs_lb, dfs_ub = get_confidence_interval(dfs_imputed_sampling_, df_data_raw_eval, df_mask_eval, size, confidence=conf)
    width = (dfs_ub - dfs_lb).mean().mean()
    df_comp_size.append({'sample num': size, 'confidence': conf, 'coverage':coverage, 'mean width (upper_bound - lower_bound)': width})

display(pd.DataFrame(df_comp_size))
```

```python
size = 100
confidence = 0.95
coverage, dfs_plot_lb, dfs_plot_ub = get_confidence_interval(dfs_imputed_sampling[:size], df_data_raw_eval, df_mask_eval, size=size, confidence=confidence)

station = df_data_raw_eval.index.get_level_values(0).unique()[2]
col = 'TEMP'

df_mask_eval_plot = df_mask_eval.loc[station][col]

df_data_raw_eval_plot = df_data_raw.loc[station][col]
# df_data_raw_eval_plot[df_mask_eval_plot] = pd.NA

dfs_plot_lb_plot = dfs_plot_lb.loc[station][col]
dfs_plot_lb_plot[~df_mask_eval_plot] = pd.NA

dfs_plot_ub_plot = dfs_plot_ub.loc[station][col]
dfs_plot_ub_plot[~df_mask_eval_plot] = pd.NA

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_data_raw_eval_plot.index, y=df_data_raw_eval_plot, mode='lines', name='Ref'))
fig.add_trace(go.Scatter(x=dfs_plot_lb_plot.index, y=dfs_plot_lb_plot, mode='lines', name='lb'))
fig.add_trace(go.Scatter(x=dfs_plot_ub_plot.index, y=dfs_plot_ub_plot, mode='lines', name='ub'))
fig.update_layout(title=f'Sample num= {size}, confidence={confidence}')
fig.show()
```

```python
# dfs_mask_eval = []
# for m in range(10):
#     df_data_eval = data.add_holes(df_data_raw, ratio_masked=.2, mean_size=120)
#     df_mask_eval = df_data.isna()
#     df_mask_eval[df_data_raw.isna()] = False
#     dfs_mask_eval.append(df_mask_eval)

#     dfs_imputed_sampling = []
#     dict_imputers["TabDDPM"].model.set_hyperparams_predict(num=1, batch_size_predict=100000)
#     for i in range(100):
#         dfs_imputed_sampling.append(dict_imputers["TabDDPM"].transform(df_data_eval))
```
