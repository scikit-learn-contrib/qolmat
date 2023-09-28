---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: qolmat-_zMstDTT
    language: python
    name: python3
---

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
from qolmat.imputations.diffusions import diffusions
from qolmat.utils import data
```

```python
df_data_raw = data.get_data("Beijing_offline", datapath='/home/ec2-ngo/qolmat/examples/data')

cols_to_impute = df_data_raw.columns
n_stations = len(df_data_raw.groupby("station").size())
n_cols = len(cols_to_impute)

display(df_data_raw.describe())
display(df_data_raw.isna().sum())
```

```python
import pandas as pd
from pypots.imputation import SAITS, Transformer, BRITS, MRNN
from qolmat.imputations.imputers import _Imputer
from typing import Optional, List
from sklearn import preprocessing

def process_data(x: pd.DataFrame, freq_str: str, index_datetime: str):
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

def process_reversely_data(x_imputed, x_indices, x_normalizer, df_ref):
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

def process_attention_matrix(x_attentions, x_indices):
    x_index_attention = {}
    for x_attentions_batch, x_indices_batch in zip(x_attentions, x_indices):
        for idx in range(len(x_indices_batch)):
            x_index_attention[x_indices_batch[idx]] = dict(zip(x_indices_batch, x_attentions_batch[idx]) )

    return x_index_attention

def plot_attention_matrix(dict_matrix, index):
    matrix = dict_matrix[index]
    df_matrix = pd.DataFrame(matrix.values(), columns=pd.MultiIndex.from_tuples([index]), index=pd.MultiIndex.from_tuples(matrix.keys())).T
    return df_matrix
```

```python
from qolmat.imputations.imputers_pypots import TransformerExplaination

arr_preprocessed, list_indices, normalizer = process_data(df_data_raw, freq_str='1M', index_datetime='datetime')

model = TransformerExplaination(
    n_steps=arr_preprocessed.shape[1],
    n_features=11,
    n_layers=2,
    d_model=256,
    d_inner=128,
    n_heads=4,
    d_k=64,
    d_v=64,
    epochs=1,
    batch_size=1,
)

model.fit({"X": arr_preprocessed})
arr_imputed, arr_attention = model.impute({"X": arr_preprocessed})
df_imputed = process_reversely_data(arr_imputed, list_indices, normalizer, df_data_raw)

dict_attention_matrix = process_attention_matrix(arr_attention, list_indices)
```

```python
plot_attention_matrix(dict_attention_matrix, index=('Aotizhongxin', pd.Timestamp('2013-03-01 00:00:00'))).T
```

```python

```
