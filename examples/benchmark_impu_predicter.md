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

import qolmat.benchmark.imputer_predictor as imppred
```

# Load data

```python
# # from urllib import request
# # import zipfile

# # data_url = "https://archive.ics.uci.edu/static/public/20/census+income.zip"
# data_path = "data/census+income"
# # request.urlretrieve(data_url, data_path + ".zip")
# # with zipfile.ZipFile(data_path + ".zip", "r") as zip_ref:
# #     zip_ref.extractall(data_path)

# data_types = {'age': 'int32', 'workclass': 'string', 'fnlwgt': 'float32', 'education': 'string', 'education-num': 'int32', 'marital-status': 'string', 'occupation': 'string', 'relationship': 'string', 'race': 'string', 'sex': 'string', 'capital-gain': 'float32', 'capital-loss': 'float32', 'hours-per-week': 'int32', 'native-country': 'string', 'income': 'string'}
# df_data = pd.read_csv(data_path+"/adult.data", header=None, names=data_types.keys(), dtype=data_types)

# columns_categorical = df_data.dtypes[(df_data.dtypes=='string')].index.to_list()
# columns_numerical = df_data.dtypes[(df_data.dtypes=='float32') | (df_data.dtypes=='int32')].index.to_list()

# print(f'df shape: {df_data.shape}, cols cat: {len(columns_categorical)}, cols num: {len(columns_numerical)}')
```

```python
# data_path = "data/conductors.csv"
# df_data = pd.read_csv(data_path)

# columns_categorical = df_data.dtypes[(df_data.dtypes=='int64')].index.to_list()
# columns_numerical = df_data.dtypes[(df_data.dtypes=='float64')].index.to_list()

# print(f'df shape: {df_data.shape}, cols cat: {len(columns_categorical)}, cols num: {len(columns_numerical)}')
```

```python
# from datasets import load_dataset

# dataset = load_dataset("inria-soda/tabular-benchmark", data_files="reg_num/elevators.csv")
# df_data = dataset["train"].to_pandas()
# column_target = df_data.columns.to_list()[-1]
# columns_numerical = df_data.select_dtypes(include='number').columns.tolist()
# columns_categorical = df_data.select_dtypes(include='object').columns.tolist()

# print(f'df shape: {df_data.shape}, cols cat: {len(columns_categorical)}, cols num: {len(columns_numerical)}')
```

# Experiment

```python
# from sklearn.compose import ColumnTransformer
# from sklearn import preprocessing

# from qolmat.benchmark import missing_patterns
# from qolmat.imputations import imputers, imputers_pytorch
# from qolmat.imputations.diffusions import ddpms

# from sklearn.linear_model import Ridge, RidgeClassifier
# from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
# from xgboost import XGBClassifier, XGBRegressor
# from lightgbm import LGBMRegressor, LGBMClassifier

# # Hole generators
# hole_generators = [
#     # None,
#     missing_patterns.MCAR(ratio_masked=0.05),
# ]

# imputation_pipelines = [#None,
#                         # {"imputer": imputers.ImputerMean()},
#                         {"imputer": imputers.ImputerEM(max_iter_em=2)},
#                         ]

# # Prediction pipelines
# transformers = []
# columns_numerical_ = [col for col in columns_numerical if col != column_target]
# if len(columns_numerical_) != 0:
#     transformers.append(("num", preprocessing.StandardScaler(), columns_numerical_))
# columns_categorical_ = [col for col in columns_categorical if col != column_target]
# if len(columns_categorical) != 0:
#     transformers.append(("cat", preprocessing.OrdinalEncoder(), columns_categorical_))
# transformer_prediction_x = ColumnTransformer(transformers=transformers)

# target_prediction_pipeline_pairs = {}

# if column_target in columns_numerical:
#     transformer_prediction_y = ColumnTransformer(
#         transformers=[
#             ("y_num", preprocessing.StandardScaler(), [column_target]),
#         ]
#     )
#     target_prediction_pipeline_pairs[column_target] = [
#         {
#             "transformer_x": transformer_prediction_x,
#             "transformer_y": transformer_prediction_y,
#             "predictor": Ridge(),
#             "handle_nan": False,
#         },
#     ]

# benchmark = imppred.BenchmarkImputationPrediction(
#     n_masks=1,
#     n_folds=2,
#     imputation_metrics=["mae", "KL_columnwise"],
#     prediction_metrics=["mae"],
# )

# results = benchmark.compare(
#     df_data=df_data,
#     columns_numerical=columns_numerical,
#     columns_categorical=columns_categorical,
#     file_path=f"data/benchmark_prediction.pkl",
#     hole_generators=hole_generators,
#     imputation_pipelines=imputation_pipelines,
#     target_prediction_pipeline_pairs=target_prediction_pipeline_pairs,
# )
```

```python
# results = pd.read_pickle('/home/ec2-ngo/qolmat/examples/data/benchmark_prediction.pkl')
# results_agg = imppred.get_benchmark_aggregate(results, cols_groupby=['hole_generator', 'ratio_masked', 'imputer', 'predictor'])
# results_agg
```

# Visualisation

```python
# results = pd.read_pickle('/home/ec2-ngo/qolmat/examples/data/benchmark_prediction_census_income.pkl')
# visualize_mlflow(results, exp_name='census_income')
# results = pd.read_pickle('/home/ec2-ngo/qolmat/examples/data/benchmark_prediction_conductors.pkl')
# results = pd.read_pickle('/home/ec2-ngo/qolmat/examples/data/imp_pred/benchmark_houses.pkl')

results = pd.read_pickle('/home/ec2-ngo/qolmat/examples/data/imp_pred/benchmark_wine_quality.pkl')
```

```python
# results_1 = pd.read_pickle('/home/ec2-ngo/qolmat/examples/data/imp_pred/benchmark_elevators_add_nan_indicator.pkl')
# results_2 = pd.read_pickle('/home/ec2-ngo/qolmat/examples/data/imp_pred/benchmark_houses_add_nan_indicator.pkl')
# results = pd.concat([results_1, results_2])
# results['hole_generator'] = results['hole_generator'].replace('UniformHoleGenerator', 'MCAR')

```

```python
results_agg = imppred.get_benchmark_aggregate(results, cols_groupby=['hole_generator', 'ratio_masked', 'imputer', 'predictor'])
display(results_agg)
```

```python

```

```python
results_agg = imppred.get_benchmark_aggregate(results[results['predictor']=='HistGradientBoostingRegressor'], cols_groupby=['hole_generator', 'ratio_masked', 'imputer'])
display(results_agg[[('prediction_score', 'test_set_not_nan', 'mae')]])
```

```python
selected_columns=['n_fold', 'hole_generator', 'ratio_masked', 'imputer', 'predictor', 'prediction_score_nan_mae', 'duration_imputation_fit']
fig = imppred.visualize_plotly(results, selected_columns=selected_columns)
# fig.update_layout(height=300, width=1000)
fig
```

```python
fig = imppred.plot_bar(
                results[results['predictor'].isin(['HistGradientBoostingRegressor'])],
                # results,
                col_displayed=("prediction_score", "test_set_with_nan", "mae"),
                cols_grouped=['hole_generator', 'ratio_masked', 'imputer'],
                add_annotation=False,
                add_confidence_interval=False,
                title='HistGradientBoostingRegressor')

fig.update_layout(height=400, width=1000)
fig

```

```python
fig = imppred.plot_bar(
    results[results['imputer'].isin(['ImputerMean', 'ImputerMICE', 'ImputerKNN', 'ImputerRPCA', 'ImputerEM', 'ImputerDiffusion'])],
    results,
    cols_displayed=[
                ("imputation_score", "train_set", "mae"),
                ("imputation_score", "test_set", "mae"),
                ("prediction_score", "test_set_with_nan", "mae"),
                ],
    cols_grouped=['hole_generator', 'imputer'],
    add_annotation=False,
    add_confidence_interval=False)

# fig.update_layout(height=300, width=1000)
fig
```

```python
import plotly.express as px

results_agg = imppred.get_benchmark_aggregate(results, cols_groupby=['hole_generator', 'ratio_masked', 'imputer', 'predictor'])
fig = px.scatter(x=results_agg[("imputation_score", "test_set", "mae")],
                 y=results_agg[("prediction_score", "test_set_with_nan", "mae")],
                 color=results_agg.index.get_level_values('imputer'),)
fig.show()
```

```python

```
