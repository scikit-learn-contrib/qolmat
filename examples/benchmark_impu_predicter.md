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
sys.path.append('/home/ec2-user/qolmat/')

# import warnings
# warnings.filterwarnings('error')
import pandas as pd
import numpy as np
import pickle

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

# dataset = load_dataset("inria-soda/tabular-benchmark", data_files="reg_num/wine_quality.csv")
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
#     # missing_patterns.MCAR(ratio_masked=0.1),
#     missing_patterns.MCAR(ratio_masked=0.3),
#     # missing_patterns.MCAR(ratio_masked=0.5),
#     # missing_patterns.MCAR(ratio_masked=0.7),
# ]

# imputation_pipelines = [None,
#                         # {"imputer": imputers.ImputerMean()},
#                         {"imputer": imputers.ImputerEM(max_iter_em=2, method='mle')},
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
#     df_data=df_data.iloc[:1000],
#     columns_numerical=columns_numerical,
#     columns_categorical=columns_categorical,
#     file_path=f"data/benchmark_prediction.pkl",
#     hole_generators=hole_generators,
#     imputation_pipelines=imputation_pipelines,
#     target_prediction_pipeline_pairs=target_prediction_pipeline_pairs,
# )
```

```python
# results = pd.read_pickle('/home/ec2-user/qolmat/examples/data/benchmark_prediction.pkl')
# results_agg = imppred.get_benchmark_aggregate(results, cols_groupby=['hole_generator', 'ratio_masked', 'imputer', 'predictor'])
# results_agg
```

# Visualisation

```python
# results = pd.read_pickle('/home/ec2-user/qolmat/examples/data/imp_pred/benchmark_houses.pkl')
# results = pd.read_pickle('/home/ec2-user/qolmat/examples/data/imp_pred/benchmark_MiamiHousing2016.pkl')
# results = pd.read_pickle('/home/ec2-user/qolmat/examples/data/imp_pred/benchmark_elevators.pkl')
```

```python
# visualize_mlflow(results, exp_name='census_income')
```

```python
# results_agg = imppred.get_benchmark_aggregate(results, cols_groupby=['hole_generator', 'ratio_masked', 'imputer', 'predictor'])
# display(results_agg)
```

```python
# results_agg = imppred.get_benchmark_aggregate(results[results['predictor']=='HistGradientBoostingRegressor'], cols_groupby=['hole_generator', 'ratio_masked', 'imputer'])
# display(results_agg[[('prediction_score', 'test_set_not_nan', 'mae')]])
```

```python
# selected_columns=['n_fold', 'hole_generator', 'ratio_masked', 'imputer', 'predictor', 'prediction_score_nan_mae', 'duration_imputation_fit']
# fig = imppred.visualize_plotly(results, selected_columns=selected_columns)
# fig.update_layout(height=300, width=1000)
# fig
```

# Export

```python
# results_1 = pd.read_pickle('/home/ec2-user/qolmat/examples/data/imp_pred/benchmark_sulfur.pkl')
# results_1['dataset'] = 'sulfur'
# results_2 = pd.read_pickle('/home/ec2-user/qolmat/examples/data/imp_pred/benchmark_wine_quality.pkl')
# results_2['dataset'] = 'wine_quality'
# results_3 = pd.read_pickle('/home/ec2-user/qolmat/examples/data/imp_pred/benchmark_MiamiHousing2016.pkl')
# results_3['dataset'] = 'MiamiHousing2016'
# results_4 = pd.read_pickle('/home/ec2-user/qolmat/examples/data/imp_pred/benchmark_elevators.pkl')
# results_4['dataset'] = 'elevators'

# results = pd.concat([results_1, results_2, results_3, results_4]).reset_index(drop=True)
# with open('/home/ec2-user/qolmat/examples/data/imp_pred/benchmark_all.pkl', "wb") as handle:
#     pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

```python
# results = pd.read_pickle('/home/ec2-user/qolmat/examples/data/imp_pred/benchmark_all.pkl')

# results_agg = imppred.get_benchmark_aggregate(results, cols_groupby=['dataset', 'hole_generator', 'ratio_masked', 'imputer', 'predictor'])

# results_agg.reset_index(inplace=True)
# results_agg.columns = ['_'.join(col).replace('__', '') for col in results_agg.columns.values]
# results_agg.to_csv('/home/ec2-user/qolmat/examples/data/imp_pred/benchmark_all.csv', index=False)
```

# Questions

```python
results = pd.read_pickle('/home/ec2-user/qolmat/examples/data/imp_pred/benchmark_all.pkl')
results_plot = results.copy()
```

```python
results_agg = imppred.get_benchmark_aggregate(results, cols_groupby=['dataset', 'hole_generator', 'ratio_masked', 'imputer', 'predictor'])
display(results_agg)
```

```python
print(results['dataset'].unique())
print(results['predictor'].unique())
```

## L’imputation améliore la performance des modèles ayant la capacité de traitement des nans ?

```python
results_plot
```

```python
results_plot['prediction_score_notnan_mae_relative_percentage_gain'] = results_plot.apply(lambda x: imppred.get_relative_score(x, results_plot, col='prediction_score_notnan_mae', method='relative_percentage_gain'), axis=1)

results_plot['prediction_score_notnan_mae_gain'] = results_plot.apply(lambda x: imppred.get_relative_score(x, results_plot, col='prediction_score_notnan_mae', method='gain'), axis=1)
results_plot['prediction_score_notnan_mae_gain_count'] = results_plot.apply(lambda x: 1 if x['prediction_score_notnan_mae_gain'] > 0 else 0, axis=1)

num_runs = results_plot.groupby(['hole_generator', 'ratio_masked', 'imputer', 'predictor']).count().max().max()
results_plot['prediction_score_notnan_mae_gain_ratio'] = results_plot['prediction_score_notnan_mae_gain_count']/num_runs
```

```python
# model = 'HistGradientBoostingRegressor'
model = 'XGBRegressor'

fig = imppred.plot_bar(
    results_plot[(results_plot['predictor'].isin([model]))
                 & ~(results_plot['imputer'].isin(['None']))
                 ],
    col_displayed=("prediction_score", "test_set_not_nan", "mae_gain_ratio"),
    cols_grouped=['hole_generator', 'ratio_masked', 'imputer'],
    add_annotation=True,
    add_confidence_interval=False,
    title='XGBRegressor',
    agg_func=pd.DataFrame.sum)

fig.update_layout(title=f"Ratio of runs (over 25 trials x 4 datasets) where predictive performance MAE <br>of {model} is enhanced through imputation compared to scenarios without imputation.")
fig.update_xaxes(title="Types and Ratios of missing values")
fig.update_yaxes(title="Ratio of runs")
fig.update_layout(height=400, width=1000)
fig
```

```python
model = 'HistGradientBoostingRegressor'
# model = 'XGBRegressor'

fig = imppred.plot_bar(
    results_plot[(results_plot['predictor'].isin([model]))
                 & ~(results_plot['imputer'].isin(['None']))
                #  & (results_plot['dataset'].isin(['sulfur']))
                 ],
    col_displayed=("prediction_score", "test_set_not_nan", "mae_relative_percentage_gain"),
    cols_grouped=['dataset', 'ratio_masked', 'imputer'],
    add_annotation=False,
    add_confidence_interval=True,
    confidence_level=0.95,
    title='XGBRegressor',
    agg_func=pd.DataFrame.mean)

fig.update_layout(title=f"Mean relative percentage gain of MAE over 25 trials, for {model}")
fig.update_xaxes(title="Datasets and Ratios of missing values")
fig.update_yaxes(title="1 - MAE(I+P) / MAE(P)")
fig.update_layout(height=400, width=2000)
fig
```

## Quel couple imputeur-prédicteur trouve le meilleur résultat pour quel ratio de nan ?

```python
results_plot['prediction_score_notnan_mae_rank'] = results_plot.groupby(['dataset', 'n_fold', 'hole_generator', 'ratio_masked', 'n_mask'])['prediction_score_notnan_mae'].rank()
```

```python
# model = 'HistGradientBoostingRegressor'
# model = 'XGBRegressor'
model = 'Ridge'

fig = imppred.plot_bar(
    results_plot[
        ~(results_plot['hole_generator'].isin(['None']))
        # (results_plot['predictor'].isin([model ]))
        ],
    col_displayed=("prediction_score", "test_set_not_nan", "mae_rank"),
    cols_grouped=['predictor','ratio_masked', 'imputer'],
    add_annotation=True,
    add_confidence_interval=False,
    agg_func=pd.DataFrame.mean)


fig.update_layout(title=f"Average ranks for each pair imputer-predictor for datasets and trials.")
# fig.update_xaxes(title=f"Ratios of nan with predictor={model}")
fig.update_xaxes(title=f"Predictors and ratios of nan")
fig.update_yaxes(title="Average rank")
fig.update_layout(height=500, width=2000)
fig
```

## La performance de prédiction est-elle corrélée à celle de l’imputation ?

```python
print(results['dataset'].unique())

dataset = 'elevators'
fig = imppred.plot_scatter(results, cond={'dataset':dataset}, col_x='imputation_score_mae_train_set', col_y='prediction_score_notnan_mae')
fig.update_layout(legend_title="Nan ratio")
fig.update_layout(title=f"MAEs of all pairs imputer-predictor for 25 trials, on the dataset {dataset}")
fig.update_xaxes(title=f"MAE for imputation on the train set")
fig.update_yaxes(title="MAE for prediction on the test set (without imputation)")
fig.update_layout(height=500, width=1000)

fig
```

```python
print(results['predictor'].unique())

predictor = 'Ridge'
fig = imppred.plot_scatter(results, cond={'predictor':predictor}, col_x='imputation_score_mae_train_set', col_y='prediction_score_notnan_mae')
fig.update_layout(legend_title="Nan ratio")
fig.update_layout(title=f"MAEs of all pairs imputer-predictor for 25 trials, on the predictor {predictor}")
fig.update_xaxes(title=f"MAE for imputation on the train set")
fig.update_yaxes(title="MAE for prediction on the test set (without imputation)")
fig.update_layout(height=500, width=1000)

fig
```

```python
# results_corr = results[['imputation_score_mae_train_set', 'prediction_score_notnan_mae']].corr(method='spearman')
results_corr = results.groupby('predictor')[['imputation_score_mae_train_set', 'prediction_score_notnan_mae']].corr(method='spearman')

multi_index_columns = [
    ('imputation', 'mae', 'train_set'),
    ('prediction', 'mae', 'test_set_not_nan'),
]

results_corr.columns = pd.MultiIndex.from_tuples(multi_index_columns)
multi_index_rows = []
if results_corr.index.shape[0] > results_corr.columns.shape[0]:
    for row_index_0 in results_corr.index.get_level_values(0).unique():
        for row_index_1 in multi_index_columns:
            multi_index_rows.append([row_index_0] + list(row_index_1))
    results_corr.index = pd.MultiIndex.from_tuples(multi_index_rows)
else:
    results_corr.index = pd.MultiIndex.from_tuples(multi_index_columns)

display(results_corr)
```

```python
# results_corr = results[['imputation_score_mae_test_set', 'prediction_score_nan_mae']].corr(method='spearman')
results_corr = results.groupby('predictor')[['imputation_score_mae_test_set', 'prediction_score_nan_mae']].corr(method='spearman')

multi_index_columns = [
    ('imputation', 'mae', 'test_set'),
    ('prediction', 'mae', 'test_set_with_nan'),
]

results_corr.columns = pd.MultiIndex.from_tuples(multi_index_columns)
multi_index_rows = []
if results_corr.index.shape[0] > results_corr.columns.shape[0]:
    for row_index_0 in results_corr.index.get_level_values(0).unique():
        for row_index_1 in multi_index_columns:
            multi_index_rows.append([row_index_0] + list(row_index_1))
    results_corr.index = pd.MultiIndex.from_tuples(multi_index_rows)
else:
    results_corr.index = pd.MultiIndex.from_tuples(multi_index_columns)

display(results_corr)
```
