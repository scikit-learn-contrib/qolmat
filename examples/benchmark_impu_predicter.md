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
from scipy import stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import scikit_posthocs as sp

from datasets import load_dataset

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

# results = pd.read_pickle('data/benchmark_prediction.pkl')
# results_agg = imppred.get_benchmark_aggregate(results, cols_groupby=['hole_generator', 'ratio_masked', 'imputer', 'predictor'])
# results_agg
```

## Computational time

```python
# from qolmat.imputations import imputers, imputers_pytorch
# from qolmat.imputations.diffusions import ddpms
# from qolmat.benchmark import missing_patterns
# from xgboost import XGBRegressor
# import time

# imputers = [
#     imputers.ImputerMedian(),
#     imputers.ImputerShuffle(),
#     imputers.ImputerMICE(estimator=XGBRegressor(tree_method="hist", n_jobs=1), max_iter=100),
#     imputers.ImputerKNN(),
#     imputers.ImputerRPCA(max_iterations=100),
#     imputers.ImputerEM(max_iter_em=100, method="mle"),
#     imputers_pytorch.ImputerDiffusion(model=ddpms.TabDDPM(num_sampling=50), batch_size=1000, epochs=100)
#     ]

# benchmark_duration_rows = []
# num_cols = 5
# for num_rows in [100, 150]:
#     df_sub_data = df_data.iloc[:num_rows, :num_cols]
#     hole_generator = missing_patterns.MCAR(ratio_masked=0.1)
#     df_sub_mask = hole_generator.split(df_sub_data)[0]
#     df_sub_data[df_sub_mask] = np.nan

#     for imputer in imputers:
#         start_time = time.time()
#         imputer = imputer.fit(df_sub_data)
#         duration_imputation_fit = time.time() - start_time

#         start_time = time.time()
#         df_imputed = imputer.transform(df_sub_data)
#         duration_imputation_transform = time.time() - start_time

#         benchmark_duration_rows.append({
#             'imputer': imputer.__class__.__name__,
#             'n_columns': df_sub_data.shape[1],
#             'size_data': df_sub_data.shape[0],
#             'duration_imputation_fit': duration_imputation_fit,
#             'duration_imputation_transform': duration_imputation_transform,
#         })

# benchmark_duration_cols = []
# num_rows = 100
# for num_cols in [5, 6]:
#     df_sub_data = df_data.iloc[:num_rows, :num_cols]
#     hole_generator = missing_patterns.MCAR(ratio_masked=0.1)
#     df_sub_mask = hole_generator.split(df_sub_data)[0]
#     df_sub_data[df_sub_mask] = np.nan

#     for imputer in imputers:
#         start_time = time.time()
#         imputer = imputer.fit(df_sub_data)
#         duration_imputation_fit = time.time() - start_time

#         start_time = time.time()
#         df_imputed = imputer.transform(df_sub_data)
#         duration_imputation_transform = time.time() - start_time

#         benchmark_duration_cols.append({
#             'imputer': imputer.__class__.__name__,
#             'n_columns': df_sub_data.shape[1],
#             'size_data': df_sub_data.shape[0],
#             'duration_imputation_fit': duration_imputation_fit,
#             'duration_imputation_transform': duration_imputation_transform,
#         })

# df_benchmark_rows = pd.DataFrame(benchmark_duration_rows)
# with open('data/imp_pred/benchmark_time_rows.pkl', "wb") as handle:
#     pickle.dump(df_benchmark_rows, handle, protocol=pickle.HIGHEST_PROTOCOL)

# df_benchmark_cols = pd.DataFrame(benchmark_duration_cols)
# with open('data/imp_pred/benchmark_time_cols.pkl', "wb") as handle:
#     pickle.dump(df_benchmark_cols, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

# Visualisation

```python
# results = pd.read_pickle('data/imp_pred/benchmark_houses.pkl')
# results = pd.read_pickle('data/imp_pred/benchmark_elevators.pkl')
# results = pd.read_pickle('data/imp_pred/benchmark_MiamiHousing2016.pkl')
# results = pd.read_pickle('data/imp_pred/benchmark_Brazilian_houses.pkl')
# results = pd.read_pickle('data/imp_pred/benchmark_sulfur.pkl')
# results = pd.read_pickle('data/imp_pred/benchmark_wine_quality.pkl')
```

```python
# visualize_mlflow(results, exp_name='census_income')
```

```python
# results_agg = imppred.get_benchmark_aggregate(results, cols_groupby=['hole_generator', 'ratio_masked', 'imputer', 'predictor'])
# display(results_agg)
```

```python
# selected_columns=['n_fold', 'hole_generator', 'ratio_masked', 'imputer', 'predictor', 'prediction_score_nan_mae', 'duration_imputation_fit']
# fig = imppred.visualize_plotly(results, selected_columns=selected_columns)
# fig.update_layout(height=300, width=1000)
# fig
```

# Export

```python
# results_1 = pd.read_pickle('data/imp_pred/benchmark_sulfur.pkl')
# results_1['dataset'] = 'sulfur'
# results_2 = pd.read_pickle('data/imp_pred/benchmark_wine_quality.pkl')
# results_2['dataset'] = 'wine_quality'
# results_3 = pd.read_pickle('data/imp_pred/benchmark_MiamiHousing2016.pkl')
# results_3['dataset'] = 'MiamiHousing2016'
# results_4 = pd.read_pickle('data/imp_pred/benchmark_elevators.pkl')
# results_4['dataset'] = 'elevators'
# results_5 = pd.read_pickle('data/imp_pred/benchmark_houses.pkl')
# results_5['dataset'] = 'houses'
# results_6 = pd.read_pickle('data/imp_pred/benchmark_Brazilian_houses.pkl')
# results_6['dataset'] = 'Brazilian_houses'
# results_7 = pd.read_pickle('data/imp_pred/benchmark_Bike_Sharing_Demand.pkl')
# results_7['dataset'] = 'Bike_Sharing_Demand'
# results_8 = pd.read_pickle('data/imp_pred/benchmark_diamonds.pkl')
# results_8['dataset'] = 'diamonds'
# results_9 = pd.read_pickle('data/imp_pred/benchmark_medical_charges.pkl')
# results_9['dataset'] = 'medical_charges'

# results = pd.concat([results_1, results_2, results_3,
#                      results_4, results_5, results_6,
#                      results_7, results_8, results_9,]).reset_index(drop=True)
# with open('data/imp_pred/benchmark_all.pkl', "wb") as handle:
#     pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # results = pd.read_pickle('data/imp_pred/benchmark_all.pkl')

# results_agg = imppred.get_benchmark_aggregate(results, cols_groupby=['dataset', 'hole_generator', 'ratio_masked', 'imputer', 'predictor'])

# results_agg.reset_index(inplace=True)
# results_agg.columns = ['_'.join(col).replace('__', '') for col in results_agg.columns.values]
# results_agg.to_csv('data/imp_pred/benchmark_all.csv', index=False)
```

```python
# results_1 = pd.read_pickle('data/imp_pred/benchmark_sulfur_new.pkl')
# results_1['dataset'] = 'sulfur'
# results_2 = pd.read_pickle('data/imp_pred/benchmark_wine_quality_new.pkl')
# results_2['dataset'] = 'wine_quality'
# results_3 = pd.read_pickle('data/imp_pred/benchmark_MiamiHousing2016_new.pkl')
# results_3['dataset'] = 'MiamiHousing2016'
# results_4 = pd.read_pickle('data/imp_pred/benchmark_elevators_new.pkl')
# results_4['dataset'] = 'elevators'
# results_5 = pd.read_pickle('data/imp_pred/benchmark_houses_new.pkl')
# results_5['dataset'] = 'houses'
# results_6 = pd.read_pickle('data/imp_pred/benchmark_Brazilian_houses_new.pkl')
# results_6['dataset'] = 'Brazilian_houses'
# results_7 = pd.read_pickle('data/imp_pred/benchmark_Bike_Sharing_Demand_new.pkl')
# results_7['dataset'] = 'Bike_Sharing_Demand'
# results_8 = pd.read_pickle('data/imp_pred/benchmark_diamonds_new.pkl')
# results_8['dataset'] = 'diamonds'
# results_9 = pd.read_pickle('data/imp_pred/benchmark_medical_charges_new.pkl')
# results_9['dataset'] = 'medical_charges'

# results = pd.concat([results_1, results_2, results_3,
#                      results_4, results_5, results_6,
#                      results_7, results_8, results_9]).reset_index(drop=True)
# with open('data/imp_pred/benchmark_all_new.pkl', "wb") as handle:
#     pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

# results_agg = imppred.get_benchmark_aggregate(results, cols_groupby=['dataset', 'hole_generator', 'ratio_masked', 'imputer', 'predictor'])

# results_agg.reset_index(inplace=True)
# results_agg.columns = ['_'.join(col).replace('__', '') for col in results_agg.columns.values]
# results_agg.to_csv('data/imp_pred/benchmark_all_new.csv', index=False)
```

# Questions

```python
results = pd.read_pickle('data/imp_pred/benchmark_all_new.pkl')
results_plot = results.copy()
```

```python
# results_agg = imppred.get_benchmark_aggregate(results, cols_groupby=['dataset', 'hole_generator', 'ratio_masked', 'imputer', 'predictor'], keep_values=True)
# display(results_agg)
```

```python
num_dataset = len(results['dataset'].unique())
num_predictor = len(results['predictor'].unique())
num_imputer = len(results['imputer'].unique()) - 1
num_fold = len(results['n_fold'].unique())
# We remove the case [hole_generator=None, ratio_masked=0, n_mask=nan]
num_mask = len(results['n_mask'].unique()) - 1
num_ratio_masked = len(results['ratio_masked'].unique()) - 1
num_trial = num_fold * num_mask

print(f"datasets: {results['dataset'].unique()}")
print(f"predictor: {results['predictor'].unique()}")
print(f"imputer: {results['imputer'].unique()}")
```

```python
results_plot[['dataset', 'hole_generator', 'ratio_masked', 'imputer', 'predictor']]
```

```python
results_plot.columns
```

## Are the performances significantly different under all trials and configurations?

Friedman test tests the null hypothesis that performance scores of different imputers in the same trial and configuration have the same distribution.
I.e., we have N sets of performance scores for N imputers. Each set has a size of M trials/configurations.

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html


### Prediction performance

```python
# metric = "mae"
metric = "wmape"
# type_set = 'nan'
type_set = 'notnan'

results_plot_friedman_test_imputer_prediction = results_plot[results_plot['imputer']!='None'].copy()
results_plot_friedman_test_imputer_prediction = results_plot_friedman_test_imputer_prediction.groupby(['dataset', 'n_fold', 'hole_generator', 'ratio_masked', 'n_mask', 'predictor', 'imputer'])[f'prediction_score_{type_set}_{metric}'].aggregate('first').unstack()

col_1 = 'ratio_masked'
col_2 = 'predictor'

values = results_plot_friedman_test_imputer_prediction
for v_1 in results_plot_friedman_test_imputer_prediction.index.get_level_values(col_1).unique():
    values_1 = values.xs(v_1, level=col_1)
    print(f'+ {col_1}={v_1}:')
    for v_2 in results_plot_friedman_test_imputer_prediction.index.get_level_values(col_2).unique():
        values_2 = values_1.xs(v_2, level=col_2).values.T
        print(np.shape(values_2))
        res = stats.friedmanchisquare(*values_2)
        print(f' - {col_2}={v_2}: statistic={res.statistic}, pvalue={res.pvalue}')
```

```python
# metric = "mae"
metric = "wmape"

type_set = 'nan'

results_plot_friedman_test_imputer_predictor = results_plot[results_plot['imputer']!='None'].copy()
results_plot_friedman_test_imputer_predictor['imputer_predictor'] = results_plot_friedman_test_imputer_predictor['imputer'] + '_' + results_plot_friedman_test_imputer_predictor['predictor']

results_plot_friedman_test_imputer_predictor = results_plot_friedman_test_imputer_predictor.groupby(['dataset', 'n_fold', 'hole_generator', 'ratio_masked', 'n_mask', 'imputer_predictor'])[f'prediction_score_{type_set}_{metric}'].aggregate('first').unstack()

col = 'ratio_masked'
for ratio_masked in results_plot_friedman_test_imputer_predictor.index.get_level_values(col).unique():
    values = results_plot_friedman_test_imputer_predictor.xs(ratio_masked, level=col).values.T
    res = stats.friedmanchisquare(*values)
    print(f'{col}={ratio_masked}: statistic={res.statistic}, pvalue={res.pvalue}')
```

The null hypothesis is rejected with p-values way below the 0.05 level for all the ratios. This indicates that at least one algorithm has significantly different performances from one other.


### Imputation performance

```python
# metric = "mae"
metric = "wmape"
evaluated_set = 'train_set'
# evaluated_set = 'test_set'

results_plot_friedman_test_imputer_imputation = results_plot[results_plot['imputer']!='None'].copy()
results_plot_friedman_test_imputer_imputation = results_plot_friedman_test_imputer_imputation.groupby(['dataset', 'n_fold', 'hole_generator', 'ratio_masked', 'n_mask', 'predictor', 'imputer'])[f'imputation_score_{metric}_{evaluated_set}'].aggregate('first').unstack()

col = 'ratio_masked'
for v in results_plot_friedman_test_imputer_imputation.index.get_level_values(col).unique():
    values = results_plot_friedman_test_imputer_imputation.xs(v, level=col).values.T
    res = stats.friedmanchisquare(*values)
    print(f'{col}={v}: statistic={res.statistic}, pvalue={res.pvalue}')
```

## Does Imputation improves the prediction performance of the predictors supporting missing values ?

- Gain = Score(Prediction) - Score(Imputation + Prediction)

```python
num_runs = results_plot.groupby(['hole_generator', 'ratio_masked', 'imputer', 'predictor']).count().max().max()
print(f"num_runs = {num_runs} runs for each {num_dataset} datasets * {num_fold} folds * {num_mask - 1} masks = {num_dataset * num_fold * num_mask}")
```

```python
# metric = 'mae'
metric = 'wmape'

type_set = 'notnan'
# type_set = 'nan'

results_plot[f'prediction_score_{type_set}_{metric}_relative_percentage_gain'] = results_plot.apply(lambda x: imppred.get_relative_score(x, results_plot, col=f'prediction_score_{type_set}_{metric}', method='relative_percentage_gain'), axis=1)

results_plot[f'prediction_score_{type_set}_{metric}_gain'] = results_plot.apply(lambda x: imppred.get_relative_score(x, results_plot, col=f'prediction_score_{type_set}_{metric}', method='gain'), axis=1)
results_plot[f'prediction_score_{type_set}_{metric}_gain_count'] = results_plot.apply(lambda x: 1 if x[f'prediction_score_{type_set}_{metric}_gain'] > 0 else 0, axis=1)

results_plot[f'prediction_score_{type_set}_{metric}_gain_ratio'] = results_plot[f'prediction_score_{type_set}_{metric}_gain_count']/num_runs
```

### Ratio of runs

```python
# model = 'HistGradientBoostingRegressor'
model = 'XGBRegressor'

# metric = 'mae_gain_ratio'
metric = 'wmape_gain_ratio'

type_set = "test_set_not_nan"
# type_set = "test_set_with_nan"

fig = imppred.plot_bar(
    results_plot[(results_plot['predictor'].isin([model]))
                 & ~(results_plot['imputer'].isin(['None']))
                 ],
    col_displayed=("prediction_score", type_set, metric),
    cols_grouped=['hole_generator', 'ratio_masked', 'imputer'],
    add_annotation=True,
    add_confidence_interval=False,
    agg_func=pd.DataFrame.sum)

# fig.update_layout(title=f"Ratio of runs (over {num_trial * num_dataset} runs = {num_trial} trials x {num_dataset} datasets) where a gain of prediction performance <br>is found for {model}. Evaluation based on WMAPE computed on imputed test sets.")
fig.update_layout(title=f"Ratio of runs (over {num_trial * num_dataset} runs = {num_trial} trials x {num_dataset} datasets) where a gain of prediction performance <br>is found for {model}. Evaluation based on WMAPE computed on complete test sets.")
fig.update_xaxes(title="Types and Ratios of missing values")
fig.update_yaxes(title="Ratio of runs")
fig.update_layout(height=400, width=1000)
fig
```

### Gain

```python
# model = 'HistGradientBoostingRegressor'
model = 'XGBRegressor'

# metric = "mae_relative_percentage_gain"
# metric = "wmape_gain"
metric = "wmape_relative_percentage_gain"

type_set = "test_set_not_nan"
# type_set = "test_set_with_nan"

fig = imppred.plot_bar(
    results_plot[(results_plot['predictor'].isin([model]))
                 & ~(results_plot['imputer'].isin(['None']))
                & (results_plot['dataset'].isin(['MiamiHousing2016', 'medical_charges']))
                 ],
    col_displayed=("prediction_score", type_set, metric),
    cols_grouped=['dataset', 'ratio_masked', 'imputer'],
    add_annotation=False,
    add_confidence_interval=True,
    confidence_level=0.95,
    agg_func=pd.DataFrame.mean,
    yaxes_type='log')

# fig.update_layout(title=f"Mean relative percentage gain of prediction performance over {num_trial} trials, for {model}.<br>Evaluation based on WMAPE computed on imputed test sets.")
fig.update_layout(title=f"Mean relative percentage gain of prediction performance over {num_trial} trials, for {model}.<br>Evaluation based on WMAPE computed on complete test sets.")
fig.update_xaxes(title="Datasets and Ratios of missing values")
fig.update_yaxes(title="(WMAPE(P) - WMAPE(I+P))/WMAPE(P)")
fig.update_layout(height=400, width=1000)
fig
```

### The Wilcoxon signed-rank test

```python
# metric = 'mae_gain'
metric = 'wmape_gain'

# type_set = 'nan'
type_set = 'notnan'

results_plot_wilcoxon_test = results_plot[~(results_plot['imputer'].isin(['None'])) & (results_plot['predictor'].isin(['HistGradientBoostingRegressor','XGBRegressor']))].copy()
groupby_cols = ['ratio_masked', 'predictor', 'imputer']
num_runs = results_plot_wilcoxon_test.groupby(groupby_cols).count()[f'prediction_score_{type_set}_{metric}'].max()
print(f'For a combinaison of {groupby_cols}, there are {num_runs} gains')
results_plot_wilcoxon_test = pd.DataFrame(results_plot_wilcoxon_test.groupby(groupby_cols).apply(lambda x: stats.wilcoxon(x[f'prediction_score_{type_set}_{metric}'], alternative='greater').pvalue).rename('wilcoxon_test_pvalue'))

results_plot_wilcoxon_test[results_plot_wilcoxon_test['wilcoxon_test_pvalue'] < 0.05]

```

If a p-value < 5%, the null hypothesis that the median is negative can be rejected at a confidence level of 5% in favor of the alternative that the median is greater than zero.


## The methods of multiple imputation are better than the methods of constant imputation (Mean/Shuffle)

```python
results_plot[f'prediction_score_{type_set}_{metric}_relative_percentage_gain']
```

```python
# metric = 'mae'
metric = 'wmape'

type_set = 'notnan'
# type_set = 'nan'

results_plot_ = results_plot[~(results_plot['hole_generator'].isin(['None']))].copy()

results_plot_[f'prediction_score_{type_set}_{metric}_relative_percentage_gain'] = results_plot_.apply(lambda x: imppred.get_relative_score(x, results_plot, col=f'prediction_score_{type_set}_{metric}', method='relative_percentage_gain', ref_imputer='ImputerMedian'), axis=1)

results_plot_[f'prediction_score_{type_set}_{metric}_gain'] = results_plot_.apply(lambda x: imppred.get_relative_score(x, results_plot, col=f'prediction_score_{type_set}_{metric}', method='gain', ref_imputer='ImputerMedian'), axis=1)
results_plot_[f'prediction_score_{type_set}_{metric}_gain_count'] = results_plot_.apply(lambda x: 1 if x[f'prediction_score_{type_set}_{metric}_gain'] > 0 else 0, axis=1)

results_plot_[f'prediction_score_{type_set}_{metric}_gain_ratio'] = results_plot_[f'prediction_score_{type_set}_{metric}_gain_count']/num_runs
```

```python
# model = 'HistGradientBoostingRegressor'
model = 'XGBRegressor'

# metric = 'mae_gain_ratio'
metric = 'wmape_gain_ratio'

type_set = "test_set_not_nan"
# type_set = "test_set_with_nan"

fig = imppred.plot_bar(
    results_plot_[(results_plot_['predictor'].isin([model]))
                 & ~(results_plot_['imputer'].isin(['None']))
                 ],
    col_displayed=("prediction_score", type_set, metric),
    cols_grouped=['hole_generator', 'ratio_masked', 'imputer'],
    add_annotation=True,
    add_confidence_interval=False,
    agg_func=pd.DataFrame.sum)

# fig.update_layout(title=f"Ratio of runs (over {num_trial * num_dataset} runs = {num_trial} trials x {num_dataset} datasets) where a gain of prediction performance <br>is found for {model}. Evaluation based on WMAPE computed on imputed test sets.")
fig.update_layout(title=f"Ratio of runs (over {num_trial * num_dataset} runs = {num_trial} trials x {num_dataset} datasets) where a gain of prediction performance <br>is found for {model}. Evaluation based on WMAPE computed on complete test sets.")
fig.update_xaxes(title="Types and Ratios of missing values")
fig.update_yaxes(title="Ratio of runs")
fig.update_layout(height=400, width=1000)
fig
```

```python
# model = 'HistGradientBoostingRegressor'
# model = 'XGBRegressor'
model = 'Ridge'

# metric = "mae_relative_percentage_gain"
# metric = "wmape_gain"
metric = "wmape_relative_percentage_gain"

type_set = "test_set_not_nan"
# type_set = "test_set_with_nan"

fig = imppred.plot_bar(
    results_plot_[(results_plot_['predictor'].isin([model]))
                 & ~(results_plot_['imputer'].isin(['None']))
                # & (results_plot_['dataset'].isin(['MiamiHousing2016', 'medical_charges']))
                 ],
    col_displayed=("prediction_score", type_set, metric),
    cols_grouped=['dataset', 'ratio_masked', 'imputer'],
    add_annotation=False,
    add_confidence_interval=True,
    confidence_level=0.95,
    agg_func=pd.DataFrame.mean,
    yaxes_type='log')

# fig.update_layout(title=f"Mean relative percentage gain of prediction performance over {num_trial} trials, for {model}.<br>Evaluation based on WMAPE computed on imputed test sets.")
fig.update_layout(title=f"Mean relative percentage gain of prediction performance over {num_trial} trials, for {model}.<br>Evaluation based on WMAPE computed on complete test sets.")
fig.update_xaxes(title="Datasets and Ratios of missing values")
fig.update_yaxes(title="(WMAPE(P) - WMAPE(I+P))/WMAPE(P)")
fig.update_layout(height=400, width=1000)
fig
```

## Which imputer achieves the best imputation performance, and at what ratio of missing values?


### Mean scores over all features

```python
# metric = 'mae'
metric = 'wmape'

type_set = 'notnan'
# type_set = 'nan'

results_plot_ = results_plot[~(results_plot['imputer'].isin(['None']))].copy()

results_plot_[f'imputation_score_{metric}_rank_train_set'] = results_plot_.groupby(['dataset', 'n_fold', 'hole_generator', 'ratio_masked', 'n_mask', 'predictor'])[f'imputation_score_{metric}_train_set'].rank()
results_plot_[f'imputation_score_{metric}_rank_test_set'] = results_plot_.groupby(['dataset', 'n_fold', 'hole_generator', 'ratio_masked', 'n_mask', 'predictor'])[f'imputation_score_{metric}_test_set'].rank()

results_plot_[f'prediction_score_{type_set}_{metric}_rank'] = results_plot_.groupby(['dataset', 'n_fold', 'hole_generator', 'ratio_masked', 'n_mask', 'predictor'])[f'prediction_score_{type_set}_{metric}'].rank()
```

```python
# metric = "dist_corr_pattern"
metric = "wmape"

type_set = "test_set_not_nan"
# type_set = "test_set_with_nan"

fig = imppred.plot_bar(
    results_plot_[~(results_plot_['imputer'].isin(['None']))
                 & (results_plot_['dataset'].isin(['Bike_Sharing_Demand', 'medical_charges']))
                 ],
    col_displayed=("imputation_score", "test_set", metric),
    # col_displayed=("prediction_score", "test_set_not_nan", metric),
    cols_grouped=['dataset', 'ratio_masked', 'imputer'],
    add_annotation=False,
    add_confidence_interval=True,
    confidence_level=0.95,
    agg_func=pd.DataFrame.mean,
    yaxes_type='log')

# fig.update_layout(title=f"Average imputation performance over {num_trial} trials.<br>Evaluation based on WMAPE computed on imputed test sets.")
fig.update_layout(title=f"Average imputation performance over {num_trial} trials.<br>Evaluation based on WMAPE computed on complete test sets.")
fig.update_yaxes(title="WMAPE(I)")

# fig.update_layout(title=f"Average prediction performance over {num_trial} trials.<br>Evaluation based on WMAPE computed on imputed test sets.")
# fig.update_yaxes(title="WMAPE(P)")

fig.update_xaxes(title="Datasets and Ratios of missing values")
fig.update_layout(height=400, width=1000)
fig
```

```python
# metric = 'mae'
metric = 'wmape'

fig = imppred.plot_bar(
    results_plot_,
    cols_displayed=(("imputation_score", "test_set", f"{metric}_rank"),
                   ("imputation_score", "train_set", f"{metric}_rank")),
    cols_grouped=['ratio_masked', 'imputer'],
    add_annotation=True,
    add_confidence_interval=False,
    agg_func=pd.DataFrame.mean)


fig.update_layout(title=f"Average imputation performance ranks of {num_imputer} imputers with {num_dataset} datasets and {num_fold * num_mask} trials")
fig.update_xaxes(title=f"Predictors and ratios of nan")
fig.update_yaxes(title="Average rank")
fig.update_layout(height=400, width=2000)
fig
```

```python
# metric = 'mae_rank'
metric = 'wmape_rank'

type_set = "test_set_not_nan"
# type_set = "test_set_with_nan"

fig = imppred.plot_bar(
    results_plot_,
    # col_displayed=("imputation_score", "test_set", metric),
    col_displayed=("prediction_score", type_set, metric),
    cols_grouped=['ratio_masked', 'imputer'],
    add_annotation=True,
    add_confidence_interval=False,
    confidence_level=0.95,
    agg_func=pd.DataFrame.mean,
    yaxes_type='log')

# fig.update_layout(title=f"Average ranks of imputeurs for {num_dataset *num_trial *num_predictor *num_ratio_masked} rounds ({num_dataset} datasets * {num_ratio_masked} ratios of nan * {num_predictor} predictors * {num_trial} trials).<br>Evaluation based on prediction performance WMAPE computed on imputed test sets.")
# fig.update_layout(title=f"Average ranks of imputeurs for {num_dataset *num_trial *num_predictor *num_ratio_masked} rounds ({num_dataset} datasets * {num_ratio_masked} ratios of nan * {num_predictor} predictors * {num_trial} trials).<br>Evaluation based on imputation performance WMAPE computed on imputed test sets.")

fig.update_layout(title=f"Average ranks of imputeurs for {num_dataset *num_trial *num_predictor *num_ratio_masked} rounds ({num_dataset} datasets * {num_ratio_masked} ratios of nan * {num_predictor} predictors * {num_trial} trials).<br>Evaluation based on prediction performance WMAPE computed on complete test sets.")

fig.update_xaxes(title=f"Ratios of nan")
fig.update_yaxes(title="Average rank")
fig.update_layout(height=400, width=1000)
fig
```

### Critical difference diagram of average score ranks

```python
color_palette = dict([(key, value) for key, value in zip(results_plot_avg_rank.index, np.random.rand(len(results_plot_avg_rank),3))])
```

```python
metric = 'wmape'
# type_set = 'nan'
type_set = 'notnan'

ratio_masked = 0.7
results_plot_critial_difference = results_plot_[~(results_plot_['hole_generator'].isin(['None'])) & ~(results_plot_['imputer'].isin(['None'])) & (results_plot_['ratio_masked'].isin([ratio_masked]))].copy()

# results_plot_critial_difference = results_plot_[~(results_plot_['hole_generator'].isin(['None'])) & ~(results_plot_['imputer'].isin(['None']))].copy()

results_plot_critial_difference['imputer_predictor'] = results_plot_critial_difference['imputer'] #+ '_' + results_plot_critial_difference['predictor']
results_plot_avg_rank = results_plot_critial_difference.groupby(['imputer_predictor'])[f'prediction_score_{type_set}_{metric}_rank'].mean()

# color_palette = dict([(key, value) for key, value in zip(results_plot_avg_rank.index, np.random.rand(len(results_plot_avg_rank),3))])

results_plot_posthoc_conover_friedman = results_plot_critial_difference.groupby('imputer_predictor')[f'prediction_score_{type_set}_{metric}'].apply(list)
results_plot_posthoc_conover_friedman = results_plot_posthoc_conover_friedman[~results_plot_posthoc_conover_friedman.index.isin(['None_Ridge'])]
imputer_predictor_names = results_plot_posthoc_conover_friedman.index

results_plot_posthoc_conover_friedman = np.array(list(results_plot_posthoc_conover_friedman.values)).T
results_plot_posthoc_conover_friedman = sp.posthoc_conover_friedman(results_plot_posthoc_conover_friedman)

results_plot_posthoc_conover_friedman.index = imputer_predictor_names
results_plot_posthoc_conover_friedman.columns = imputer_predictor_names

plt.figure(figsize=(7, 2))
# plt.title(f'Average ranks for prediction performance')
plt.title(f'Average ranks for prediction performance, ratio of nan = {ratio_masked}')
_ = sp.critical_difference_diagram(results_plot_avg_rank,
                                   results_plot_posthoc_conover_friedman,
                                   color_palette=color_palette,)
```

### Observe separately each feature

```python
# def scale_score(row, score_col, metric, data_mean):
#     scores_in = row[score_col][metric]
#     scores_out = []
#     for feature in scores_in:
#         scores_out.append(scores_in[feature]/np.abs(data_mean[feature]))
#     return np.mean(scores_out)

# score_col_in = 'imputation_scores_trainset'
# score_col_out = 'imputation_score_mae_scaled_train_set'

# # score_col_in = 'imputation_scores_testset'
# # score_col_out = 'imputation_score_mae_scaled_test_set'

# metric = 'imputation_score_mae'

# results_plot[score_col_out] = np.NAN
# for dataset_name in results_plot['dataset'].unique():
#     print(dataset_name)
#     dataset = load_dataset("inria-soda/tabular-benchmark", data_files=f"reg_num/{dataset_name}.csv")
#     data_mean = dataset["train"].to_pandas().abs().mean()
#     index = results_plot[(results_plot['dataset']==dataset_name) & (results_plot['imputer']!='None')].index
#     results_plot.loc[index, score_col_out] = results_plot.loc[index, :].apply(lambda x: scale_score(x, score_col = score_col_in, metric = metric, data_mean = data_mean), axis=1)

#     # print(results_plot_features[results_plot_features['dataset']==dataset_name]['imputation_score_mae_scaled_train_set'].mean())
```

```python
# results_plot_ = results_plot[~(results_plot['imputer'].isin(['None']))].copy()

# results_plot_['imputation_score_mae_scaled_rank_train_set'] = results_plot_.groupby(['dataset', 'n_fold', 'hole_generator', 'ratio_masked', 'n_mask', 'predictor'])['imputation_score_mae_scaled_train_set'].rank()
# results_plot_['imputation_score_mae_scaled_rank_test_set'] = results_plot_.groupby(['dataset', 'n_fold', 'hole_generator', 'ratio_masked', 'n_mask', 'predictor'])['imputation_score_mae_scaled_test_set'].rank()

# fig = imppred.plot_bar(
#     results_plot_,
#     cols_displayed=(("imputation_score", "test_set", "mae_scaled"),
#                    ("imputation_score", "train_set", "mae_scaled")),
#     cols_grouped=['ratio_masked', 'imputer'],
#     add_annotation=True,
#     add_confidence_interval=False,
#     agg_func=pd.DataFrame.mean)


# # fig.update_layout(title=f"Average imputation performance ranks of {num_imputer} imputers with {num_dataset} datasets and {num_fold * num_mask} trials")
# # fig.update_yaxes(title="Average rank")

# fig.update_layout(title=f"Average imputation performance score of {num_imputer} imputers with {num_dataset} datasets and {num_fold * num_mask} trials")
# fig.update_yaxes(title="Average score")

# fig.update_xaxes(title=f"Imputers and ratios of nan")
# fig.update_layout(height=500, width=2000)
# fig
```

## Which pair imputer-predictor achieves the best imputation performance, and at what ratio of missing values?

```python
metric = 'wmape'
type_set = 'notnan'
# type_set = 'nan'

results_plot[f'prediction_score_{type_set}_{metric}_rank'] = results_plot.groupby(['dataset', 'n_fold', 'hole_generator', 'ratio_masked', 'n_mask'])[f'prediction_score_{type_set}_{metric}'].rank()
```

```python
# model = 'HistGradientBoostingRegressor'
# model = 'XGBRegressor'
# model = 'Ridge'

# metric = 'mae_rank'
metric = 'wmape_rank'

type_set = "test_set_not_nan"
# type_set = "test_set_with_nan"

fig = imppred.plot_bar(
    results_plot[
        ~(results_plot['hole_generator'].isin(['None']))
        # (results_plot['predictor'].isin([model ]))
        ],
    col_displayed=("prediction_score", type_set, metric),
    cols_grouped=['predictor','ratio_masked', 'imputer'],
    add_annotation=True,
    add_confidence_interval=False,
    agg_func=pd.DataFrame.mean)

# fig.update_layout(title=f"Average ranks of {num_imputer * num_predictor} pairs imputer-predictor for {num_dataset * num_trial * num_ratio_masked} rounds ({num_dataset} datasets * {num_ratio_masked} ratios of nan * {num_trial} trials).<br>Evaluation based on prediction performance WMAPE computed on imputed test sets.")

fig.update_layout(title=f"Average ranks of {num_imputer * num_predictor} pairs imputer-predictor for {num_dataset * num_trial * num_ratio_masked} rounds ({num_dataset} datasets * {num_ratio_masked} ratios of nan * {num_trial} trials).<br>Evaluation based on prediction performance WMAPE computed on complete test sets.")

# fig.update_xaxes(title=f"Ratios of nan with predictor={model}")
fig.update_xaxes(title=f"Predictors and ratios of nan")
fig.update_yaxes(title="Average rank")
fig.update_layout(height=500, width=2000)
fig
```

### Critical difference diagram of average score ranks

```python
color_palette = dict([(key, value) for key, value in zip(results_plot_avg_rank.index, np.random.rand(len(results_plot_avg_rank),3))])
```

```python
metric = 'wmape'
type_set = 'nan'

ratio_masked = 0.7
results_plot_critial_difference = results_plot[~(results_plot['hole_generator'].isin(['None'])) & ~(results_plot['imputer'].isin(['None'])) & (results_plot['ratio_masked'].isin([ratio_masked]))].copy()

# results_plot_critial_difference = results_plot[~(results_plot['hole_generator'].isin(['None'])) & ~(results_plot['imputer'].isin(['None']))].copy()

results_plot_critial_difference['imputer_predictor'] = results_plot_critial_difference['imputer'] + '_' + results_plot_critial_difference['predictor']
results_plot_avg_rank = results_plot_critial_difference.groupby(['imputer_predictor'])[f'prediction_score_{type_set}_{metric}_rank'].mean()

# color_palette = dict([(key, value) for key, value in zip(results_plot_avg_rank.index, np.random.rand(len(results_plot_avg_rank),3))])

results_plot_posthoc_conover_friedman = results_plot_critial_difference.groupby('imputer_predictor')[f'prediction_score_{type_set}_{metric}'].apply(list)
results_plot_posthoc_conover_friedman = results_plot_posthoc_conover_friedman[~results_plot_posthoc_conover_friedman.index.isin(['None_Ridge'])]
imputer_predictor_names = results_plot_posthoc_conover_friedman.index

results_plot_posthoc_conover_friedman = np.array(list(results_plot_posthoc_conover_friedman.values)).T
results_plot_posthoc_conover_friedman = sp.posthoc_conover_friedman(results_plot_posthoc_conover_friedman)

results_plot_posthoc_conover_friedman.index = imputer_predictor_names
results_plot_posthoc_conover_friedman.columns = imputer_predictor_names

plt.figure(figsize=(10, 5))
# plt.title(f'Average ranks for prediction performance')
plt.title(f'Average ranks for prediction performance, ratio of nan = {ratio_masked}')
_ = sp.critical_difference_diagram(results_plot_avg_rank,
                                   results_plot_posthoc_conover_friedman,
                                   color_palette=color_palette,)
```

## Is there a correlation between prediction performance and imputation performance ?

```python
print(results['dataset'].unique())

metric = 'wmape'
type_set = 'nan'
dataset = 'medical_charges'

fig = imppred.plot_scatter(results_plot, cond={'dataset':dataset}, col_x=f'imputation_score_{metric}_train_set', col_y=f'prediction_score_{type_set}_{metric}')
fig.update_layout(legend_title="Nan ratio")
fig.update_layout(title=f"Performance scores of all pairs imputer-predictor for {num_trial} trials, on the dataset {dataset}")
fig.update_xaxes(title=f"WMAPE for imputation on the train set")
fig.update_yaxes(title="WMAPE for prediction on the test set without nan")
fig.update_layout(height=500, width=1000)

fig
```

```python
print(results['predictor'].unique())

metric = 'wmape'
type_set = 'nan'
predictor = 'XGBRegressor'

fig = imppred.plot_scatter(results_plot, cond={'predictor':predictor}, col_x=f'imputation_score_{metric}_train_set', col_y=f'prediction_score_{type_set}_{metric}')
fig.update_layout(legend_title="Nan ratio")
fig.update_layout(title=f"MAEs of all pairs imputer-predictor for {num_trial} trials, on the predictor {predictor}")
fig.update_xaxes(title=f"MAE for imputation on the train set")
fig.update_yaxes(title="MAE for prediction on the test set without nan")
fig.update_layout(height=500, width=1000)

fig
```

```python
# groupby_col = 'predictor'
# groupby_col = 'dataset'
groupby_col = 'ratio_masked'
# groupby_col = 'imputer'

metric = 'wmape'
type_set = 'nan'

results_plot_ = results_plot[~(results_plot['imputer'].isin(['None']))].copy()
# score_cols = [f'imputation_score_{metric}_train_set', f'imputation_score_{metric}_test_set',f'prediction_score_{type_set}_{metric}', f'prediction_score_nan_{metric}']
score_cols = [f'imputation_score_{metric}_train_set', f'imputation_score_{metric}_test_set',f'prediction_score_{type_set}_{metric}', f'prediction_score_nan_{metric}']
if groupby_col is None:
    results_corr = results_plot_[score_cols].corr(method='spearman')
else:
    results_corr = results_plot_.groupby(groupby_col)[score_cols].corr(method='spearman')
    print(f'#num_scores = {results_plot_.groupby(groupby_col).count().max().max()}')

multi_index_columns = [
    ('imputation', metric, 'train_set'),
    ('imputation', metric, 'test_set'),
    ('prediction', metric, 'test_set_not_nan'),
    ('prediction', metric, 'test_set_with_nan'),
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

if groupby_col is None:
    results_corr.index.names = ['task', 'metric', 'set']
    reorder_levels = ['task', 'metric', 'set']
    hide_indices_test = (slice(None), slice(None), 'test_set')
    hide_indices_train = (slice(None), slice(None), 'train_set')
else:
    results_corr.index.names = [groupby_col, 'task', 'metric', 'set']
    reorder_levels = ['task', 'metric', groupby_col, 'set']
    hide_indices_test = (slice(None), slice(None), slice(None), 'test_set')
    hide_indices_train = (slice(None), slice(None), slice(None), 'train_set')

results_corr.columns.names = ['task', 'metric', 'set']
results_corr_plot = results_corr.xs('imputation', level=1, drop_level=False)[[('prediction', metric, 'test_set_not_nan'), ('prediction', metric, 'test_set_with_nan'),]].reorder_levels(reorder_levels)

def mask_values(val):
    return f"opacity: {0}"

results_corr_plot\
.style.applymap(
    mask_values,
    subset=(
        hide_indices_test,
        ('prediction', metric, 'test_set_not_nan')
    ),
).applymap(
    mask_values,
    subset=(
        hide_indices_train,
        ('prediction', metric, 'test_set_with_nan')
    ),
)
```

## Dataset

```python
metric = 'wmape'
type_set = 'nan'

results_plot_wilcoxon_test = results_plot[~(results_plot['imputer'].isin(['None']))
                                          & (results_plot['predictor'].isin(['HistGradientBoostingRegressor','XGBRegressor']))].copy()
groupby_cols = ['size_test_set', 'predictor', 'imputer']
num_runs = results_plot_wilcoxon_test.groupby(groupby_cols).count()[f'prediction_score_{type_set}_{metric}_gain'].max()
print(f'For a combinaison of {groupby_cols}, there are {num_runs} gains')
results_plot_wilcoxon_test = pd.DataFrame(results_plot_wilcoxon_test.groupby(groupby_cols).apply(lambda x: stats.wilcoxon(x[f'prediction_score_{type_set}_{metric}_gain'], alternative='greater').pvalue).rename('wilcoxon_test_pvalue'))

```

```python
results_plot_wilcoxon_test['wilcoxon_test_pvalue_count'] = results_plot_wilcoxon_test['wilcoxon_test_pvalue'].apply(lambda x: x < 0.05)
```

```python
fig = go.Figure()

for value in results_plot_wilcoxon_test.index.get_level_values('imputer').unique():
    df_plot_ = results_plot_wilcoxon_test.xs(value, level='imputer')
    fig.add_trace(
        go.Scatter(
            x=df_plot_.index.get_level_values(level='size_test_set'),
            y=df_plot_['wilcoxon_test_pvalue'],
            name=value,
            mode="markers",
            )
        )

fig
```

```python
# model = 'HistGradientBoostingRegressor'
# model = 'XGBRegressor'
# model = 'Ridge'

# metric = "mae_rank"
metric = "wmape_rank"

fig = imppred.plot_bar(
    results_plot[
        ~(results_plot['hole_generator'].isin(['None']))
        # (results_plot['predictor'].isin([model ]))
        ],
    col_displayed=("prediction_score", "test_set_not_nan", metric),
    cols_grouped=['dataset', 'predictor', 'imputer'],
    add_annotation=True,
    add_confidence_interval=False,
    agg_func=pd.DataFrame.mean)


fig.update_layout(title=f"Average prediction performance ranks of {num_imputer * num_predictor} pairs imputer-predictor for {num_dataset} datasets and {num_fold * num_mask} trials")
# fig.update_xaxes(title=f"Ratios of nan with predictor={model}")
fig.update_xaxes(title=f"Predictors and ratios of nan")
fig.update_yaxes(title="Average rank")
fig.update_layout(height=500, width=2000)
fig
```

### Find best features

```python
from sklearn.ensemble import GradientBoostingRegressor

k_top_features = []
for dataset_name in results_plot['dataset'].unique():
    print(dataset_name)
    dataset = load_dataset("inria-soda/tabular-benchmark", data_files=f"reg_num/{dataset_name}.csv")
    df_data = dataset["train"].to_pandas()

    columns = df_data.columns.to_list()
    df_data_x = df_data[columns[:-1]]
    df_data_y = df_data[columns[-1]]

    model = GradientBoostingRegressor().fit(df_data_x,df_data_y)

    feature_importances = dict([(key, value) for key, value in zip(columns, model.feature_importances_)])

    print(feature_importances)
    break
```

```python

```
