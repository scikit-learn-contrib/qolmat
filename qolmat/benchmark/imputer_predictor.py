from typing import Dict, List
import mlflow
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.model_selection import KFold

from qolmat.benchmark.missing_patterns import _HoleGenerator
from qolmat.benchmark import metrics as _imputation_metrics
from qolmat.imputations.imputers import _BaseImputer
from rdt.hyper_transformer import HyperTransformer


class BenchmarkImputationPrediction:
    def __init__(
        self,
        imputation_metrics: List = ["mae"],
        prediction_metrics: List = ["mae"],
        n_splits: int = 2,
    ):

        self.imputation_metrics = imputation_metrics
        self.prediction_metrics = prediction_metrics
        self.n_splits = n_splits

    def compare(
        self,
        df_data: pd.DataFrame,
        file_path: str,
        hole_generators: List[_HoleGenerator],
        imputation_pipelines,
        target_prediction_pipeline_pairs,
        # imputation_columns: List[str] = []
    ):

        # self.imputation_columns = imputation_columns

        self.columns = df_data.columns.to_list()
        self.columns_numerical = df_data.select_dtypes(include=np.number).columns.tolist()
        self.columns_categorical = [
            col for col in df_data.columns.to_list() if col not in self.columns_numerical
        ]

        list_benchmark = []
        for idx_fold, (idx_train, idx_test) in enumerate(
            KFold(n_splits=self.n_splits).split(df_data)
        ):
            df_train = df_data.iloc[idx_train, :]
            df_test = df_data.iloc[idx_test, :]
            # Suppose that all categories are known
            for target_column, prediction_pipelines in target_prediction_pipeline_pairs.items():
                if target_column in self.columns_numerical:
                    prediction_task = "regression"
                if target_column in self.columns_categorical:
                    prediction_task = "classification"
                feature_columns = [col for col in self.columns if col != target_column]
                df_train_x = df_train[feature_columns]
                df_train_y = df_train[[target_column]]
                df_test_x = df_test[feature_columns]
                df_test_y = df_test[[target_column]]

                for hole_generator in hole_generators:
                    hole_generator.subset = feature_columns
                    hole_generator.n_splits = self.n_splits
                    for idx_mask, (df_mask_train, df_mask_test) in enumerate(
                        zip(hole_generator.split(df_train_x), hole_generator.split(df_test_x))
                    ):
                        for imputation_pipeline in imputation_pipelines:
                            transformer_imputation = imputation_pipeline["transformer"]
                            imputer = imputation_pipeline["imputer"]

                            transformer_imputation.fit(df_data)
                            # df_train_x
                            df_train_x_transformed = transformer_imputation.transform_subset(
                                df_train_x
                            )
                            df_train_x_transformed_corrupted = df_train_x_transformed.copy()
                            df_train_x_transformed_corrupted[df_mask_train] = np.nan

                            df_train_x_transformed_imputed = imputer.fit_transform(
                                df_train_x_transformed_corrupted
                            )
                            df_train_x_reversed_imputed = (
                                transformer_imputation.reverse_transform_subset(
                                    df_train_x_transformed_imputed
                                )
                            )

                            (
                                dict_imp_score_mean_train,
                                dict_imp_scores_train,
                            ) = self.get_imputation_scores_by_dataframe(
                                df_train_x, df_train_x_reversed_imputed, df_mask_train
                            )

                            # df_test_x
                            df_test_x_transformed = transformer_imputation.transform_subset(
                                df_test_x
                            )
                            df_test_x_transformed_corrupted = df_test_x_transformed.copy()
                            df_test_x_transformed_corrupted[df_mask_test] = np.nan

                            df_test_x_transformed_imputed = imputer.fit_transform(
                                df_test_x_transformed_corrupted
                            )
                            df_test_x_reversed_imputed = (
                                transformer_imputation.reverse_transform_subset(
                                    df_test_x_transformed_imputed
                                )
                            )

                            (
                                dict_imp_score_mean_test,
                                dict_imp_scores_test,
                            ) = self.get_imputation_scores_by_dataframe(
                                df_test_x, df_test_x_reversed_imputed, df_mask_test
                            )

                            for prediction_pipeline in prediction_pipelines:
                                transformer_prediction = prediction_pipeline["transformer"]
                                predictor = prediction_pipeline["predictor"]

                                if transformer_prediction is not None:
                                    transformer_prediction.fit(df_data)
                                    df_train_x_transformed_imputed = (
                                        transformer_prediction.transform_subset(
                                            df_train_x_reversed_imputed
                                        )
                                    )
                                    df_train_y_transformed = (
                                        transformer_prediction.transform_subset(df_train_y)
                                    )
                                    df_test_x_transformed = (
                                        transformer_prediction.transform_subset(df_test_x)
                                    )
                                else:
                                    df_train_x_transformed_imputed = df_train_x_reversed_imputed
                                    df_train_y_transformed = df_train_y
                                    df_test_x_transformed = df_test_x

                                predictor = predictor.fit(
                                    df_train_x_transformed_imputed,
                                    df_train_y_transformed[target_column],
                                )
                                df_test_y_transformed_predicted = pd.DataFrame(
                                    predictor.predict(df_test_x_transformed),
                                    columns=[target_column],
                                    index=df_test_y.index,
                                )
                                df_test_y_reserved_predicted = (
                                    transformer_prediction.reverse_transform_subset(
                                        df_test_y_transformed_predicted
                                    )
                                )

                                (
                                    dict_pred_score_mean,
                                    dict_pred_scores,
                                ) = self.get_prediction_scores_by_column(
                                    df_test_y, df_test_y_reserved_predicted
                                )

                                tran_imp_name = transformer_imputation.__class__.__name__
                                tran_pre_name = transformer_prediction.__class__.__name__
                                row_benchmark = {
                                    "n_fold": idx_fold,
                                    "n_mask": idx_mask,
                                    "hole_generator": hole_generator.__class__.__name__,
                                    "ratio_masked": hole_generator.ratio_masked,
                                    "transformer_imputation": tran_imp_name,
                                    "imputer": imputer.__class__.__name__,
                                    "target_column": target_column,
                                    "prediction_task": prediction_task,
                                    "transformer_prediction": tran_pre_name,
                                    "predictor": predictor.__class__.__name__,
                                }
                                dict_imp_score_mean_train_ = dict(
                                    (f"{k}_train_set", v)
                                    for k, v in dict_imp_score_mean_train.items()
                                )
                                dict_imp_score_mean_test_ = dict(
                                    (f"{k}_test_set", v)
                                    for k, v in dict_imp_score_mean_test.items()
                                )
                                row_benchmark = {**row_benchmark, **dict_imp_score_mean_train_}
                                row_benchmark = {**row_benchmark, **dict_imp_score_mean_test_}
                                row_benchmark = {**row_benchmark, **dict_pred_score_mean}

                                row_benchmark["imputation_scores_trainset"] = dict_imp_scores_train
                                row_benchmark["imputation_scores_testset"] = dict_imp_scores_test
                                row_benchmark["prediction_scores"] = dict_pred_scores

                                list_benchmark.append(row_benchmark)
                                df_benchmark = pd.DataFrame(list_benchmark)
                                with open(file_path, "wb") as handle:
                                    pickle.dump(
                                        df_benchmark, handle, protocol=pickle.HIGHEST_PROTOCOL
                                    )

        return df_benchmark

    def get_imputation_scores_by_dataframe(self, df_true, df_imputed, df_mask):
        dict_score_mean = {}
        dict_scores = {}

        for metric in self.imputation_metrics:
            func_metric = _imputation_metrics.get_metric(metric)
            score_by_col = func_metric(df_true, df_imputed, df_mask)
            dict_scores[f"imputation_score_{metric}"] = score_by_col.to_dict()
            dict_score_mean[f"imputation_score_{metric}"] = score_by_col.mean()
        return dict_score_mean, dict_scores

    def get_prediction_scores_by_column(self, df_true, df_imputed):
        dict_score_mean = {}
        dict_scores = {}

        df_mask = df_true.notnull()
        for metric in self.prediction_metrics:
            func_metric = _imputation_metrics.get_metric(metric)
            try:
                score_by_col = func_metric(df_true, df_imputed, df_mask)
            except Exception:
                score_by_col = pd.Series(
                    [np.nan for col in df_true.columns], index=df_true.columns
                )

            dict_scores[f"prediction_score_{metric}"] = score_by_col.to_dict()
            dict_score_mean[f"prediction_score_{metric}"] = score_by_col.mean()
        return dict_score_mean, dict_scores


def visualize_mlflow(df, exp_name, cols_mean_on=["n_fold", "n_mask"]):
    cols_full_scores = [col for col in df.columns if "_scores" in col]
    metrics = [col for col in df.columns if "_score_" in col]
    cols_groupby = [
        col for col in df.columns if col not in metrics + cols_mean_on + cols_full_scores
    ]
    df_groupby = df.groupby(cols_groupby)[metrics].mean()

    experiment_id = mlflow.create_experiment(name=exp_name)
    num_index = np.prod([len(df[col].unique()) for col in cols_mean_on])
    for idx in df_groupby.index:
        dict_settings = dict(zip(df_groupby.index.names, idx))
        with mlflow.start_run(
            experiment_id=experiment_id, run_name=dict_settings["target_column"]
        ) as run:
            query = " and ".join([f"{k} == {repr(v)}" for k, v in dict_settings.items()])

            for col in cols_mean_on:
                dict_settings[col] = len(df[col].unique())
                dict_settings[f"{col}_values"] = df[col].unique()
            dict_scores = df_groupby.loc[idx][metrics].to_dict()

            mlflow.log_params(dict_settings)
            mlflow.log_metrics(dict_scores)

            df_query = df.query(query)
            for col_full_scores in cols_full_scores:
                dict_full_scores = df_query[col_full_scores].values
                list_scores = []
                list_indices = []
                num_index = 0
                for dict_full_score_metric in dict_full_scores:
                    df_full_score_metric = pd.DataFrame(
                        list(dict_full_score_metric.values()),
                        index=list(dict_full_score_metric.keys()),
                    ).T
                    num_index = df_full_score_metric.shape[0]
                    list_scores.append(df_full_score_metric)

                list_indices = [df_query[cols_mean_on] for i in range(num_index)]
                df_scores = pd.concat(list_scores)
                df_indices = pd.concat(list_indices)
                df_indices.index = df_scores.index

                df_scores = pd.concat([df_scores, df_indices], axis=1)
                df_scores.index.name = "columns"
                df_scores = df_scores.set_index(cols_mean_on, append=True)

                file_path_html = Path(f"{run.info.artifact_uri[7:]}/{col_full_scores}.html")
                file_path_html.parent.mkdir(parents=True, exist_ok=True)
                df_scores.to_html(file_path_html)
                mlflow.log_artifact(file_path_html)


def get_benchmark_aggregate(df, cols_mean_on=["n_fold", "n_mask"]):
    cols_full_scores = [col for col in df.columns if "scores" in col]
    metrics = [col for col in df.columns if "_score_" in col]
    cols_groupby = [
        col for col in df.columns if col not in metrics + cols_mean_on + cols_full_scores
    ]
    df_groupby = df.groupby(cols_groupby)[metrics].mean()
    df_groupby_max = df.groupby(cols_groupby)[metrics].max()
    df_groupby_min = df.groupby(cols_groupby)[metrics].min()

    list_scores = []
    for idx in df_groupby.index:
        dict_settings = dict(zip(df_groupby.index.names, idx))
        for col in cols_mean_on:
            dict_settings[col] = len(df[col].unique())
            dict_settings[f"{col}_values"] = df[col].unique()

        for col in metrics:
            dict_settings[f"{col}_interval"] = [
                df_groupby_min.loc[idx][col],
                df_groupby_max.loc[idx][col],
            ]

        dict_scores = df_groupby.loc[idx][metrics].to_dict()
        list_scores.append({**dict_settings, **dict_scores})

    return pd.DataFrame(list_scores)


def visualize_plotly(df, selected_columns):
    columns_numerical = df.select_dtypes(include=np.number).columns.tolist()
    columns_categorical = [col for col in df.columns.to_list() if col not in columns_numerical]

    dimensions = []
    for col in selected_columns:
        if col in columns_categorical:
            dfg = pd.DataFrame({col: df[col].unique()})
            dfg[f"{col}_dummy"] = dfg.index
            df = pd.merge(df, dfg, on=col, how="left")

    df = df[selected_columns + [col for col in df.columns if "dummy" in col]]
    df = df.dropna()

    for col in selected_columns:
        if col in columns_categorical:
            dfg = pd.DataFrame({col: df[col].unique()})
            dfg[f"{col}_dummy"] = dfg.index
            dimensions.append(
                dict(
                    range=[0, df[f"{col}_dummy"].max()],
                    tickvals=dfg[f"{col}_dummy"],
                    ticktext=dfg[f"{col}"],
                    label=col,
                    values=df[f"{col}_dummy"],
                ),
            )
        else:
            dimensions.append(
                dict(
                    range=[df[f"{col}"].min(), df[f"{col}"].max()], label=col, values=df[f"{col}"]
                ),
            )
    fig = go.Figure(data=go.Parcoords(dimensions=dimensions))

    return fig
