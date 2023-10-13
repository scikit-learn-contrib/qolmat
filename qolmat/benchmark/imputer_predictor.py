from typing import Dict, List, Tuple
import mlflow
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import tqdm
import re
import plotly.graph_objects as go

from sklearn.model_selection import KFold

from qolmat.benchmark.missing_patterns import _HoleGenerator
from qolmat.benchmark import metrics as _imputation_metrics
from qolmat.imputations.imputers import _Imputer
from rdt.hyper_transformer import HyperTransformer


class BenchmarkImputationPrediction:
    def __init__(
        self,
        imputation_metrics: List = ["mae"],
        prediction_metrics: List = ["mae"],
        n_folds: int = 2,
        n_masks: int = 2,
    ):

        self.imputation_metrics = imputation_metrics
        self.prediction_metrics = prediction_metrics
        self.n_folds = n_folds
        self.n_masks = n_masks

    def compare(
        self,
        df_data: pd.DataFrame,
        file_path: str,
        hole_generators: List[_HoleGenerator],
        imputation_pipelines,
        target_prediction_pipeline_pairs,
        imputation_columns: List[str] = [],
    ):
        self.columns = df_data.columns.to_list()
        self.columns_numerical = df_data.select_dtypes(include=np.number).columns.tolist()
        self.columns_categorical = [
            col for col in df_data.columns.to_list() if col not in self.columns_numerical
        ]
        if len(imputation_columns) == 0:
            self.imputation_columns = self.columns
        else:
            self.imputation_columns = imputation_columns

        list_benchmark = []
        for idx_fold, (idx_train, idx_test) in tqdm.tqdm(
            enumerate(KFold(n_splits=self.n_folds).split(df_data)), position=0, desc="benchmark"
        ):
            df_train = df_data.iloc[idx_train, :]
            df_test = df_data.iloc[idx_test, :]
            for target_column, prediction_pipelines in tqdm.tqdm(
                target_prediction_pipeline_pairs.items(),
                position=1,
                leave=True,
                desc=f"n_fold={idx_fold}",
            ):
                feature_columns = [col for col in self.columns if col != target_column]
                df_train_x = df_train[feature_columns]
                df_test_x = df_test[feature_columns]

                for hole_generator in tqdm.tqdm(
                    hole_generators, position=1, leave=False, desc=f"target_column={target_column}"
                ):
                    if hole_generator is not None:
                        hole_generator.subset = [
                            col for col in feature_columns if col in self.imputation_columns
                        ]
                        hole_generator.n_splits = self.n_masks
                        for idx_mask, (df_mask_train, df_mask_test) in enumerate(
                            zip(hole_generator.split(df_train_x), hole_generator.split(df_test_x))
                        ):
                            for imputation_pipeline in tqdm.tqdm(
                                imputation_pipelines, position=1, leave=False
                            ):
                                out_imputation, benchmark_imputation = self.benchmark_imputation(
                                    imputation_pipeline,
                                    feature_columns,
                                    df_data,
                                    df_train,
                                    df_test,
                                    df_mask_train,
                                    df_mask_test,
                                )

                                for prediction_pipeline in tqdm.tqdm(
                                    prediction_pipelines, position=1, leave=False
                                ):
                                    benchmark_prediction = self.benchmark_prediction(
                                        prediction_pipeline,
                                        target_column,
                                        df_data,
                                        df_train,
                                        df_test,
                                        out_imputation["df_train_x_reversed_imputed"],
                                        out_imputation["df_test_x_reversed_imputed"],
                                        df_mask_train,
                                        df_mask_test,
                                    )

                                    row_benchmark = self.get_row_benchmark(
                                        idx_fold,
                                        target_column,
                                        hole_generator,
                                        idx_mask,
                                        imputation_pipeline,
                                        prediction_pipeline,
                                        benchmark_imputation,
                                        benchmark_prediction,
                                    )

                                    list_benchmark.append(row_benchmark)
                                    df_benchmark = pd.DataFrame(list_benchmark)
                                    with open(file_path, "wb") as handle:
                                        pickle.dump(
                                            df_benchmark, handle, protocol=pickle.HIGHEST_PROTOCOL
                                        )
                    else:
                        for prediction_pipeline in tqdm.tqdm(
                            prediction_pipelines, position=1, leave=False
                        ):
                            benchmark_prediction = self.benchmark_prediction(
                                prediction_pipeline,
                                target_column,
                                df_data,
                                df_train,
                                df_test,
                                df_train_x,
                                df_test_x,
                                None,
                                None,
                            )

                            row_benchmark = self.get_row_benchmark(
                                idx_fold,
                                target_column,
                                None,
                                np.nan,
                                None,
                                prediction_pipeline,
                                None,
                                benchmark_prediction,
                            )

                            list_benchmark.append(row_benchmark)
                            df_benchmark = pd.DataFrame(list_benchmark)
                            with open(file_path, "wb") as handle:
                                pickle.dump(df_benchmark, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return df_benchmark

    def benchmark_imputation(
        self,
        imputation_pipeline,
        feature_columns,
        df_data,
        df_train,
        df_test,
        df_mask_train,
        df_mask_test,
    ):
        feature_columns_ = [col for col in feature_columns if col in self.imputation_columns]
        df_train_x = df_train[feature_columns]
        df_test_x = df_test[feature_columns]
        if imputation_pipeline is not None:
            transformer_imputation = imputation_pipeline["transformer"]
            imputer = imputation_pipeline["imputer"]

            # Suppose that all categories/values are known
            transformer_imputation.fit(df_data)

            df_train_x_transformed = transformer_imputation.transform_subset(df_train_x)
            df_train_x_transformed_corrupted = df_train_x_transformed.copy()
            df_train_x_transformed_corrupted[df_mask_train] = np.nan
            imputer = imputer.fit(df_train_x_transformed_corrupted)

            df_train_x_reversed_imputed = self.impute(
                transformer_imputation, imputer, df_train_x, df_mask_train, df_train_x_transformed
            )

            (
                dict_imp_score_mean_train,
                dict_imp_scores_train,
            ) = self.get_imputation_scores_by_dataframe(
                df_train_x[feature_columns_],
                df_train_x_reversed_imputed[feature_columns_],
                df_mask_train[feature_columns_],
            )

            df_test_x_reversed_imputed = self.impute(
                transformer_imputation, imputer, df_test_x, df_mask_test, df_train_x_transformed
            )

            (
                dict_imp_score_mean_test,
                dict_imp_scores_test,
            ) = self.get_imputation_scores_by_dataframe(
                df_test_x[feature_columns_],
                df_test_x_reversed_imputed[feature_columns_],
                df_mask_test[feature_columns_],
            )

            benchmark = {
                "dict_imp_score_mean_train": dict_imp_score_mean_train,
                "dict_imp_scores_train": dict_imp_scores_train,
                "dict_imp_score_mean_test": dict_imp_score_mean_test,
                "dict_imp_scores_test": dict_imp_scores_test,
            }

        else:
            df_train_x_corrupted = df_train_x.copy()
            df_train_x_corrupted[df_mask_train] = np.nan
            df_train_x_reversed_imputed = df_train_x_corrupted

            df_test_x_corrupted = df_test_x.copy()
            df_test_x_corrupted[df_mask_test] = np.nan
            df_test_x_reversed_imputed = df_test_x_corrupted

            benchmark = None

        output = {
            "df_train_x_reversed_imputed": df_train_x_reversed_imputed,
            "df_test_x_reversed_imputed": df_test_x_reversed_imputed,
        }

        return output, benchmark

    def impute(self, transformer, imputer, df, df_mask, df_transformed):
        df_transformed_corrupted = transformer.transform_subset(df)
        df_transformed_corrupted[df_mask] = np.nan

        df_transformed_imputed = imputer.transform(df_transformed_corrupted)

        df_transformed_imputed = df_transformed_imputed.clip(upper=df_transformed.max(), axis=1)
        df_transformed_imputed = df_transformed_imputed.clip(lower=df_transformed.min(), axis=1)

        df_reversed_imputed = transformer.reverse_transform_subset(df_transformed_imputed)

        return df_reversed_imputed

    def benchmark_prediction(
        self,
        prediction_pipeline,
        target_column,
        df_data,
        df_train,
        df_test,
        df_train_x_reversed_imputed,
        df_test_x_reversed_imputed,
        df_mask_train=None,
        df_mask_test=None,
    ):
        transformer_prediction = prediction_pipeline["transformer"]
        predictor = prediction_pipeline["predictor"]
        handle_nan = prediction_pipeline["handle_nan"]

        df_train_y = df_train[[target_column]]
        df_test_y = df_test[[target_column]]

        if (
            df_train_x_reversed_imputed.isna().sum().sum() > 0
            and df_test_x_reversed_imputed.isna().sum().sum() > 0
            and not handle_nan
        ):
            return None

        if transformer_prediction is not None:
            # Suppose that all categories/values are known
            if (
                df_train_x_reversed_imputed.isna().sum().sum() > 0
                or df_test_x_reversed_imputed.isna().sum().sum() > 0
            ):
                df_train_reversed_imputed = pd.concat(
                    [df_train_x_reversed_imputed, df_train_y], axis=1
                )
                df_test_reversed_imputed = pd.concat(
                    [df_test_x_reversed_imputed, df_test_y], axis=1
                )
                transformer_prediction.fit(
                    pd.concat([df_train_reversed_imputed, df_test_reversed_imputed], axis=0)
                )

                df_train_x_transformed_imputed = transformer_prediction.transform_subset(
                    df_train_x_reversed_imputed
                )
                df_train_x_transformed_imputed[df_mask_train] = np.nan

                df_test_x_transformed = transformer_prediction.transform_subset(
                    df_test_x_reversed_imputed
                )
                df_test_x_transformed[df_mask_test] = np.nan

            else:
                transformer_prediction.fit(df_data)

                df_train_x_transformed_imputed = transformer_prediction.transform_subset(
                    df_train_x_reversed_imputed
                )
                df_test_x_transformed = transformer_prediction.transform_subset(
                    df_test_x_reversed_imputed
                )

            df_train_y_transformed = transformer_prediction.transform_subset(df_train_y)

        else:
            df_train_x_transformed_imputed = df_train_x_reversed_imputed
            df_train_y_transformed = df_train_y
            df_test_x_transformed = df_test_x_reversed_imputed

        predictor = predictor.fit(
            df_train_x_transformed_imputed,
            df_train_y_transformed[target_column],
        )
        df_test_y_transformed_predicted = pd.DataFrame(
            predictor.predict(df_test_x_transformed),
            columns=[target_column],
            index=df_test_y.index,
        )
        if transformer_prediction is not None:
            df_test_y_reserved_predicted = transformer_prediction.reverse_transform_subset(
                df_test_y_transformed_predicted
            )
        else:
            df_test_y_reserved_predicted = df_test_y_transformed_predicted

        (
            dict_pred_score_mean_test,
            dict_pred_scores_test,
        ) = self.get_prediction_scores_by_column(df_test_y, df_test_y_reserved_predicted)

        output = {
            "dict_pred_score_mean_test": dict_pred_score_mean_test,
            "dict_pred_scores_test": dict_pred_scores_test,
        }

        return output

    def get_row_benchmark(
        self,
        idx_fold,
        target_column,
        hole_generator,
        idx_mask,
        imputation_pipeline,
        prediction_pipeline,
        benchmark_imputation,
        benchmark_prediction,
    ):

        if target_column in self.columns_numerical:
            prediction_task = "regression"
        elif target_column in self.columns_categorical:
            prediction_task = "classification"
        else:
            prediction_task = "unknown"

        transformer_prediction = prediction_pipeline["transformer"]
        predictor = prediction_pipeline["predictor"]
        if transformer_prediction is not None:
            tran_pre_name = transformer_prediction.__class__.__name__
        else:
            tran_pre_name = "None"

        if hole_generator is not None:
            if imputation_pipeline is not None:
                transformer_imputation = imputation_pipeline["transformer"]
                imputer = imputation_pipeline["imputer"]

                if transformer_imputation is not None:
                    tran_imp_name = transformer_imputation.__class__.__name__
                else:
                    tran_imp_name = "None"
                imputer_name = imputer.__class__.__name__
            else:
                tran_imp_name = "None"
                imputer_name = "None"

            hole_generator_name = hole_generator.__class__.__name__
            ratio_masked = hole_generator.ratio_masked
        else:
            hole_generator_name = "None"
            ratio_masked = 0
            tran_imp_name = "None"
            imputer_name = "None"

        row_benchmark = {
            "n_fold": idx_fold,
            "n_mask": idx_mask,
            "hole_generator": hole_generator_name,
            "ratio_masked": ratio_masked,
            "transformer_imputation": tran_imp_name,
            "imputer": imputer_name,
            "target_column": target_column,
            "prediction_task": prediction_task,
            "transformer_prediction": tran_pre_name,
            "predictor": predictor.__class__.__name__,
        }

        if benchmark_imputation is not None:
            dict_imp_score_mean_train = benchmark_imputation["dict_imp_score_mean_train"]
            dict_imp_scores_train = benchmark_imputation["dict_imp_scores_train"]
            dict_imp_score_mean_test = benchmark_imputation["dict_imp_score_mean_test"]
            dict_imp_scores_test = benchmark_imputation["dict_imp_scores_test"]

            dict_imp_score_mean_train_ = dict(
                (f"{k}_train_set", v) for k, v in dict_imp_score_mean_train.items()
            )
            dict_imp_score_mean_test_ = dict(
                (f"{k}_test_set", v) for k, v in dict_imp_score_mean_test.items()
            )

            row_benchmark = {**row_benchmark, **dict_imp_score_mean_train_}
            row_benchmark = {**row_benchmark, **dict_imp_score_mean_test_}
            row_benchmark["imputation_scores_trainset"] = dict_imp_scores_train
            row_benchmark["imputation_scores_testset"] = dict_imp_scores_test

        if benchmark_prediction is not None:
            dict_pred_score_mean_test = benchmark_prediction["dict_pred_score_mean_test"]
            dict_pred_scores_test = benchmark_prediction["dict_pred_scores_test"]
            row_benchmark = {**row_benchmark, **dict_pred_score_mean_test}
            row_benchmark["prediction_scores_testset"] = dict_pred_scores_test

        # print({
        #     "n_fold": idx_fold,
        #     "target_column": target_column,
        #     "hole_generator": hole_generator_name,
        #     "ratio_masked": ratio_masked,
        #     "n_mask": idx_mask,
        #     "transformer_imputation": tran_imp_name,
        #     "imputer": imputer_name,
        #     "transformer_prediction": tran_pre_name,
        #     "predictor": predictor.__class__.__name__,
        # })

        return row_benchmark

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


def highlight_best(x, color="green"):
    if re.search("|".join(["f1_score", "roc_auc_score"]), "_".join(x.name)):
        return [f"background: {color}" if v == x.max() else "" for v in x]
    else:
        return [f"background: {color}" if v == x.min() else "" for v in x]


def get_benchmark_aggregate(df, cols_groupby=["imputer", "predictor"], agg_func=pd.DataFrame.mean):
    metrics = [col for col in df.columns if "_score_" in col]
    if cols_groupby is None:
        cols_groupby = [col for col in df.columns if col not in metrics]
    df_groupby = df.groupby(cols_groupby)[metrics].apply(agg_func)

    # if keep_values:
    #     for metric in metrics:
    #         df_groupby[f"{metric}_values"] = df.groupby(cols_groupby)[metric].apply(list)
    cols_imputation = [col for col in df_groupby.columns if "imputation_score_" in col]
    cols_prediction = [col for col in df_groupby.columns if "prediction_score_" in col]
    cols_train_set = [col for col in df_groupby.columns if "_train_set" in col]
    cols_test_set = [col for col in df_groupby.columns if "_test_set" in col]
    cols_multi_index = []
    for col in df_groupby.columns:
        if col in cols_imputation and col in cols_train_set:
            cols_multi_index.append(
                (
                    "imputation_score",
                    "train_set",
                    col.replace("imputation_score_", "").replace("_train_set", ""),
                )
            )
        if col in cols_imputation and col in cols_test_set:
            cols_multi_index.append(
                (
                    "imputation_score",
                    "test_set",
                    col.replace("imputation_score_", "").replace("_test_set", ""),
                )
            )
        if col in cols_prediction:
            cols_multi_index.append(
                ("prediction_score", "test_set", col.replace("prediction_score_", ""))
            )
    df_groupby.columns = pd.MultiIndex.from_tuples(cols_multi_index)
    return df_groupby


def visualize_mlflow(df, exp_name):
    cols_mean_on = ["n_fold", "n_mask"]
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


def visualize_plotly(df, selected_columns):
    columns_numerical = df.select_dtypes(include=np.number).columns.tolist()
    columns_categorical = [col for col in df.columns.to_list() if col not in columns_numerical]

    df = df[selected_columns]
    df = df.dropna()

    dimensions = []
    for col in selected_columns:
        if col in columns_categorical:
            dfg = pd.DataFrame({col: df[col].unique()})
            dfg[f"{col}_dummy"] = dfg.index
            df = pd.merge(df, dfg, on=col, how="left")

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


def plot_stack_bar(
    df,
    col_y=("prediction_score", "test_set", "wmape"),
    col_x=["hole_generator", "imputer"],
    col_legend="predictor",
):
    cols_groupby = col_x + [col_legend]
    df_agg = get_benchmark_aggregate(df, cols_groupby=cols_groupby)
    df_agg_plot = df_agg.reset_index()

    fig = go.Figure()

    for value in df_agg_plot[col_legend].unique():
        df_agg_plot_ = df_agg_plot[df_agg_plot[col_legend] == value]
        fig.add_trace(
            go.Bar(
                x=[df_agg_plot_[col] for col in col_x],
                y=df_agg_plot_.loc[:, col_y],
                showlegend=True,
                name=value,
            )
        )

    fig.update_layout(barmode="stack")
    fig.update_layout(title=f'{col_y[2]} as a function of {"+".join(cols_groupby)}')

    return fig
