from typing import Dict, List, Tuple, Optional
import mlflow
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import tqdm
import re
import scipy
import time
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
        columns_numerical: List[str],
        columns_categorical: List[str],
        file_path: str,
        hole_generators: List,
        imputation_pipelines,
        target_prediction_pipeline_pairs,
        imputation_columns: List[str] = [],
    ):
        self.columns = df_data.columns.to_list()
        self.columns_numerical = columns_numerical
        self.columns_categorical = columns_categorical
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
                                        feature_columns,
                                        df_train,
                                        df_test,
                                        out_imputation["df_train_x_imputed"],
                                        out_imputation["df_test_x_imputed"],
                                        df_mask_train,
                                        df_mask_test,
                                    )

                                    row_benchmark = self.get_row_benchmark(
                                        df_train,
                                        df_test,
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
                                feature_columns,
                                df_train,
                                df_test,
                                df_train_x,
                                df_test_x,
                                None,
                                None,
                            )

                            row_benchmark = self.get_row_benchmark(
                                df_train,
                                df_test,
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
        df_train,
        df_test,
        df_mask_train,
        df_mask_test,
    ):
        feature_columns_ = [col for col in feature_columns if col in self.imputation_columns]
        df_train_x = df_train[feature_columns]
        df_test_x = df_test[feature_columns]

        df_train_x_corrupted = df_train_x.copy()
        df_train_x_corrupted[df_mask_train] = np.nan

        df_test_x_corrupted = df_test_x.copy()
        df_test_x_corrupted[df_mask_test] = np.nan

        benchmark = None
        df_train_x_imputed = df_train_x_corrupted
        df_test_x_imputed = df_test_x_corrupted
        if imputation_pipeline is not None:
            if "transformer_x" in imputation_pipeline:
                transformer_imputation_x = imputation_pipeline["transformer_x"]
            else:
                transformer_imputation_x = None
            imputer = imputation_pipeline["imputer"]

            if transformer_imputation_x is not None:
                # Suppose that all categories/values are known
                df_data_x = pd.concat([df_train_x, df_test_x], axis=0)
                transformer_imputation_x = transformer_imputation_x.fit(df_data_x)

                df_train_x_transformed_corrupted = pd.DataFrame(
                    transformer_imputation_x.transform(df_train_x_corrupted),
                    columns=transformer_imputation_x.get_feature_names_out(
                        df_train_x_corrupted.columns
                    ),
                    index=df_train_x_corrupted.index,
                )
                df_test_x_transformed_corrupted = pd.DataFrame(
                    transformer_imputation_x.transform(df_test_x_corrupted),
                    columns=transformer_imputation_x.get_feature_names_out(
                        df_test_x_corrupted.columns
                    ),
                    index=df_test_x_corrupted.index,
                )
            else:
                df_train_x_transformed_corrupted = df_train_x_corrupted
                df_test_x_transformed_corrupted = df_test_x_corrupted

            start_time = time.time()
            imputer = imputer.fit(df_train_x_transformed_corrupted)
            duration_imputation_fit = time.time() - start_time

            start_time = time.time()
            df_train_x_transformed_imputed = imputer.transform(df_train_x_transformed_corrupted)
            duration_imputation_transform_train = time.time() - start_time

            start_time = time.time()
            df_test_x_transformed_imputed = imputer.transform(df_test_x_transformed_corrupted)
            duration_imputation_transform_test = time.time() - start_time

            if transformer_imputation_x is not None:
                df_train_x_imputed = self.inverse_transform(
                    transformer_imputation_x, df_train_x_transformed_imputed
                )
                df_test_x_imputed = self.inverse_transform(
                    transformer_imputation_x, df_test_x_transformed_imputed
                )
            else:
                df_train_x_imputed = df_train_x_transformed_imputed
                df_test_x_imputed = df_test_x_transformed_imputed

            (
                dict_imp_score_mean_train,
                dict_imp_scores_train,
            ) = self.get_imputation_scores_by_dataframe(
                df_train_x[feature_columns_],
                df_train_x_imputed[feature_columns_],
                df_mask_train[feature_columns_],
            )

            (
                dict_imp_score_mean_test,
                dict_imp_scores_test,
            ) = self.get_imputation_scores_by_dataframe(
                df_test_x[feature_columns_],
                df_test_x_imputed[feature_columns_],
                df_mask_test[feature_columns_],
            )

            benchmark = {
                "dict_imp_score_mean_train": dict_imp_score_mean_train,
                "dict_imp_scores_train": dict_imp_scores_train,
                "dict_imp_score_mean_test": dict_imp_score_mean_test,
                "dict_imp_scores_test": dict_imp_scores_test,
                "duration_imputation_fit": duration_imputation_fit,
                "duration_imputation_transform_train": duration_imputation_transform_train,
                "duration_imputation_transform_test": duration_imputation_transform_test,
            }

        output = {
            "df_train_x_imputed": df_train_x_imputed,
            "df_test_x_imputed": df_test_x_imputed,
        }

        return output, benchmark

    def inverse_transform(self, transformer, df_transformed):
        df_reversed = pd.DataFrame()
        for transformer_values in transformer.transformers_:
            cols_in = transformer_values[2]
            cols_out = transformer.get_feature_names_out(cols_in)
            df_reversed[cols_in] = transformer_values[1].inverse_transform(
                df_transformed[cols_out]
            )
        df_reversed.index = df_transformed.index
        return df_reversed

    def benchmark_prediction(
        self,
        prediction_pipeline,
        target_column,
        feature_columns,
        df_train,
        df_test,
        df_train_x_imputed,
        df_test_x_imputed,
        df_mask_train,
        df_mask_test,
    ):
        predictor = prediction_pipeline["predictor"]
        if "transformer_x" in prediction_pipeline:
            transformer_prediction_x = prediction_pipeline["transformer_x"]
        else:
            transformer_prediction_x = None
        if "transformer_y" in prediction_pipeline:
            transformer_prediction_y = prediction_pipeline["transformer_y"]
        else:
            transformer_prediction_y = None
        if "handle_nan" in prediction_pipeline:
            handle_nan = prediction_pipeline["handle_nan"]
        else:
            handle_nan = False
        if "add_nan_indicator" in prediction_pipeline:
            add_nan_indicator = prediction_pipeline["add_nan_indicator"]
        else:
            add_nan_indicator = True

        # df_train_x = df_train[feature_columns]
        df_test_x = df_test[feature_columns]

        df_train_y = df_train[[target_column]]
        df_test_y = df_test[[target_column]]

        if (
            df_train_x_imputed.isna().sum().sum() > 0 or df_test_x_imputed.isna().sum().sum() > 0
        ) and not handle_nan:
            return None

        if transformer_prediction_x is not None and transformer_prediction_y is not None:
            # Suppose that all categories/values are known
            df_data_x_imputed = pd.concat([df_train_x_imputed, df_test_x_imputed], axis=0)
            df_data_y = pd.concat([df_train_y, df_test_y], axis=0)
            transformer_prediction_x = transformer_prediction_x.fit(df_data_x_imputed)

            df_train_x_transformed_imputed = transformer_prediction_x.transform(df_train_x_imputed)

            # Evaluate prediction performance on imputed test set
            df_test_x_transformed_imputed = transformer_prediction_x.transform(df_test_x_imputed)

            # Evaluate prediction performance on reference test set
            df_test_x_transformed_notnan = transformer_prediction_x.transform(df_test_x)

            transformer_prediction_y = transformer_prediction_y.fit(df_data_y)

            df_train_y_transformed = transformer_prediction_y.transform(df_train_y)

        else:
            df_train_x_transformed_imputed = df_train_x_imputed
            df_train_y_transformed = df_train_y
            df_test_x_transformed_imputed = df_test_x_imputed
            df_test_x_transformed_notnan = df_test_x

        # add indicator for missing values into x
        if df_mask_train is not None and df_mask_test is not None and add_nan_indicator:
            df_train_x_input = np.concatenate(
                [df_train_x_transformed_imputed, df_mask_train.astype(int).values], axis=1
            )
            df_test_x_input_imputed = np.concatenate(
                [df_test_x_transformed_imputed, df_mask_test.astype(int).values], axis=1
            )
            df_mask_test_ = np.zeros(df_mask_test.shape)
            df_test_x_input_notnan = np.concatenate(
                [df_test_x_transformed_notnan, df_mask_test_], axis=1
            )
        else:
            df_train_x_input = df_train_x_transformed_imputed
            df_test_x_input_imputed = df_test_x_transformed_imputed
            df_test_x_input_notnan = df_test_x_transformed_notnan

        # predictor fit
        start_time = time.time()
        predictor = predictor.fit(
            df_train_x_input,
            np.squeeze(df_train_y_transformed),
        )
        duration_prediction_fit = time.time() - start_time

        if transformer_prediction_y is not None:
            # predictor predict for test without nan
            df_test_y_transformed_notnan_predicted = predictor.predict(df_test_x_input_notnan)
            df_test_y_transformed_notnan_predicted = pd.DataFrame(
                df_test_y_transformed_notnan_predicted,
                columns=transformer_prediction_y.get_feature_names_out([target_column]),
                index=df_test_y.index,
            )
            # predictor predict for test with nan
            start_time = time.time()
            df_test_y_transformed_imputed_predicted = predictor.predict(df_test_x_input_imputed)
            duration_prediction_transform = time.time() - start_time

            df_test_y_transformed_imputed_predicted = pd.DataFrame(
                df_test_y_transformed_imputed_predicted,
                columns=transformer_prediction_y.get_feature_names_out([target_column]),
                index=df_test_y.index,
            )

            df_test_y_reversed_notnan_predicted = self.inverse_transform(
                transformer_prediction_y, df_test_y_transformed_notnan_predicted
            )
            df_test_y_reversed_imputed_predicted = self.inverse_transform(
                transformer_prediction_y, df_test_y_transformed_imputed_predicted
            )
        else:
            # predictor predict for test without nan
            df_test_y_reversed_notnan_predicted = predictor.predict(df_test_x_input_notnan)
            df_test_y_reversed_notnan_predicted = pd.DataFrame(
                df_test_y_reversed_notnan_predicted,
                columns=[target_column],
                index=df_test_y.index,
            )

            # predictor predict for test with nan
            start_time = time.time()
            df_test_y_reversed_imputed_predicted = predictor.predict(df_test_x_input_imputed)
            duration_prediction_transform = time.time() - start_time

            df_test_y_reversed_imputed_predicted = pd.DataFrame(
                df_test_y_reversed_imputed_predicted,
                columns=[target_column],
                index=df_test_y.index,
            )

        (
            dict_pred_score_mean_test_notnan,
            dict_pred_scores_test_notnan,
        ) = self.get_prediction_scores_by_column(
            df_test_y, df_test_y_reversed_notnan_predicted, key="notnan"
        )

        (
            dict_pred_score_mean_test_nan,
            dict_pred_scores_test_nan,
        ) = self.get_prediction_scores_by_column(
            df_test_y, df_test_y_reversed_imputed_predicted, key="nan"
        )

        output = {
            "dict_pred_score_mean_test_nan": dict_pred_score_mean_test_nan,
            "dict_pred_scores_test_nan": dict_pred_scores_test_nan,
            "dict_pred_score_mean_test_notnan": dict_pred_score_mean_test_notnan,
            "dict_pred_scores_test_notnan": dict_pred_scores_test_notnan,
            "duration_prediction_fit": duration_prediction_fit,
            "duration_prediction_transform": duration_prediction_transform,
        }

        return output

    def get_row_benchmark(
        self,
        df_train,
        df_test,
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

        predictor = prediction_pipeline["predictor"]
        if "transformer_x" in prediction_pipeline:
            transformer_prediction_x = prediction_pipeline["transformer_x"]
        else:
            transformer_prediction_x = None
        if "transformer_y" in prediction_pipeline:
            transformer_prediction_y = prediction_pipeline["transformer_y"]
        else:
            transformer_prediction_y = None
        if "handle_nan" in prediction_pipeline:
            handle_nan = prediction_pipeline["handle_nan"]
        else:
            handle_nan = False
        if "add_nan_indicator" in prediction_pipeline:
            add_nan_indicator = prediction_pipeline["add_nan_indicator"]
        else:
            add_nan_indicator = True
        if "add_nan_indicator" in prediction_pipeline:
            add_nan_indicator = prediction_pipeline["add_nan_indicator"]
        else:
            add_nan_indicator = True
        if transformer_prediction_x is not None:
            tran_pre_name_x = "+".join(
                set([tf[1].__class__.__name__ for tf in transformer_prediction_x.transformers_])
            )
        else:
            tran_pre_name_x = "None"
        if transformer_prediction_y is not None:
            tran_pre_name_y = "+".join(
                set([tf[1].__class__.__name__ for tf in transformer_prediction_y.transformers_])
            )
        else:
            tran_pre_name_y = "None"

        if hole_generator is not None:
            if imputation_pipeline is not None:
                if "transformer_x" in imputation_pipeline:
                    transformer_imputation_x = imputation_pipeline["transformer_x"]
                else:
                    transformer_imputation_x = None
                imputer = imputation_pipeline["imputer"]

                if transformer_imputation_x is not None:
                    tran_imp_name = "+".join(
                        set(
                            [
                                tf[1].__class__.__name__
                                for tf in transformer_imputation_x.transformers_
                            ]
                        )
                    )
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
            "size_train_set": len(df_train),
            "size_test_set": len(df_test),
            "n_columns": len(self.columns),
            "n_mask": idx_mask,
            "hole_generator": hole_generator_name,
            "ratio_masked": ratio_masked,
            "transformer_imputation": tran_imp_name,
            "imputer": imputer_name,
            "target_column": target_column,
            "prediction_task": prediction_task,
            "transformer_prediction_x": tran_pre_name_x,
            "transformer_prediction_y": tran_pre_name_y,
            "predictor": predictor.__class__.__name__,
            "handle_nan": handle_nan,
            "add_nan_indicator": add_nan_indicator,
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
            row_benchmark["duration_imputation_fit"] = benchmark_imputation[
                "duration_imputation_fit"
            ]
            row_benchmark["duration_imputation_transform_train"] = benchmark_imputation[
                "duration_imputation_transform_train"
            ]
            row_benchmark["duration_imputation_transform_test"] = benchmark_imputation[
                "duration_imputation_transform_test"
            ]

        if benchmark_prediction is not None:
            dict_pred_score_mean_test_nan = benchmark_prediction["dict_pred_score_mean_test_nan"]
            dict_pred_scores_test_nan = benchmark_prediction["dict_pred_scores_test_nan"]
            dict_pred_score_mean_test_notnan = benchmark_prediction[
                "dict_pred_score_mean_test_notnan"
            ]
            dict_pred_scores_test_notnan = benchmark_prediction["dict_pred_scores_test_notnan"]
            row_benchmark = {
                **row_benchmark,
                **dict_pred_score_mean_test_nan,
                **dict_pred_score_mean_test_notnan,
            }
            row_benchmark["prediction_scores_testset_nan"] = dict_pred_scores_test_nan
            row_benchmark["prediction_scores_testset_notnan"] = dict_pred_scores_test_notnan
            row_benchmark["duration_prediction_fit"] = benchmark_prediction[
                "duration_prediction_fit"
            ]
            row_benchmark["duration_prediction_transform"] = benchmark_prediction[
                "duration_prediction_transform"
            ]

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

    def get_prediction_scores_by_column(self, df_true, df_imputed, key=""):
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

            dict_scores[f"prediction_score_{key}_{metric}"] = score_by_col.to_dict()
            dict_score_mean[f"prediction_score_{key}_{metric}"] = score_by_col.mean()
        return dict_score_mean, dict_scores


def highlight_best(x, color="green"):
    if re.search("|".join(["f1_score", "roc_auc_score"]), "_".join(x.name)):
        return [f"background: {color}" if v == x.max() else "" for v in x]
    else:
        return [f"background: {color}" if v == x.min() else "" for v in x]


def get_benchmark_aggregate(
    df, cols_groupby=["imputer", "predictor"], agg_func=pd.DataFrame.mean, keep_values=False
):
    metrics = [col for col in df.columns if "_score_" in col]
    durations = [col for col in df.columns if "duration_" in col]
    if cols_groupby is None:
        cols_groupby = [col for col in df.columns if col not in metrics and col not in durations]
    df_groupby = df.groupby(cols_groupby)[metrics + durations].apply(agg_func)

    if keep_values:
        for metric in metrics:
            df_groupby[f"{metric}_values"] = df.groupby(cols_groupby)[metric].apply(list)
        for duration in durations:
            df_groupby[f"{duration}_values"] = df.groupby(cols_groupby)[duration].apply(list)
    cols_imputation = [col for col in df_groupby.columns if "imputation_score_" in col]
    cols_prediction = [col for col in df_groupby.columns if "prediction_score_" in col]
    cols_train_set = [col for col in df_groupby.columns if "_train_set" in col]
    cols_test_set = [col for col in df_groupby.columns if "_test_set" in col]

    cols_duration_imputation = [col for col in df_groupby.columns if "_imputation_" in col]
    cols_duration_prediction = [col for col in df_groupby.columns if "_prediction_" in col]

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
            if "notnan" in col:
                cols_multi_index.append(
                    (
                        "prediction_score",
                        "test_set_not_nan",
                        col.replace("prediction_score_notnan_", ""),
                    )
                )
            else:
                cols_multi_index.append(
                    (
                        "prediction_score",
                        "test_set_with_nan",
                        col.replace("prediction_score_nan_", ""),
                    )
                )
        if col in cols_duration_imputation:
            cols_multi_index.append(
                (
                    "duration",
                    "imputation",
                    col.replace("duration_imputation_", ""),
                )
            )
        if col in cols_duration_prediction:
            cols_multi_index.append(
                (
                    "duration",
                    "prediction",
                    col.replace("duration_prediction_", ""),
                )
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
                if df_query[col_full_scores].notna().all():
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


def get_confidence_interval(x, confidence_level=0.95):
    # https://www.statology.org/confidence-intervals-python/
    interval = scipy.stats.norm.interval(
        confidence=confidence_level, loc=np.mean(x), scale=scipy.stats.sem(x)
    )
    width = interval[1] - interval[0]
    return [interval[0], interval[1], width]


def plot_bar_y_1D(
    df_agg,
    col_displayed=("prediction_score", "test_set", "wmape"),
    cols_grouped=["hole_generator", "imputer", "predictor"],
    add_annotation=True,
    add_confidence_interval=False,
    confidence_level=0.95,
):
    df_agg_plot = df_agg.reset_index()
    col_legend = cols_grouped[-1]
    cols_x = [col for col in cols_grouped if col != col_legend]

    fig = go.Figure()
    for value in df_agg_plot[col_legend].unique():
        df_agg_plot_ = df_agg_plot[df_agg_plot[col_legend] == value]

        error_y_width = None
        if add_confidence_interval:
            value_ = list(col_displayed)
            value_[2] = value_[2] + "_values"
            error_y = np.array(
                df_agg_plot_.loc[:, tuple(value_)]
                .apply(lambda x: get_confidence_interval(x, confidence_level))
                .to_list()
            )
            error_y_width = dict(type="data", array=error_y[:, 2] / 2)

        text = None
        if add_annotation:
            text = df_agg_plot_.loc[:, col_displayed]

        fig.add_trace(
            go.Bar(
                x=[df_agg_plot_[col].astype(str) for col in cols_x],
                y=df_agg_plot_.loc[:, col_displayed],
                showlegend=True,
                name=str(value),
                text=text,
                error_y=error_y_width,
            )
        )
    metric_name = col_displayed[2]
    if add_annotation:
        fig.update_traces(texttemplate="%{text:.2}", textposition="outside")
    fig.update_layout(barmode="group")
    fig.update_layout(title=f'{metric_name} as a function of {"+".join(cols_grouped)}')

    return fig


def plot_bar_y_nD(
    df_agg,
    cols_displayed=[
        ("imputation_score", "test_set", "wmape"),
        ("prediction_score", "test_set", "wmape"),
    ],
    cols_grouped=["hole_generator", "imputer", "predictor"],
    add_annotation=True,
    add_confidence_interval=False,
    confidence_level=0.95,
):
    col_legend_idx = []
    for i in range(len(cols_displayed) - 1):
        for j in range(len(cols_displayed[i])):
            if cols_displayed[i][j] != cols_displayed[i + 1][j]:
                col_legend_idx.append(j)

    fig = go.Figure()
    for value in cols_displayed:
        name = "_".join([value[i] for i in set(col_legend_idx)])

        error_y_width = None
        if add_confidence_interval:
            value_ = list(value)
            value_[2] = value[2] + "_values"

            error_y = np.array(
                df_agg.loc[:, tuple(value_)]
                .apply(lambda x: get_confidence_interval(x, confidence_level))
                .to_list()
            )
            error_y_width = dict(type="data", array=error_y[:, 2] / 2)

        text = None
        if add_annotation:
            text = df_agg.loc[:, value]

        fig.add_trace(
            go.Bar(
                name=name,
                x=np.array(df_agg.index.to_list()).transpose(),
                y=df_agg.loc[:, value],
                text=text,
                error_y=error_y_width,
            )
        )

    metric_names = set([col[2] for col in cols_displayed])

    if add_annotation:
        fig.update_traces(texttemplate="%{text:.2}", textposition="outside")
    fig.update_layout(barmode="group")

    col_y_inter = set(cols_displayed[0])
    for s in cols_displayed[1:]:
        col_y_inter.intersection_update(s[:2])
    if len(col_y_inter) != 0:
        title = f'{" and ".join(metric_names)} as a function of {"+".join(cols_grouped)}'
        title += f'for {"+".join(list(col_y_inter))}'
        fig.update_layout(title=title)
    else:
        fig.update_layout(
            title=f'{" and ".join(metric_names)} as a function of {"+".join(cols_grouped)}'
        )

    return fig


def plot_bar(
    df,
    col_displayed=("prediction_score", "test_set", "wmape"),
    cols_displayed=None,
    cols_grouped=["hole_generator", "imputer", "predictor"],
    add_annotation=True,
    add_confidence_interval=False,
    confidence_level=0.95,
):
    df_agg = get_benchmark_aggregate(df, cols_groupby=cols_grouped, keep_values=True)

    if cols_displayed is None:
        fig = plot_bar_y_1D(
            df_agg,
            col_displayed,
            cols_grouped,
            add_annotation,
            add_confidence_interval,
            confidence_level,
        )
    else:
        fig = plot_bar_y_nD(
            df_agg,
            cols_displayed,
            cols_grouped,
            add_annotation,
            add_confidence_interval,
            confidence_level,
        )

    return fig
