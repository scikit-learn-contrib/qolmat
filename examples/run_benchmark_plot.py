import argparse
import sys

sys.path.append("/home/ec2-user/qolmat/")

import pickle
import pandas as pd
import qolmat.benchmark.imputer_predictor as imppred

results = pd.read_pickle("data/imp_pred/benchmark_all_new.pkl")
results_plot = results.copy()

num_dataset = len(results["dataset"].unique())
num_predictor = len(results["predictor"].unique())
num_imputer = len(results["imputer"].unique()) - 1
num_fold = len(results["n_fold"].unique())
# We remove the case [hole_generator=None, ratio_masked=0, n_mask=nan]
num_mask = len(results["n_mask"].unique()) - 1
num_ratio_masked = len(results["ratio_masked"].unique()) - 1
num_trial = num_fold * num_mask

print(f"datasets: {results['dataset'].unique()}")
print(f"predictor: {results['predictor'].unique()}")
print(f"imputer: {results['imputer'].unique()}")

num_runs_each_predictor = (
    results_plot.groupby(["hole_generator", "ratio_masked", "imputer", "predictor"])
    .count()
    .max()
    .max()
)
num_runs_all_predictors = (
    results_plot.groupby(["hole_generator", "ratio_masked", "imputer"]).count().max().max()
)

results_plot["imputer_predictor"] = results_plot["imputer"] + "_" + results_plot["predictor"]

imputation_metrics = ["wmape", "dist_corr_pattern"]
prediction_metrics = ["wmape"]

for metric in prediction_metrics:
    for type_set in ["notnan", "nan"]:

        results_plot[
            f"prediction_score_{type_set}_{metric}_relative_percentage_gain_data_complete"
        ] = results_plot.apply(
            lambda x: imppred.get_relative_score(
                x,
                results_plot,
                col=f"prediction_score_{type_set}_{metric}",
                method="relative_percentage_gain",
                is_ref_hole_generator_none=True,
            ),
            axis=1,
        )

        results_plot[
            f"prediction_score_{type_set}_{metric}_gain_data_complete"
        ] = results_plot.apply(
            lambda x: imppred.get_relative_score(
                x,
                results_plot,
                col=f"prediction_score_{type_set}_{metric}",
                method="gain",
                is_ref_hole_generator_none=True,
            ),
            axis=1,
        )
        results_plot[
            f"prediction_score_{type_set}_{metric}_gain_count_data_complete"
        ] = results_plot.apply(
            lambda x: 1
            if x[f"prediction_score_{type_set}_{metric}_gain_data_complete"] > 0
            else 0,
            axis=1,
        )

        results_plot[f"prediction_score_{type_set}_{metric}_gain_ratio_data_complete"] = (
            results_plot[f"prediction_score_{type_set}_{metric}_gain_count_data_complete"]
            / num_runs_each_predictor
        )

for metric in prediction_metrics:
    for type_set in ["notnan", "nan"]:

        results_plot[
            f"prediction_score_{type_set}_{metric}_relative_percentage_gain"
        ] = results_plot.apply(
            lambda x: imppred.get_relative_score(
                x,
                results_plot,
                col=f"prediction_score_{type_set}_{metric}",
                method="relative_percentage_gain",
            ),
            axis=1,
        )

        results_plot[f"prediction_score_{type_set}_{metric}_gain"] = results_plot.apply(
            lambda x: imppred.get_relative_score(
                x, results_plot, col=f"prediction_score_{type_set}_{metric}", method="gain"
            ),
            axis=1,
        )
        results_plot[f"prediction_score_{type_set}_{metric}_gain_count"] = results_plot.apply(
            lambda x: 1 if x[f"prediction_score_{type_set}_{metric}_gain"] > 0 else 0, axis=1
        )

        results_plot[f"prediction_score_{type_set}_{metric}_gain_ratio"] = (
            results_plot[f"prediction_score_{type_set}_{metric}_gain_count"]
            / num_runs_each_predictor
        )


for metric in prediction_metrics:
    for type_set in ["notnan", "nan"]:
        for ref_imputer in ["ImputerMedian", "ImputerShuffle"]:

            results_plot[
                f"prediction_score_{type_set}_{metric}_relative_percentage_gain_{ref_imputer}"
            ] = results_plot.apply(
                lambda x: imppred.get_relative_score(
                    x,
                    results_plot,
                    col=f"prediction_score_{type_set}_{metric}",
                    method="relative_percentage_gain",
                    ref_imputer=ref_imputer,
                ),
                axis=1,
            )

            results_plot[
                f"prediction_score_{type_set}_{metric}_gain_{ref_imputer}"
            ] = results_plot.apply(
                lambda x: imppred.get_relative_score(
                    x,
                    results_plot,
                    col=f"prediction_score_{type_set}_{metric}",
                    method="gain",
                    ref_imputer=ref_imputer,
                ),
                axis=1,
            )
            results_plot[
                f"prediction_score_{type_set}_{metric}_gain_count_{ref_imputer}"
            ] = results_plot.apply(
                lambda x: 1
                if x[f"prediction_score_{type_set}_{metric}_gain_{ref_imputer}"] > 0
                else 0,
                axis=1,
            )

            results_plot[f"prediction_score_{type_set}_{metric}_gain_ratio_{ref_imputer}_all"] = (
                results_plot[f"prediction_score_{type_set}_{metric}_gain_count_{ref_imputer}"]
                / num_runs_all_predictors
            )

            results_plot[f"prediction_score_{type_set}_{metric}_gain_ratio_{ref_imputer}_each"] = (
                results_plot[f"prediction_score_{type_set}_{metric}_gain_count_{ref_imputer}"]
                / num_runs_each_predictor
            )


# metric = 'mae'
metric = "wmape"

for metric in prediction_metrics:
    for type_set in ["notnan", "nan"]:
        results_plot_ = results_plot[~(results_plot["imputer"].isin(["None"]))].copy()

        results_plot_[
            f"prediction_score_{type_set}_{metric}_imputer_rank"
        ] = results_plot_.groupby(
            ["dataset", "n_fold", "hole_generator", "ratio_masked", "n_mask", "predictor"]
        )[
            f"prediction_score_{type_set}_{metric}"
        ].rank()

        results_plot = results_plot.merge(
            results_plot_[[f"prediction_score_{type_set}_{metric}_imputer_rank"]],
            left_index=True,
            right_index=True,
            how="left",
        )

for metric in imputation_metrics:
    results_plot_ = results_plot[~(results_plot["imputer"].isin(["None"]))].copy()

    results_plot_[f"imputation_score_{metric}_rank_train_set"] = results_plot_.groupby(
        ["dataset", "n_fold", "hole_generator", "ratio_masked", "n_mask", "predictor"]
    )[f"imputation_score_{metric}_train_set"].rank()
    results_plot_[f"imputation_score_{metric}_rank_test_set"] = results_plot_.groupby(
        ["dataset", "n_fold", "hole_generator", "ratio_masked", "n_mask", "predictor"]
    )[f"imputation_score_{metric}_test_set"].rank()

    results_plot = results_plot.merge(
        results_plot_[
            [
                f"imputation_score_{metric}_rank_train_set",
                f"imputation_score_{metric}_rank_test_set",
            ]
        ],
        left_index=True,
        right_index=True,
        how="left",
    )

for metric in prediction_metrics:
    for type_set in ["notnan", "nan"]:
        results_plot[
            f"prediction_score_{type_set}_{metric}_imputer_predictor_rank"
        ] = results_plot.groupby(
            ["dataset", "n_fold", "hole_generator", "ratio_masked", "n_mask"]
        )[
            f"prediction_score_{type_set}_{metric}"
        ].rank()

with open("data/imp_pred/benchmark_plot.pkl", "wb") as handle:
    pickle.dump(results_plot, handle, protocol=pickle.HIGHEST_PROTOCOL)
