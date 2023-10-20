import argparse
import sys

sys.path.append("/home/ec2-ngo/qolmat/")

import pandas as pd

from datasets import load_dataset

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing

from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from qolmat.benchmark import missing_patterns
from qolmat.imputations import imputers, imputers_pytorch
from qolmat.imputations.diffusions import ddpms

from qolmat.benchmark.imputer_predictor import BenchmarkImputationPrediction


parser = argparse.ArgumentParser(description="Tabular data benchmark")
parser.add_argument("--data", type=str, help="Name of data")
parser.add_argument("--path", type=str, help="Path to store benchmarks", default="data/imp_pred")

args = parser.parse_args()

dataset = load_dataset("inria-soda/tabular-benchmark", data_files=f"reg_num/{args.data}.csv")
df_data = dataset["train"].to_pandas()
column_target = df_data.columns.to_list()[-1]

columns_categorical = df_data.dtypes[(df_data.dtypes == "int64")].index.to_list()
columns_numerical = df_data.dtypes[(df_data.dtypes == "float64")].index.to_list()

# Hole generators
hole_generators = [
    None,
    missing_patterns.UniformHoleGenerator(ratio_masked=0.2, n_splits=0),
    missing_patterns.UniformHoleGenerator(ratio_masked=0.4, n_splits=0),
    missing_patterns.UniformHoleGenerator(ratio_masked=0.6, n_splits=0),
    missing_patterns.UniformHoleGenerator(ratio_masked=0.8, n_splits=0),
]

# Imputation pipelines
transformers = []
columns_numerical_ = [col for col in columns_numerical if col != column_target]
if len(columns_numerical_) != 0:
    transformers.append(("num", preprocessing.StandardScaler(), columns_numerical_))
columns_categorical_ = [col for col in columns_categorical if col != column_target]
if len(columns_categorical_) != 0:
    transformers.append(("cat", preprocessing.OrdinalEncoder(), columns_categorical_))
transformer_imputation_x = ColumnTransformer(transformers=transformers)

imputation_pipelines = [
    None,
    {"transformer_x": transformer_imputation_x, "imputer": imputers.ImputerMean()},
    {"transformer_x": transformer_imputation_x, "imputer": imputers.ImputerMedian()},
    # {'transformer': hyper_transformer_imputation,
    #  'imputer': imputers.ImputerMICE(estimator=LinearRegression())},
    # {'transformer': hyper_transformer_imputation,
    #  'imputer': imputers.ImputerEM(max_iter_em=100)},
    # {'transformer': hyper_transformer_imputation,
    #  'imputer': imputers.ImputerRPCA(max_iterations=100)},
    # {'transformer': hyper_transformer_imputation,
    #  'imputer': imputers_pytorch.ImputerDiffusion(model=ddpms.TabDDPM(num_sampling=50),
    # batch_size=1000)}
]

# Prediction pipelines
transformers = []
columns_numerical_ = [col for col in columns_numerical if col != column_target]
if len(columns_numerical_) != 0:
    transformers.append(("num", preprocessing.StandardScaler(), columns_numerical_))
columns_categorical_ = [col for col in columns_categorical if col != column_target]
if len(columns_categorical) != 0:
    transformers.append(("cat", preprocessing.OrdinalEncoder(), columns_categorical_))
transformer_prediction_x = ColumnTransformer(transformers=transformers)

target_prediction_pipeline_pairs = {}
if column_target in columns_numerical:
    transformer_prediction_y = ColumnTransformer(
        transformers=[
            ("y_num", preprocessing.StandardScaler(), [column_target]),
        ]
    )
    target_prediction_pipeline_pairs[column_target] = [
        {
            "transformer_x": transformer_prediction_x,
            "transformer_y": transformer_prediction_y,
            "predictor": Ridge(),
            "handle_nan": False,
        },
        {
            "transformer_x": transformer_prediction_x,
            "transformer_y": transformer_prediction_y,
            "predictor": HistGradientBoostingRegressor(),
            "handle_nan": True,
        },
    ]

if column_target in columns_categorical:
    transformer_prediction_y = ColumnTransformer(
        transformers=[
            ("y_cat", preprocessing.OrdinalEncoder(), [column_target]),
        ]
    )
    target_prediction_pipeline_pairs[column_target] = [
        {
            "transformer_x": transformer_prediction_x,
            "transformer_y": transformer_prediction_y,
            "predictor": RidgeClassifier(),
            "handle_nan": False,
        },
        {
            "transformer_x": transformer_prediction_x,
            "transformer_y": transformer_prediction_y,
            "predictor": HistGradientBoostingClassifier(),
            "handle_nan": True,
        },
    ]

benchmark = BenchmarkImputationPrediction(
    n_masks=1, n_folds=2, imputation_metrics=["mae", "KL_columnwise"], prediction_metrics=["mae"]
)

results = benchmark.compare(
    df_data=df_data,
    columns_numerical=columns_numerical,
    columns_categorical=columns_categorical,
    file_path=f"{args.path}/benchmark_{args.data}.pkl",
    hole_generators=hole_generators,
    imputation_pipelines=imputation_pipelines,
    target_prediction_pipeline_pairs=target_prediction_pipeline_pairs,
)
