import argparse
import sys

sys.path.append("/home/ec2-ngo/qolmat/")

from datasets import load_dataset

from sklearn.compose import ColumnTransformer
from sklearn import preprocessing

from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

from qolmat.benchmark import missing_patterns
from qolmat.imputations import imputers, imputers_pytorch
from qolmat.imputations.diffusions import ddpms

from qolmat.benchmark.imputer_predictor import BenchmarkImputationPrediction

parser = argparse.ArgumentParser(description="Tabular data benchmark")
parser.add_argument("--data", type=str, help="Name of data")
parser.add_argument("--path", type=str, help="Path to store benchmarks", default="data/imp_pred")
parser.add_argument("--batch_size", type=int, help="Batch size", default=1000)
parser.add_argument("--n_folds", type=int, help="#folds", default=10)
parser.add_argument("--n_masks", type=int, help="#masks", default=5)

args = parser.parse_args()

dataset = load_dataset("inria-soda/tabular-benchmark", data_files=f"reg_num/{args.data}.csv")
df_data = dataset["train"].to_pandas()
column_target = df_data.columns.to_list()[-1]
columns_numerical = df_data.select_dtypes(include="number").columns.tolist()
columns_categorical = df_data.select_dtypes(include="object").columns.tolist()
size_data = len(df_data)

benchmark = BenchmarkImputationPrediction(
    n_masks=args.n_masks,
    n_folds=args.n_folds,
    imputation_metrics=["mae", "KL_columnwise"],
    prediction_metrics=["mae"],
)

# Hole generators
hole_generators = [
    None,
    missing_patterns.MCAR(ratio_masked=0.05),
    missing_patterns.MCAR(ratio_masked=0.1),
    missing_patterns.MCAR(ratio_masked=0.2),
    missing_patterns.MCAR(ratio_masked=0.5),
    missing_patterns.MAR(ratio_masked=0.05),
    missing_patterns.MAR(ratio_masked=0.1),
    missing_patterns.MAR(ratio_masked=0.2),
    missing_patterns.MAR(ratio_masked=0.5),
    missing_patterns.MNAR(ratio_masked=0.05),
    missing_patterns.MNAR(ratio_masked=0.1),
    missing_patterns.MNAR(ratio_masked=0.2),
    missing_patterns.MNAR(ratio_masked=0.5),
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

# Format of prediction pipeline
# {
# "transformer_x": transformer_x/None,
# "imputer": imputer,
# }

imputation_pipelines = [
    None,
    {"imputer": imputers.ImputerMean()},
    {"imputer": imputers.ImputerMedian()},
    {"imputer": imputers.ImputerMode()},
    {"imputer": imputers.ImputerShuffle()},
    {"imputer": imputers.ImputerMICE(estimator=Ridge(), max_iter=100)},
    {"imputer": imputers.ImputerKNN()},
    {"imputer": imputers.ImputerRPCA(max_iterations=100)},
    {"imputer": imputers.ImputerEM(max_iter_em=100)},
    {
        "imputer": imputers_pytorch.ImputerDiffusion(
            model=ddpms.TabDDPM(num_sampling=100), batch_size=args.batch_size, epochs=100
        )
    },
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

# Format of prediction pipeline
# {
# "transformer_x": transformer_x/None,
# "transformer_y": transformer_y/None,
# "predictor": RidgeClassifier(),
# "handle_nan": True/False (default=False),
# "add_nan_indicator": True/False (default=True)
# }

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
        {
            "transformer_x": transformer_prediction_x,
            "transformer_y": transformer_prediction_y,
            "predictor": XGBRegressor(),
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
            "add_nan_indicator": True,
        },
        {
            "transformer_x": transformer_prediction_x,
            "transformer_y": transformer_prediction_y,
            "predictor": HistGradientBoostingClassifier(),
            "handle_nan": True,
            "add_nan_indicator": True,
        },
        {
            "transformer_x": transformer_prediction_x,
            "transformer_y": transformer_prediction_y,
            "predictor": XGBClassifier(),
            "handle_nan": True,
            "add_nan_indicator": True,
        },
    ]

results = benchmark.compare(
    df_data=df_data,
    columns_numerical=columns_numerical,
    columns_categorical=columns_categorical,
    file_path=f"{args.path}/benchmark_{args.data}.pkl",
    hole_generators=hole_generators,
    imputation_pipelines=imputation_pipelines,
    target_prediction_pipeline_pairs=target_prediction_pipeline_pairs,
)
