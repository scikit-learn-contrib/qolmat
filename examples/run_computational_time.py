import time
import pickle
import pandas as pd
import numpy as np
from datasets import load_dataset

from qolmat.imputations import imputers, imputers_pytorch
from qolmat.imputations.diffusions import ddpms
from qolmat.benchmark import missing_patterns

from xgboost import XGBRegressor

data_name = "house_sales"
dataset = load_dataset("inria-soda/tabular-benchmark", data_files=f"reg_num/{data_name}.csv")
df_data = dataset["train"].to_pandas()
column_target = df_data.columns.to_list()[-1]
columns_numerical = df_data.select_dtypes(include="number").columns.tolist()
columns_categorical = df_data.select_dtypes(include="object").columns.tolist()

list_imputers = [
    imputers.ImputerMedian(),
    imputers.ImputerShuffle(),
    imputers.ImputerMICE(estimator=XGBRegressor(tree_method="hist", n_jobs=1), max_iter=100),
    imputers.ImputerKNN(),
    imputers.ImputerRPCA(max_iterations=100),
    imputers.ImputerEM(max_iter_em=100, method="mle"),
    imputers_pytorch.ImputerDiffusion(
        model=ddpms.TabDDPM(num_sampling=50), batch_size=1000, epochs=100
    ),
]

benchmark_duration_rows = []
num_cols = 10
for num_rows in [1000, 10000, 20000]:
    df_sub_data = df_data.iloc[:num_rows, :num_cols]
    hole_generator = missing_patterns.MCAR(ratio_masked=0.1)
    df_sub_mask = hole_generator.split(df_sub_data)[0]
    df_sub_data[df_sub_mask] = np.nan

    for imputer in list_imputers:
        start_time = time.time()
        imputer = imputer.fit(df_sub_data)
        duration_imputation_fit = time.time() - start_time

        start_time = time.time()
        df_imputed = imputer.transform(df_sub_data)
        duration_imputation_transform = time.time() - start_time

        benchmark_duration_rows.append(
            {
                "imputer": imputer.__class__.__name__,
                "n_columns": df_sub_data.shape[1],
                "size_data": df_sub_data.shape[0],
                "duration_imputation_fit": duration_imputation_fit,
                "duration_imputation_transform": duration_imputation_transform,
            }
        )

        df_benchmark_rows = pd.DataFrame(benchmark_duration_rows)
        with open(f"data/imp_pred/benchmark_time_rows_{data_name}.pkl", "wb") as handle:
            pickle.dump(df_benchmark_rows, handle, protocol=pickle.HIGHEST_PROTOCOL)

benchmark_duration_cols = []
num_rows = 1000
for num_cols in [5, 10, 15]:
    df_sub_data = df_data.iloc[:num_rows, :num_cols]
    hole_generator = missing_patterns.MCAR(ratio_masked=0.1)
    df_sub_mask = hole_generator.split(df_sub_data)[0]
    df_sub_data[df_sub_mask] = np.nan

    for imputer in list_imputers:
        start_time = time.time()
        imputer = imputer.fit(df_sub_data)
        duration_imputation_fit = time.time() - start_time

        start_time = time.time()
        df_imputed = imputer.transform(df_sub_data)
        duration_imputation_transform = time.time() - start_time

        benchmark_duration_cols.append(
            {
                "imputer": imputer.__class__.__name__,
                "n_columns": df_sub_data.shape[1],
                "size_data": df_sub_data.shape[0],
                "duration_imputation_fit": duration_imputation_fit,
                "duration_imputation_transform": duration_imputation_transform,
            }
        )

        df_benchmark_cols = pd.DataFrame(benchmark_duration_cols)
        with open(f"data/imp_pred/benchmark_time_cols_{data_name}.pkl", "wb") as handle:
            pickle.dump(df_benchmark_cols, handle, protocol=pickle.HIGHEST_PROTOCOL)
