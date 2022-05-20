import pandas as pd
import numpy as np
from robust_pca.benchmark import cross_validation
from robust_pca.benchmark import utils
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)  # , mean_absolute_percentage_error
from sklearn.utils import resample
from collections import defaultdict
from typing import Optional
from math import floor
from skopt.space import Categorical, Real, Integer
import matplotlib.pyplot as plt


class Comparator:
    def __init__(
        self,
        data,
        ratio_missing,
        models_to_test,
        cols_to_impute,
        n_samples=1,
        search_params={},
        corruption="missing",
        filter_value_nan=-1e10,
    ):

        self.df = data[cols_to_impute]
        self.ratio_missing = ratio_missing
        self.cols_to_impute = cols_to_impute
        self.n_samples = n_samples
        self.filter_value_nan = filter_value_nan
        self.models_to_test = models_to_test
        self.search_params = search_params
        self.corruption = corruption

    def create_corruptions(self, df: pd.DataFrame, random_state: Optional[int] = 29):

        self.df_is_altered = utils.choice_with_mask(
            df, df.notna(), self.ratio_missing, self.filter_value_nan, random_state
        )

        self.corrupted_df = df.copy()
        if self.corruption == "missing":
            self.corrupted_df[self.df_is_altered] = np.nan
        elif self.corruption == "outlier":
            self.corrupted_df[self.df_is_altered] = np.random.randint(
                0, high=3 * np.max(df), size=(int(len(df) * self.ratio_missing))
            )

    def get_errors(
        self,
        signal_ref: pd.DataFrame,
        signal_imputed: pd.DataFrame,
    ) -> float:

        rmse = utils.mean_squared_error(
            signal_ref[self.df_is_altered],
            signal_imputed[self.df_is_altered],
            squared=False,
        )
        mae = utils.mean_absolute_error(
            signal_ref[self.df_is_altered], signal_imputed[self.df_is_altered]
        )
        wmape = utils.weighted_mean_absolute_percentage_error(
            signal_ref[self.df_is_altered], signal_imputed[self.df_is_altered]
        )
        return {"rmse": round(rmse, 4), "mae": round(mae, 4), "wmape": round(wmape, 4)}

    def compare(self):

        results = {}
        for tested_model in self.models_to_test:
            print(type(tested_model).__name__)

            search_space, search_name = utils.get_search_space(
                tested_model, self.search_params
            )

            df = self.df[self.cols_to_impute]
            errors = defaultdict(list)
            for _ in range(self.n_samples):
                random_state = np.random.randint(0, 10 * 9)
                self.create_corruptions(df, random_state=random_state)
                cv = cross_validation.CrossValidation(
                    tested_model,
                    search_space=search_space,
                    search_name=search_name,
                    ratio_missing=self.ratio_missing,
                    corruption=self.corruption,
                )
                # print("# nan before imputation:", df.isna().sum().sum())
                imputed_df = cv.fit_transform(self.corrupted_df)
                # print("# nan after imputation...:", imputed_df.isna().sum().sum())
                for k, v in self.get_errors(df, imputed_df).items():
                    errors[k].append(v)

            results[type(tested_model).__name__] = {
                k: np.mean(v) for k, v in errors.items()
            }

        return results
