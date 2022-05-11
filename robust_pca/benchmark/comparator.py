import pandas as pd
import numpy as np
import cross_validation
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
import utils


class Comparator:
    def __init__(
        self,
        data,
        ratio_missing,
        models_to_test,
        cols_to_impute,
        search_params={},
        corruption="missing",
    ):

        self.df = data[cols_to_impute]
        self.ratio_missing = ratio_missing
        self.cols_to_impute = cols_to_impute
        self.models_to_test = models_to_test
        self.search_params = search_params
        self.corruption = corruption

    def create_corruptions(self, df: pd.DataFrame, random_state: Optional[int] = 29):

        # # indices = list(map(tuple, np.argwhere(~np.isnan(df.values))))
        # indices = np.argwhere(df.notna().to_numpy().flatten())
        # print(indices)
        # indices = resample(
        #     indices,
        #     replace=False,
        #     n_samples=floor(len(indices) * self.ratio_missing),
        #     random_state=random_state,
        #     stratify=None,
        # )
        # print(indices)
        # # for i, j in indices:
        # #     print(self.df.iloc[i, j])
        # self.df_is_altered = np.zeros(df.size)
        # print(self.df_is_altered)
        # self.df_is_altered[indices] = 1
        # print(self.df_is_altered)
        # self.df_is_altered = pd.DataFrame(self.df_is_altered.reshape(df.shape), index=df.index, columns=df.columns, dtype=bool)
        # print(self.df_is_altered)

        self.df_is_altered = utils.choice_with_mask(
            df, df.notna(), self.ratio_missing, random_state
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
        print(rmse)

        mae = utils.mean_absolute_error(
            signal_ref[self.df_is_altered], signal_imputed[self.df_is_altered]
        )
        print(mae)

        # rmse = mean_squared_error(
        #     signal_ref.iloc[self.indices], signal_imputed.iloc[self.indices], squared=False
        # )
        # mae = mean_absolute_error(
        #     signal_ref.iloc[self.indices], signal_imputed.iloc[self.indices]
        # )
        # # mape = mean_absolute_percentage_error(signal_ref[self.indices], signal_imputed[self.indices])
        # wmape = np.mean(
        #     np.abs(signal_ref[self.indices] - signal_imputed[self.indices])
        # ) / np.mean(np.abs(signal_ref[self.indices]))
        return {"rmse": rmse, "mae": mae}  # , "wmape": wmape}  # "mape": mape,

    def compare(self):

        results = {}
        for tested_model in self.models_to_test:
            search_space, search_name = utils.get_search_space(
                tested_model, self.search_params
            )
            res_intermediate = {}
            df = self.df[self.cols_to_impute]
            errors = defaultdict(list)
            for _ in range(1):
                random_state = np.random.randint(0, 10 * 9)
                self.create_corruptions(df, random_state=random_state)
                cv = cross_validation.CrossValidation(
                    tested_model,
                    search_space=search_space,
                    search_name=search_name,
                    ratio_missing=self.ratio_missing,
                    corruption=self.corruption,
                )
                imputed_df = cv.fit_transform(self.corrupted_df)
                for k, v in self.get_errors(df, imputed_df).items():
                    errors[k].append(v)

                print(errors)
        results[type(tested_model).__name__] = errors
        return results
