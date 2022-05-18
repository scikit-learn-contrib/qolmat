import pandas as pd
import numpy as np
import skopt
from typing import Optional
from sklearn.utils import resample
from math import floor
from robust_pca.benchmark import utils


class CrossValidation:
    def __init__(
        self,
        model,
        search_space=None,
        search_name=None,
        value_params=None,
        cv=2,
        loss_norm="1",
        ratio_missing=0.1,
        corruption="missing",
    ):
        self.model = model
        self.search_space = search_space
        self.search_name = search_name
        self.value_params = value_params
        self.cv = cv
        self.loss_norm = loss_norm
        self.ratio_missing = ratio_missing
        self.corruption = corruption

    def create_corruptions(self, df: pd.DataFrame, random_state: Optional[int] = 129):

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

    def loss_function(self, initial, imputed):
        return np.sum(abs(initial[self.df_is_altered] - imputed[self.df_is_altered]))

    def objective(self, args):
        if hasattr(self.model, "rpca"):
            for param_name, param_value in zip(self.search_name, args):
                setattr(self.model.rpca, param_name, param_value)
        else:
            for param_name, param_value in zip(self.search_name, args):
                setattr(self.model, param_name, param_value)

        errors = []
        for _ in range(self.cv):
            self.create_corruptions(self.signal)
            imputed = self.model.fit_transform(self.corrupted_df)
            errors.append(self.loss_function(self.signal, imputed))

        return np.nanmean(errors)

    def fit_transform(self, signal: pd.Series) -> pd.Series:
        self.signal = signal

        if self.search_space is None:
            imputed_signal = self.model.fit_transform(self.signal)

        else:
            res = skopt.gp_minimize(
                self.objective,
                self.search_space,
                n_calls=10,
                random_state=42,
                n_jobs=-1,
            )

            for param_name, param_value in zip(self.search_name, res.x):
                setattr(self.model, param_name, param_value)

            imputed_signal = self.model.fit_transform(self.signal)

        return imputed_signal
