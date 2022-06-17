import pandas as pd
import numpy as np
import skopt
from typing import Optional
from sklearn.utils import resample
from math import floor
from qolmat.benchmark import utils


class CrossValidation:
    def __init__(
        self,
        model,
        search_space=None,
        cv=2,
        n_calls=10,
        n_jobs = -1,
        loss_norm = 1,
        ratio_missing=0.1,
        corruption="missing",
    ):
        self.model = model
        self.search_space = search_space
        self.cv = cv
        self.n_calls = n_calls
        self.n_jobs = n_jobs
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
        initial_nan = np.where(self.df_is_altered, np.nan, initial)
        imputed_nan = np.where(self.df_is_altered, np.nan, imputed)
        if self.loss_norm == 1:
            return np.nansum(np.abs(initial_nan - imputed_nan))
        elif self.loss_norm == 2:
            return np.sqrt(np.nansum(np.square(initial_nan - imputed_nan)))
        else:
            raise ValueError("loss_norm has to be 0 or 1 (int)")
        

    def _set_params(self, all_params):
        if hasattr(self.model, "rpca"):
                for param_name, param_value in all_params.items():
                    setattr(self.model.rpca, param_name, param_value)
        else:
            for param_name, param_value in all_params.items():
                setattr(self.model, param_name, param_value)

    def objective(self):
        @skopt.utils.use_named_args(self.search_space)
        def obj_func(**all_params):
            
            self._set_params(all_params=all_params)
            print(all_params)
            errors = []
            for _ in range(self.cv):
                self.create_corruptions(self.signal)
                imputed = self.model.fit_transform(self.corrupted_df)
                error = self.loss_function(self.signal, imputed)
                errors.append(error)
            
            mean_errors = np.mean(errors)
            print(mean_errors)
            return mean_errors
        return obj_func

    def fit_transform(self, signal: pd.Series, return_hyper_params = False) -> pd.Series:
        self.signal = signal

        if self.search_space is None:
            imputed_signal = self.model.fit_transform(self.signal)

        else:
            res = skopt.gp_minimize(
                self.objective(),
                self.search_space,
                n_calls=self.n_calls,
                n_initial_points = self.n_calls//5,
                random_state=42,
                n_jobs=self.n_jobs,
            )
            best_params = {self.search_space[param].name: res["x"][param] for param in range(len(res["x"]))}
            self._set_params(all_params=best_params)
            imputed_signal = self.model.fit_transform(self.signal)
        
        res = [imputed_signal]

        if return_hyper_params:
            res += [best_params]
        return tuple(res)
