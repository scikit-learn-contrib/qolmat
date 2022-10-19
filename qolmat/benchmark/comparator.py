from collections import defaultdict
from math import floor
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qolmat.benchmark import cross_validation, utils
from qolmat.utils import missing_patterns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils import resample
from skopt.space import Categorical, Integer, Real


class Comparator:
    """
    This class implements a comparator for evaluating different imputation methods.

    Parameters
    ----------
    data: pd.DataFrame
        dataframe to impute
    ratio_missing: float
        ratio of articially created corruptions (missign values)
    dict_models: Dict[str, ]
        dictionary of imputation methods
    cols_to_impute: List[str]
        list of column's names to impute
    n_samples: Optional[int] = 1
        number of times the cross-validation is done. By default, the value is set to 1.
    search_params: Optional[Dict[str, Dict[str, Union[str, float, int]]]] = {}
        dictionary of search space for each implementation method. By default, the value is set to {}.
    corruption: Optional[str] = "missing"
        type of corruptions to create: missing or outlier. By default, the value is set to "missing".
    filter_value_nan: Optional[float] = -1e10
    """

    def __init__(
        self,
        data,
        ratio_missing,
        dict_models,
        cols_to_impute,
        n_samples=1,
        search_params={},
        corruption="missing",
        missing_mechanism="MCAR",
        opt=None,
        p_obs=None,
        quantile=None,
        filter_value_nan=-1e10,
    ):

        self.df = data
        self.ratio_missing = ratio_missing
        self.cols_to_impute = cols_to_impute
        self.n_samples = n_samples
        self.filter_value_nan = filter_value_nan
        self.dict_models = dict_models
        self.search_params = search_params
        self.corruption = corruption
        self.missing_mechanism = missing_mechanism
        self.opt = opt
        self.p_obs = p_obs
        self.quantile = quantile

    def create_corruptions(
        self, df: pd.DataFrame  # , random_state: Optional[int] = 29, mode_anomaly="iid"
    ):
        """Create corruption in a dataframe

        Parameters
        ----------
        df : pd.DataFrame
            dataframe to be corrupted
        random_state : Optional[int], optional
            seed used by the ranom number generator, by default 29
        mode_anomaly : str, optional
            way to generate corruptions, by default "iid"
        """

        # self.df_is_altered = utils.choice_with_mask(
        #     df,
        #     df.notna(),
        #     self.ratio_missing,
        #     self.filter_value_nan,
        #     random_state,
        #     mode_anomaly=mode_anomaly,
        # )

        df_corrupted_select = df[self.cols_to_impute].copy()
        res = missing_patterns.produce_NA(
            df_corrupted_select,
            self.ratio_missing,
            mecha=self.missing_mechanism,
            opt=self.opt,
            p_obs=self.p_obs,
            q=self.quantile,
            filter_value=self.filter_value_nan,
        )

        self.df_is_altered = res["mask"]
        if self.corruption == "missing":
            df_corrupted_select[self.df_is_altered] = np.nan
        elif self.corruption == "outlier":
            df_corrupted_select[self.df_is_altered] = np.random.randint(
                0, high=3 * np.max(df), size=(int(len(df) * self.ratio_missing))
            )
        self.df_corrupted = df.copy()
        self.df_corrupted[self.cols_to_impute] = df_corrupted_select

    def get_errors(
        self,
        signal_ref: pd.DataFrame,
        signal_imputed: pd.DataFrame,
    ) -> float:
        """Functions evaluating the reconstruction's quality

        Parameters
        ----------
        signal_ref : pd.DataFrame
            reference/orginal signal
        signal_imputed : pd.DataFrame
            imputed signal

        Returns
        -------
        dictionary
            dictionay of results obtained via different metrics
        """

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

    def compare(self, full: bool = True, verbose: bool = True):
        """Function to compare different imputation methods

        Parameters
        ----------
        full : bool, optional
            _description_, by default True
        verbose : bool, optional
            _description_, by default True
        Returns
        -------
        pd.DataFrame
            dataframe with imputation
        """

        results = {}
        for name, tested_model in self.dict_models.items():
            if verbose:
                print(type(tested_model).__name__)

            search_space = utils.get_search_space(tested_model, self.search_params)

            errors = self.evaluate_errors_sample(tested_model, self.df, search_space)

            results[name] = {k: np.mean(v) for k, v in errors.items()}

        return pd.DataFrame(results)

    def evaluate_errors_sample(
        self, tested_model, df: pd.DataFrame, search_space: Optional[dict] = None
    ):
        """Evaluate the errors in the cross-validation

        Parameters
        ----------
        tested_model : _type_
            imputation model
        df : pd.DataFrame
            dataframe to impute
        search_space : Optional[dict], optional
            search space for tested_model's hyperparameters , by default None

        Returns
        -------
        dict
            dictionary with the errors for eahc metric and at each fold
        """
        errors = defaultdict(list)
        for _ in range(self.n_samples):
            random_state = np.random.randint(0, 10 * 9)
            self.create_corruptions(df)  # , random_state=random_state)
            cv = cross_validation.CrossValidation(
                tested_model,
                search_space=search_space,
                ratio_missing=self.ratio_missing,
                corruption=self.corruption,
            )
            df_imputed_full = cv.fit_transform(self.df_corrupted)
            df_imputed = self.df_corrupted.copy()
            df_imputed[self.cols_to_impute] = df_imputed_full[self.cols_to_impute]
            for metric, value in self.get_errors(df, df_imputed).items():
                errors[metric].append(value)
        return errors
