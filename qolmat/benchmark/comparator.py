import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from qolmat.benchmark import cross_validation, utils
from qolmat.benchmark.missing_patterns import _HoleGenerator


class Comparator:
    """
    This class implements a comparator for evaluating different imputation methods.

    Parameters
    ----------
    dict_models: Dict[str, any]
        dictionary of imputation methods
    selected_columns: List[str]
        list of column's names selected (all with at least one null value will be imputed)
    columnwise_evaluation : Optional[bool], optional
        whether the metric should be calculated column-wise or not, by default False
    search_params: Optional[Dict[str, Dict[str, Union[str, float, int]]]] = {}
        dictionary of search space for each implementation method. By default, the value is set to
        {}.
    n_calls_opt: Optional[int] = 10
        number of calls of the optimization algorithm
        10.
    """

    def __init__(
        self,
        dict_models: Dict[str, any],
        selected_columns: List[str],
        generator_holes: _HoleGenerator,
        search_params: Optional[Dict[str, Dict[str, Union[float, int, str]]]] = {},
        n_calls_opt: Optional[int] = 10,
    ):
        self.dict_imputers = dict_models
        self.selected_columns = selected_columns
        self.generator_holes = generator_holes
        self.search_params = search_params
        self.n_calls_opt = n_calls_opt

    def get_errors(
        self, df_origin: pd.DataFrame, df_imputed: pd.DataFrame, df_mask: pd.DataFrame
    ) -> pd.DataFrame:
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

        dict_errors = {}
        dict_errors["rmse"] = utils.root_mean_squared_error(
            df_origin[df_mask],
            df_imputed[df_mask],
        )
        dict_errors["mae"] = utils.mean_absolute_error(
            df_origin[df_mask],
            df_imputed[df_mask],
        )
        dict_errors["wmape"] = utils.weighted_mean_absolute_percentage_error(
            df_origin[df_mask],
            df_imputed[df_mask],
        )

        dict_errors["kl"] = utils.kl_divergence(
            df_origin[df_mask],
            df_imputed[df_mask],
        )

        errors = pd.concat(dict_errors.values(), keys=dict_errors.keys())
        return errors

    def evaluate_errors_sample(
        self, imputer: any, df: pd.DataFrame, list_spaces: List[Dict] = {}
    ) -> pd.Series:
        """Evaluate the errors in the cross-validation

        Parameters
        ----------
        tested_model : any
            imputation model
        df : pd.DataFrame
            dataframe to impute
        search_space : Dict
            search space for tested_model's hyperparameters

        Returns
        -------
        pd.DataFrame
            DataFrame with the errors for each metric (in column) and at each fold (in index)
        """
        list_errors = []
        df_origin = df[self.selected_columns].copy()
        for df_mask in self.generator_holes.split(df_origin):
            df_corrupted = df_origin.copy()
            df_corrupted[df_mask] = np.nan
            if list_spaces:
                cv = cross_validation.CrossValidation(
                    imputer,
                    list_spaces=list_spaces,
                    hole_generator=self.generator_holes,
                    n_calls=self.n_calls_opt,
                )
                df_imputed = cv.fit_transform(df_corrupted)
            else:
                df_imputed = imputer.fit_transform(df_corrupted)

            subset = self.generator_holes.subset
            errors = self.get_errors(df_origin[subset], df_imputed[subset], df_mask[subset])
            list_errors.append(errors)
        df_errors = pd.DataFrame(list_errors)
        errors_mean = df_errors.mean(axis=0)

        return errors_mean

    def compare(self, df: pd.DataFrame, verbose: bool = True):
        """Function to compare different imputation methods on dataframe df

        Parameters
        ----------
        df : pd.DataFrame
        verbose : bool, optional
            _description_, by default True
        Returns
        -------
        pd.DataFrame
            dataframe with imputation
        """

        dict_errors = {}

        for name, imputer in self.dict_imputers.items():
            print(f"Tested model: {type(imputer).__name__}")

            search_params = self.search_params.get(name, {})

            list_spaces = utils.get_search_space(search_params)

            try:
                dict_errors[name] = self.evaluate_errors_sample(imputer, df, list_spaces)
            except Exception as excp:
                print("Error while testing ", type(imputer).__name__)
                raise excp

        df_errors = pd.DataFrame(dict_errors)

        return df_errors
