import copy

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from qolmat.benchmark import cross_validation, utils
from qolmat.benchmark.missing_patterns import _HoleGenerator

import logging

logger = logging.getLogger(__file__)

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
    n_cv_calls: Optional[int] = 10
        number of calls of the hyperparameters cross-validation. By default, the value is set to
        10.
    """

    def _set_selected_columns(model, selected_columns):
        model_copy = copy.copy(model)
        if hasattr(model_copy, "selected_columns"):
            model._set_params(selected_columns=selected_columns)
        return model_copy

    def __init__(
        self,
        dict_models: Dict[str, any],
        selected_columns: List[str],
        generator_holes: _HoleGenerator,
        search_params: Optional[Dict[str, Dict[str, Union[float, int, str]]]] = {},
        n_cv_calls: Optional[int] = 10,
    ):

        self.dict_models = dict_models
        self.selected_columns = selected_columns
        self.generator_holes = generator_holes
        self.search_params = search_params
        self.n_cv_calls = n_cv_calls

    def get_errors(
        df_origin: pd.DataFrame, df_imputed: pd.DataFrame, mask: Optional[List[str]],
    ) -> pd.Series:
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
            df_origin[mask],
            df_imputed[mask],
        )
        dict_errors["mae"] = utils.mean_absolute_error(
            df_origin[mask],
            df_imputed[mask],
        )
        dict_errors["wmape"] = utils.weighted_mean_absolute_percentage_error(
            df_origin[mask],
            df_imputed[mask],
        )
        dict_errors["kl"] = utils.kl_divergence(
            df_origin[mask],
            df_imputed[mask],
        )

        errors = pd.DataFrame(dict_errors)
        return errors

    

    def evaluate_errors_sample(
        self, tested_model: any, df: pd.DataFrame, search_space: Optional[dict] = None
    ) -> Dict:
        """Evaluate the errors in the cross-validation

        Parameters
        ----------
        tested_model : any
            imputation model
        df : pd.DataFrame
            dataframe to impute
        search_space : Optional[dict], optional
            search space for tested_model's hyperparameters, by default None

        Returns
        -------
        pd.DataFrame
            DataFrame with the errors for each metric (in column) and at each fold (in index)
        """
        list_errors = []
        df_origin = df[self.selected_columns].copy()
        Comparator._set_selected_columns(tested_model, self.selected_columns)
        
        for df_mask in self.generator_holes.split(df_origin):
            df_corrupted = df_origin.copy()
            df_corrupted[df_mask] = np.nan
            if search_space is None:
                df_imputed = tested_model.fit_transform(X=df_corrupted)
            else:
                cv = cross_validation.CrossValidation(
                    tested_model,
                    search_space=search_space,
                    hole_generator=self.generator_holes,
                    n_calls=self.n_cv_calls,
                )
                df_imputed = cv.fit_transform(X=df_corrupted)

            subset = self.generator_holes.subset
            errors = Comparator.get_errors(df_origin[subset], df_imputed[subset], subset)
            list_errors.append(errors)
        df_errors = pd.concat(list_errors, axis=0)
        errors_mean = df_errors.mean(axis=0)

        return errors_mean

    def compare(self, df: pd.DataFrame):
        """Function to compare different imputation methods on dataframe df

        Parameters
        ----------
        df : pd.DataFrame
        Returns
        -------
        pd.DataFrame
            dataframe with imputation
        """

        dict_errors = {}

        for name_imputer, tested_model in self.dict_models.items():
            logger.info(type(tested_model).__name__)
            
            if name_imputer in self.search_params.keys():
                search_params = self.search_params[name_imputer]
                search_space = utils.get_search_space(tested_model, search_params)

            else:
                search_space = None
            try:
                dict_errors[name_imputer] = self.evaluate_errors_sample(
                    tested_model,
                    df,
                    search_space,
                    )
            except Exception as excp:
                print(type(tested_model).__name__)
                raise excp

        df_errors = pd.DataFrame(dict_errors)

        return df_errors
