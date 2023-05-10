import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from qolmat.benchmark import cross_validation, metrics, utils
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
    n_calls_opt: int = 10
        number of calls of the optimization algorithm
        10.
    """

    dict_metrics: Dict[str, Any] = {
        "mse": metrics.mean_squared_error,
        "rmse": metrics.root_mean_squared_error,
        "mae": metrics.mean_absolute_error,
        "wmape": metrics.weighted_mean_absolute_percentage_error,
        "wasser": metrics.wasser_distance,
        "KL": metrics.kl_divergence_columnwise,
        "ks_test": metrics.kolmogorov_smirnov_test,
        "correlation_diff": metrics.mean_difference_correlation_matrix_numerical_features,
        "pairwise_dist": metrics.sum_pairwise_distances,
        "energy": metrics.sum_energy_distances,
        "frechet": metrics.frechet_distance,
    }

    def __init__(
        self,
        dict_models: Dict[str, Any],
        selected_columns: List[str],
        generator_holes: _HoleGenerator,
        metrics: List = ["mae", "wmape", "KL"],
        search_params: Optional[Dict[str, Dict[str, Union[float, int, str]]]] = {},
        n_calls_opt: int = 10,
    ):
        self.dict_imputers = dict_models
        self.selected_columns = selected_columns
        self.generator_holes = generator_holes
        self.metrics = metrics
        self.search_params = search_params
        self.n_calls_opt = n_calls_opt

    def get_errors(
        self,
        df_origin: pd.DataFrame,
        df_imputed: pd.DataFrame,
        df_mask: pd.DataFrame,
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
        for name_metric in self.metrics:
            dict_errors[name_metric] = Comparator.dict_metrics[name_metric](
                df_origin, df_imputed, df_mask
            )
        errors = pd.concat(dict_errors.values(), keys=dict_errors.keys())
        return errors

    def evaluate_errors_sample(
        self,
        imputer: Any,
        df: pd.DataFrame,
        list_spaces: List[Dict] = [],
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

    def compare(
        self,
        df: pd.DataFrame,
    ):
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
            search_params = self.search_params.get(name, {})

            list_spaces = utils.get_search_space(search_params)

            try:
                dict_errors[name] = self.evaluate_errors_sample(imputer, df, list_spaces)
                print(f"Tested model: {type(imputer).__name__}")
            except Exception as excp:
                print("Error while testing ", type(imputer).__name__)
                raise excp

        df_errors = pd.DataFrame(dict_errors)

        return df_errors


class ComparatorBasedPattern(Comparator):
    def __init__(
        self,
        dict_models: Dict[str, Any],
        selected_columns: List[str],
        generator_holes: _HoleGenerator,
        metrics: List = ["mae", "wmape", "KL"],
        search_params: Optional[Dict[str, Dict[str, Union[float, int, str]]]] = {},
        n_calls_opt: int = 10,
        num_patterns: int = 5,
    ):
        super().__init__(
            dict_models=dict_models,
            selected_columns=selected_columns,
            generator_holes=generator_holes,
            metrics=metrics,
            search_params=search_params,
            n_calls_opt=n_calls_opt,
        )

        self.num_patterns = num_patterns

    def evaluate_errors_sample(
        self,
        imputer: Any,
        df: pd.DataFrame,
        list_spaces: List[Dict] = [],
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
        dfs_pattern = self.get_df_based_pattern(df_origin)
        weights = []
        # Fit then split, or fit in split ?
        self.generator_holes.fit(df_origin)
        for df_pattern in dfs_pattern:
            # Get all columns in pattern
            cols_pattern = df_pattern.dropna(axis=1).columns
            for df_mask in self.generator_holes.split(df_pattern):
                weights.append(len(df_pattern))
                df_corrupted = df_pattern.copy()
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

                subset = self.generator_holes.subset  # columns selected
                subset = [col for col in subset if col in cols_pattern]
                errors = self.get_errors(df_pattern[subset], df_imputed[subset], df_mask[subset])
                list_errors.append(errors)

        df_errors = pd.DataFrame(list_errors)
        # Weighted errors
        errors_mean = df_errors.apply(
            lambda x: (x * np.array(weights)).sum() / np.sum(weights), axis=0
        )
        return errors_mean.sort_index()

    def get_df_based_pattern(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        def get_pattern(row):
            list_col_pattern = [col for col in row.index.to_list() if row[col] == True]
            if len(list_col_pattern) == 0:
                return "_EMPTY_"
            elif len(list_col_pattern) == row.index.size:
                return "_ALLNAN_"
            else:
                return "_".join(list_col_pattern)

        df_isna = df.isna().apply(lambda x: get_pattern(x), axis=1).to_frame(name="pattern")
        df_isna_pattern = df_isna["pattern"].value_counts()

        patterns = df_isna_pattern.index.to_list()
        patterns.remove("_ALLNAN_")
        patterns.remove("_EMPTY_")

        dfs = []
        for idx_pattern in range(min(len(patterns), self.num_patterns)):
            patterns_selected = ["_EMPTY_"] + [patterns[idx_pattern]]
            df_pattern = df.loc[df_isna[df_isna["pattern"].isin(patterns_selected)].index]
            dfs.append(df_pattern)

        return dfs
