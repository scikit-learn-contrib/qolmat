from collections import defaultdict
from typing import Optional, Dict, List
from numpy.typing import ArrayLike

import numpy as np
import pandas as pd
from qolmat.benchmark import cross_validation, utils
from qolmat.benchmark.missing_patterns import HoleGenerator


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
    """

    def __init__(
        self,
        data: ArrayLike,
        dict_models: Dict,
        cols_to_impute: List[str],
        generated_holes: HoleGenerator,
        search_params: Optional[Dict] = {},
        columnwise_evaluation: Optional[bool] = True,
        cv_folds: Optional[int] = 5,
    ):

        self.df = data
        self.cols_to_impute = cols_to_impute
        self.dict_models = dict_models
        self.generated_holes = generated_holes
        self.search_params = search_params
        self.columnwise_evaluation = columnwise_evaluation
        self.cv_folds = cv_folds

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
            columnwise_evaluation=self.columnwise_evaluation,
        )
        mae = utils.mean_absolute_error(
            signal_ref[self.df_is_altered],
            signal_imputed[self.df_is_altered],
            columnwise_evaluation=self.columnwise_evaluation,
        )
        wmape = utils.weighted_mean_absolute_percentage_error(
            signal_ref[self.df_is_altered],
            signal_imputed[self.df_is_altered],
            columnwise_evaluation=self.columnwise_evaluation,
        )
        kl = utils.kl_divergence(
            signal_ref,
            signal_imputed,
            columnwise_evaluation=self.columnwise_evaluation,
        )
        if self.columnwise_evaluation:
            wd = utils.wasser_distance(
                signal_ref,
                signal_imputed,
            )
        if not self.columnwise_evaluation and signal_ref.shape[1] > 1:
            frechet = utils.frechet_distance(
                signal_ref,
                signal_imputed,
                normalized=False,
            )

        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "wmape": round(wmape, 4),
            "KL": round(kl, 4),
            **(
                {
                    "wasserstein": round(wd, 4),
                }
                if self.columnwise_evaluation
                else {}
            ),
            **(
                {
                    "frechet distance": round(frechet, 4),
                }
                if not self.columnwise_evaluation and signal_ref.shape[1] > 1
                else {}
            ),
        }

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
        dict
            dictionary with the errors for eahc metric and at each fold
        """

        errors = defaultdict(list)

        for df_mask in self.generated_holes.split(df):

            self.df_is_altered = df_mask
            self.df_corrupted = df[df_mask]

            df_imputed = self.df_corrupted.copy()
            if search_space is None:
                df_imputed[self.cols_to_impute] = tested_model.fit_transform(
                    self.df_corrupted
                )[self.cols_to_impute]
            else:
                cv = cross_validation.CrossValidation(
                    tested_model,
                    search_space=search_space,
                    hole_generator=self.generated_holes,
                    cv_folds=self.cv_folds,
                )
                df_imputed[self.cols_to_impute] = cv.fit_transform(self.df_corrupted)[
                    self.cols_to_impute
                ]

            dict_errors = self.get_errors(
                df[self.cols_to_impute], df_imputed[self.cols_to_impute]
            )
            for metric, value in dict_errors.items():
                errors[metric].append(value)

        return errors

    def compare(self, verbose: bool = True):
        """Function to compare different imputation methods

        Parameters
        ----------
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

            if self.columnwise_evaluation:
                results[name] = {
                    k: pd.concat([v1 for v1 in v], axis=1).mean(axis=1).to_dict()
                    for k, v in errors.items()
                }
            else:
                results[name] = {k: np.mean(v) for k, v in errors.items()}

        if self.columnwise_evaluation:
            results_d = {}
            for k, v in results.items():
                results_d[k] = pd.DataFrame(v).T
            return (
                pd.concat(results_d.values(), keys=results_d.keys())
                .swaplevel()
                .sort_index(0)
            )
        else:
            return pd.DataFrame(results)
