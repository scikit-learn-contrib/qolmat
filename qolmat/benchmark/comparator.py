from collections import defaultdict
from math import floor
from typing import Optional, Dict, List
from numpy.typing import ArrayLike

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
        data: ArrayLike,
        ratio_missing: float,
        dict_models: Dict,
        cols_to_impute: List[str],
        n_samples: Optional[int] = 1,
        search_params: Optional[Dict] = {},
        corruption: Optional[str] = "missing",
        missing_mechanism: Optional[str] = "MCAR",
        opt: Optional[float] = None,
        p_obs: Optional[float] = None,
        quantile: Optional[float] = None,
        filter_value_nan: Optional[str] = -1e10,
        columnwise: Optional[bool] = True,
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
        self.columnwise = columnwise

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
            columnwise=self.columnwise,
        )
        mae = utils.mean_absolute_error(
            signal_ref[self.df_is_altered],
            signal_imputed[self.df_is_altered],
            columnwise=self.columnwise,
        )
        wmape = utils.weighted_mean_absolute_percentage_error(
            signal_ref[self.df_is_altered],
            signal_imputed[self.df_is_altered],
            columnwise=self.columnwise,
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

            if self.columnwise:
                results[name] = {
                    k: pd.concat([v1 for v1 in v], axis=1).mean(axis=1).to_dict()
                    for k, v in errors.items()
                }
            else:
                results[name] = {k: np.mean(v) for k, v in errors.items()}

        if self.columnwise:
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

    def evaluate_errors_sample(
        self, tested_model, df: pd.DataFrame, search_space: Optional[dict] = None
    ) -> Dict:
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
            self.create_corruptions(df)
            cv = cross_validation.CrossValidation(
                tested_model,
                search_space=search_space,
                ratio_missing=self.ratio_missing,
                corruption=self.corruption,
            )
            df_imputed = self.df_corrupted.copy()
            df_imputed[self.cols_to_impute] = cv.fit_transform(self.df_corrupted)[
                self.cols_to_impute
            ]

            # df_imputed[self.cols_to_impute] = df_imputed_full[self.cols_to_impute]
            for metric, value in self.get_errors(df, df_imputed).items():
                errors[metric].append(value)
        return errors


from sklearn.model_selection import GroupShuffleSplit


class ComparatorGroups(Comparator):
    def __init__(
        self,
        data: ArrayLike,
        ratio_missing: float,
        dict_models: Dict,
        cols_to_impute: List[str],
        n_samples: Optional[int] = 1,
        search_params: Optional[Dict] = {},
        corruption: Optional[str] = "missing",
        missing_mechanism: Optional[str] = "MCAR",
        opt: Optional[float] = None,
        p_obs: Optional[float] = None,
        quantile: Optional[float] = None,
        filter_value_nan: Optional[str] = -1e10,
        columnwise: Optional[bool] = True,
        column_groups: Optional[List[str]] = [],
    ) -> None:
        super().__init__(
            data,
            ratio_missing,
            dict_models,
            cols_to_impute,
            n_samples,
            search_params,
            corruption,
            missing_mechanism,
            opt,
            p_obs,
            quantile,
            filter_value_nan,
            columnwise,
        )
        try:
            self.column_groups = column_groups
        except:
            raise ValueError("No column_groups passed!")

    def create_groups(self):
        self.groups = self.df.groupby(self.column_groups).ngroup().values

        if self.n_samples > len(np.unique(self.groups)):
            raise ValueError("n_samples has to be smaller than the number of groups.")

    def evaluate_errors_sample(
        self, tested_model, df: pd.DataFrame, search_space: Optional[dict] = None
    ) -> Dict:

        errors = defaultdict(list)

        self.create_groups()

        gss = GroupShuffleSplit(
            n_splits=self.n_samples,
            train_size=0.7,
            random_state=42,
        )
        for _, (observed_indices, missing_indices) in enumerate(
            gss.split(X=df, y=None, groups=self.groups)
        ):
            # create the boolean mask of missing values
            self.df_is_altered = pd.DataFrame(
                data=np.full((self.df[self.cols_to_impute].shape), True),
                columns=self.cols_to_impute,
                index=self.df.index,
            )
            self.df_is_altered.iloc[observed_indices, :] = False

            # create the corrupted (with artificial missing values) dataframe
            self.df_corrupted = self.df[self.cols_to_impute].copy()
            self.df_corrupted.iloc[missing_indices, :] = np.nan

            cv = cross_validation.CrossValidation(
                tested_model,
                search_space=search_space,
                ratio_missing=self.ratio_missing,
                corruption=self.corruption,
            )
            df_imputed = self.df_corrupted.copy()
            df_imputed[self.cols_to_impute] = cv.fit_transform(self.df_corrupted)[
                self.cols_to_impute
            ]

            for metric, value in self.get_errors(
                df[self.cols_to_impute], df_imputed
            ).items():
                errors[metric].append(value)
        return errors
