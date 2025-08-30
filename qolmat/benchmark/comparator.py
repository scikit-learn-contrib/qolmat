"""Script for comparator."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from sklearn import utils as sku

from qolmat.benchmark import hyperparameters, metrics
from qolmat.benchmark.missing_patterns import _HoleGenerator

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Comparator:
    """Comparator class.

    This class implements a comparator for evaluating different
    imputation methods.

    Parameters
    ----------
    dict_models: Dict[str, any]
        dictionary of imputation methods
    columnwise_evaluation : Optional[bool], optional
        whether the metric should be calculated column-wise or not,
        by default False
    dict_config_opti: Optional[Dict[str, Dict[str, Union[str, float, int]]]]
        dictionary of search space for each implementation method.
        By default, the value is set to {}.
    max_evals: int = 10
        number of calls of the optimization algorithm
        10.

    """

    def __init__(
        self,
        dict_models: Dict[str, Any],
        generator_holes: _HoleGenerator,
        metrics: List = ["mae", "wmape", "kl_columnwise"],
        dict_config_opti: Optional[Dict[str, Any]] = {},
        metric_optim: str = "mse",
        max_evals: int = 10,
        verbose: bool = False,
    ):
        self.dict_imputers = dict_models
        self.generator_holes = generator_holes
        self.metrics = metrics
        self.dict_config_opti = dict_config_opti
        self.metric_optim = metric_optim
        self.max_evals = max_evals
        self.verbose = verbose

    def get_errors(
        self,
        df_origin: pd.DataFrame,
        df_imputed: pd.DataFrame,
        df_mask: pd.DataFrame,
    ) -> pd.DataFrame:
        """Get errors - estimate the reconstruction's quality.

        Parameters
        ----------
        df_origin : pd.DataFrame
            reference/orginal signal
        df_imputed : pd.DataFrame
            imputed signal
        df_mask : pd.DataFrame
            masked dataframe (NA)

        Returns
        -------
        pd.DataFrame
            DataFrame of results obtained via different metrics

        """
        dict_errors = {}
        for name_metric in self.metrics:
            fun_metric = metrics.get_metric(name_metric)
            dict_errors[name_metric] = fun_metric(
                df_origin, df_imputed, df_mask
            )
        df_errors = pd.concat(dict_errors.values(), keys=dict_errors.keys())
        return df_errors

    def process_split(
        self, split_data: Tuple[int, pd.DataFrame, pd.DataFrame]
    ) -> pd.DataFrame:
        """Process a split.

        Parameters
        ----------
        split_data : Tuple
            contains (split_idx, df_mask, df_origin)

        Returns
        -------
        pd.DataFrame
            errors results

        """
        self.generator_holes.random_state = sku.check_random_state(
            self.generator_holes.random_state
        )
        self.generator_holes.save_rng_state()
        for name, imputer in self.dict_imputers.items():
            self.generator_holes.load_rng_state()
        _, df_mask, df_origin = split_data
        df_with_holes = df_origin.copy()
        df_with_holes[df_mask] = np.nan

        subset = self.generator_holes.subset
        if subset is None:
            raise ValueError(
                "HoleGenerator `subset` should be overwritten in split "
                "but it is none!"
            )

        split_results = {}
        for imputer_name, imputer in self.dict_imputers.items():
            dict_config_opti_imputer = self.dict_config_opti.get(
                imputer_name, {}
            )

            imputer_opti = hyperparameters.optimize(
                imputer,
                df_origin,
                self.generator_holes,
                self.metric_optim,
                dict_config_opti_imputer,
                max_evals=self.max_evals,
                verbose=self.verbose,
            )

            df_imputed = imputer_opti.fit_transform(df_with_holes)
            errors = self.get_errors(
                df_origin[subset], df_imputed[subset], df_mask[subset]
            )
            split_results[imputer_name] = errors

        return pd.concat(split_results, axis=1)

    def process_imputer(
        self, imputer_data: Tuple[str, Any, List[pd.DataFrame], pd.DataFrame]
    ) -> Tuple[str, pd.DataFrame]:
        """Process an imputer.

        Parameters
        ----------
        imputer_data : Tuple[str, Any, List[pd.DataFrame], pd.DataFrame]
            contains (imputer_name, imputer, all_masks, df_origin)

        Returns
        -------
        Tuple[str, pd.DataFrame]
            imputer name, errors results

        """
        imputer_name, imputer, all_masks, df_origin = imputer_data

        subset = self.generator_holes.subset
        if subset is None:
            raise ValueError(
                "HoleGenerator `subset` should be overwritten in split "
                "but it is none!"
            )

        dict_config_opti_imputer = self.dict_config_opti.get(imputer_name, {})
        imputer_opti = hyperparameters.optimize(
            imputer,
            df_origin,
            self.generator_holes,
            self.metric_optim,
            dict_config_opti_imputer,
            max_evals=self.max_evals,
            verbose=self.verbose,
        )

        imputer_results = []
        for i, df_mask in enumerate(all_masks):
            df_with_holes = df_origin.copy()
            df_with_holes[df_mask] = np.nan
            df_imputed = imputer_opti.fit_transform(df_with_holes)
            errors = self.get_errors(
                df_origin[subset], df_imputed[subset], df_mask[subset]
            )
            imputer_results.append(errors)

        return imputer_name, pd.concat(imputer_results).groupby(
            level=[0, 1]
        ).mean()

    def compare(
        self,
        df_origin: pd.DataFrame,
        use_parallel: bool = True,
        n_jobs: int = -1,
        parallel_over: str = "auto",
    ) -> pd.DataFrame:
        """Compare different imputers in parallel with hyperparams opti.

        Parameters
        ----------
        df_origin : pd.DataFrame
            df with missing values
        n_splits : int, optional
            number of 'splits', i.e. fake dataframe with
            artificial holes, by default 10
        use_parallel : bool, optional
            if parallelisation, by default True
        n_jobs : int, optional
            number of jobs to use for the parallelisation, by default -1
        parallel_over : str, optional
            'splits' or 'imputers', by default "auto"

        Returns
        -------
        pd.DataFrame
            DataFrame (2-level index) with results.
            Columsn are imputers.
            0-level index are the metrics.
            1-level index are the column names.

        """
        logging.info(
            f"Starting comparison for {len(self.dict_imputers)} imputers."
        )

        all_splits = list(self.generator_holes.split(df_origin))

        if parallel_over == "auto":
            parallel_over = (
                "splits"
                if len(all_splits) > len(self.dict_imputers)
                else "imputers"
            )

        if use_parallel:
            logging.info(f"Parallelisation over: {parallel_over}...")
            if parallel_over == "splits":
                split_data = [
                    (i, df_mask, df_origin)
                    for i, df_mask in enumerate(all_splits)
                ]
                n_jobs = self.get_optimal_n_jobs(split_data, n_jobs)
                results = Parallel(n_jobs=n_jobs)(
                    delayed(self.process_split)(data) for data in split_data
                )
                final_results = pd.concat(results).groupby(level=[0, 1]).mean()
            elif parallel_over == "imputers":
                imputer_data = [
                    (name, imputer, all_splits, df_origin)
                    for name, imputer in self.dict_imputers.items()
                ]
                n_jobs = self.get_optimal_n_jobs(imputer_data, n_jobs)
                results = Parallel(n_jobs=n_jobs)(
                    delayed(self.process_imputer)(data)
                    for data in imputer_data
                )
                final_results = pd.concat(dict(results), axis=1)
            else:
                raise ValueError(
                    "`parallel_over` should be `auto`, `splits` or `imputers`."
                )

        else:
            logging.info("Sequential treatment...")
            if parallel_over == "splits":
                split_data = [
                    (i, df_mask, df_origin)
                    for i, df_mask in enumerate(all_splits)
                ]
                results = [self.process_split(data) for data in split_data]
                final_results = pd.concat(results).groupby(level=[0, 1]).mean()
            elif parallel_over == "imputers":
                imputer_data = [
                    (name, imputer, all_splits, df_origin)
                    for name, imputer in self.dict_imputers.items()
                ]
                results = [self.process_imputer(data) for data in imputer_data]
                final_results = pd.concat(dict(results), axis=1)
            else:
                raise ValueError(
                    "`parallel_over` should be `auto`, `splits` or `imputers`."
                )

        logging.info("Comparison successfully terminated.")
        return final_results

    @staticmethod
    def get_optimal_n_jobs(split_data: List, n_jobs: int = -1) -> int:
        """Determine the optimal number of parallel jobs to use.

        If `n_jobs` is specified by the user, that value is used.
        Otherwise, the function returns the minimum between the number of
        CPU cores and the number of tasks (i.e., the length of `split_data`),
        ensuring that no more jobs than tasks are launched.

        Parameters
        ----------
        split_data : List
            A collection of data to be processed in parallel.
            The length of this collection determines the number of tasks.
        n_jobs : int
            The number of jobs (parallel workers) to use, by default -1

        Returns
        -------
        int
            The optimal number of jobs to run in parallel

        """
        return min(cpu_count(), len(split_data)) if n_jobs == -1 else n_jobs
