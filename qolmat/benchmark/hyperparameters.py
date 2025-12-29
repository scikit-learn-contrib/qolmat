"""Script for hyperparameter optimisation."""

import copy
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Dimension

from qolmat.benchmark import metrics
from qolmat.benchmark.missing_patterns import _HoleGenerator
from qolmat.imputations.imputers import _Imputer


def get_objective(
    imputer: _Imputer,
    df: pd.DataFrame,
    generator: _HoleGenerator,
    metric: str,
    names_hyperparams: List[str],
) -> Callable:
    """Define the objective function.

    This is the average metric computed over the folds provided by
    the hole generator, using a cross-validation.

    Parameters
    ----------
    imputer: _Imputer
        Imputer to optimize
    df : pd.DataFrame
        Input dataframe
    generator: _HoleGenerator
        Generator creating masked values for cross-validation
    metric: str
        Metric used as performance indicator (e.g., 'mse', 'mae')
    names_hyperparams: List[str]
        Names of hyperparameters being optimized

    Returns
    -------
    Callable
        Objective function returning mean error across folds

    """

    def fun_obj(args: List) -> float:
        # Set hyperparameters on imputer
        for key, value in zip(names_hyperparams, args):
            setattr(imputer, key, value)

        list_errors = []
        for df_mask in generator.split(df):
            df_origin = df.copy()
            df_corrupted = df_origin.copy()
            df_corrupted[df_mask] = np.nan
            df_imputed = imputer.fit_transform(df_corrupted)

            subset = generator.subset
            fun_metric = metrics.get_metric(metric)
            errors = fun_metric(df_origin[subset], df_imputed[subset], df_mask[subset])
            list_errors.append(errors)

        return float(np.mean(list_errors))

    return fun_obj


def optimize(
    imputer: _Imputer,
    df: pd.DataFrame,
    generator: _HoleGenerator,
    metric: str,
    dict_config: Dict[str, Dimension],
    max_evals: int = 100,
    verbose: bool = False,
    random_state: int = 42,
) -> _Imputer:
    """Optimize imputer hyperparameters using Bayesian optimization.

    Parameters
    ----------
    imputer: _Imputer
        Imputer to optimize
    df : pd.DataFrame
        Input dataframe
    generator: _HoleGenerator
        Generator for cross-validation
    metric: str
        Metric to minimize (e.g., 'mse', 'mae')
    dict_config: Dict[str, Dimension]
        Search space: keys are hyperparameter names, values are skopt Dimension objects
        (Real, Integer, or Categorical)
    max_evals: int, default=100
        Maximum number of evaluations
    verbose: bool, default=False
        Verbosity flag
    random_state: int, default=42
        Random seed for reproducibility

    Returns
    -------
    _Imputer
        Imputer with optimized hyperparameters

    """
    print("Starting hyperparameter optimization...")
    imputer = copy.deepcopy(imputer)

    if not dict_config:
        return imputer

    names_hyperparams = list(dict_config.keys())
    dimensions = list(dict_config.values())

    # Update imputer_params to include optimized parameters
    imputer.imputer_params = tuple(set(imputer.imputer_params) | set(names_hyperparams))

    # Disable verbose during optimization if applicable
    if verbose and hasattr(imputer, "verbose"):
        setattr(imputer, "verbose", False)

    fun_obj = get_objective(imputer, df, generator, metric, names_hyperparams)

    result = gp_minimize(
        func=fun_obj,
        dimensions=dimensions,
        n_calls=max_evals,
        n_initial_points=min(10, max_evals),
        verbose=verbose,
        random_state=random_state,
    )

    # Set optimal hyperparameters
    for key, value in zip(names_hyperparams, result.x):
        setattr(imputer, key, value)

    return imputer
