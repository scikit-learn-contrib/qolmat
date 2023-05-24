import numpy as np
import pandas as pd
import pytest
import skopt

from qolmat.benchmark import cross_validation
from qolmat.imputations.imputers import ImputerRPCA
from qolmat.benchmark.missing_patterns import EmpiricalHoleGenerator
from qolmat.benchmark.utils import get_search_space

df_origin = pd.DataFrame({"col1": [0, np.nan, 2, 4, np.nan], "col2": [-1, np.nan, 0.5, 1, 1.5]})
df_imputed = pd.DataFrame({"col1": [0, 1, 2, 3.5, 4], "col2": [-1.5, 0, 1.5, 2, 1.5]})
df_mask = pd.DataFrame(
    {"col1": [False, False, True, True, False], "col2": [True, False, True, True, False]}
)

df_corrupted = df_origin.copy()
df_corrupted[df_mask] = np.nan

imputer_rpca = ImputerRPCA(max_iter=100, tau=2)
generator_holes = EmpiricalHoleGenerator(n_splits=1, ratio_masked=0.5)
search_params = {"rpca": {"lam": {"min": 0.1, "max": 1, "type": "Real"}}}
list_spaces = get_search_space(search_params.get("rpca", {}))
cv = cross_validation.CrossValidation(
    imputer=imputer_rpca, list_spaces=list_spaces, hole_generator=generator_holes
)


@pytest.mark.parametrize("df1", [df_origin])
@pytest.mark.parametrize("df2", [df_imputed])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_benchmark_cross_validation_loss_function(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:

    cv.loss_norm = 3
    np.testing.assert_raises(ValueError, cv.loss_function, df1, df2, df_mask)
    cv.loss_norm = 2
    result_cv2 = cv.loss_function(df_origin=df1, df_imputed=df2, df_mask=df_mask)
    np.testing.assert_allclose(result_cv2, 1.58113, atol=1e-5)
    cv.loss_norm = 1
    result_cv1 = cv.loss_function(df_origin=df1, df_imputed=df2, df_mask=df_mask)
    np.testing.assert_allclose(result_cv1, 3, atol=1e-5)


@pytest.mark.parametrize("df", [df_corrupted])
def test_benchmark_cross_validation_deflat_hyperparams(df: pd.DataFrame) -> None:
    res = skopt.gp_minimize(
        cv.objective(df),
        dimensions=cv.list_spaces,
        n_calls=cv.n_calls,
        n_initial_points=max(5, cv.n_calls // 5),
        random_state=42,
        n_jobs=cv.n_jobs,
    )
    hyperparams_flat = {space.name: val for space, val in zip(cv.list_spaces, res["x"])}
    result_hyperparams = cv.deflat_hyperparams(hyperparams_flat)
    result = result_hyperparams["lam"]
    np.testing.assert_allclose(result, 0.816888, atol=1e-5)


@pytest.mark.parametrize("df", [df_corrupted])
@pytest.mark.parametrize("return_hyper_params", [True, False])
def test_benchmark_cross_validation_fit_transform(
    df: pd.DataFrame, return_hyper_params: bool
) -> None:

    if return_hyper_params:
        result_cv, result_hyp = cv.fit_transform(
            df_corrupted, return_hyper_params=return_hyper_params
        )
        np.testing.assert_allclose(result_hyp["lam"], 0.816888, atol=1e-5)
    else:
        result_cv = cv.fit_transform(df_corrupted, return_hyper_params=return_hyper_params)
    result = np.array(result_cv)
    result_expected = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 1.5]])
    np.testing.assert_allclose(result, result_expected, atol=1e-5)
