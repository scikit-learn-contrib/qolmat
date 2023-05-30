from typing import Dict, Union
import numpy as np
import pandas as pd
import pytest

from qolmat.benchmark import cross_validation
from qolmat.imputations.imputers import ImputerRPCA
from qolmat.benchmark.missing_patterns import EmpiricalHoleGenerator

df_origin = pd.DataFrame({"col1": [0, np.nan, 2, 4, np.nan], "col2": [-1, np.nan, 0.5, 1, 1.5]})
df_imputed = pd.DataFrame({"col1": [0, 1, 2, 3.5, 4], "col2": [-1.5, 0, 1.5, 2, 1.5]})
df_mask = pd.DataFrame(
    {"col1": [False, False, True, True, False], "col2": [True, False, True, True, False]}
)
df_corrupted = df_origin.copy()
df_corrupted[df_mask] = np.nan

imputer_rpca = ImputerRPCA(tau=2, random_state=42)
dict_imputers_rpca = {"rpca": imputer_rpca}
generator_holes = EmpiricalHoleGenerator(n_splits=1, ratio_masked=0.5)
dict_config_opti = {
    "rpca": {
        "lam": {"min": 0.1, "max": 1, "type": "Real"},
        "max_iter": {"min": 99, "max": 100, "type": "Integer"},
        "norm": {"categories": ["L1", "L2"], "type": "Categorical"},
    }
}
dict_config_opti_imputer = dict_config_opti.get("rpca", {})
hyperparams_flat = {"lam": 0.93382, "max_iter": 100, "norm": "L1"}

cv = cross_validation.CrossValidation(
    imputer=imputer_rpca,
    dict_config_opti_imputer=dict_config_opti_imputer,
    hole_generator=generator_holes,
)

result_params_expected = {"lam": (0.1, 1), "max_iter": (99, 100), "norm": ("L1", "L2")}


@pytest.mark.parametrize("dict_bounds", [dict_config_opti_imputer])
@pytest.mark.parametrize("param", ["lam", "max_iter", "norm"])
def test_benchmark_cross_validation_get_dimension(dict_bounds: Dict, param: str) -> None:
    result = cross_validation.get_dimension(dict_bounds=dict_bounds[param], name_dimension=param)
    result_expected = result_params_expected[param]
    np.testing.assert_equal(result.bounds, result_expected)


@pytest.mark.parametrize("dict_config_opti_imputer", [dict_config_opti_imputer])
def test_benchmark_cross_validation_get_search_space(dict_config_opti_imputer: Dict) -> None:
    list_result = cross_validation.get_search_space(dict_config_opti_imputer)
    result_expected = [
        result_params_expected["lam"],
        result_params_expected["max_iter"],
        result_params_expected["norm"],
    ]
    for i in range(3):
        np.testing.assert_equal(list_result[i].bounds, result_expected[i])


@pytest.mark.parametrize("hyperparams_flat", [hyperparams_flat])
def test_benchmark_cross_validation_deflat_hyperparams(
    hyperparams_flat: Dict[str, Union[float, int, str]]
) -> None:
    resul_deflat = cross_validation.deflat_hyperparams(hyperparams_flat=hyperparams_flat)
    result = list(resul_deflat.values())
    result_expected = [0.93382, 100, "L1"]
    np.testing.assert_equal(result, result_expected)


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
def test_benchmark_cross_validation_optimize_hyperparams(df: pd.DataFrame) -> None:
    result_hp = cv.optimize_hyperparams(df)
    result = list(result_hp.values())
    result_expected = [0.8168886881742098, 99, "L2"]
    np.testing.assert_equal(result, result_expected)


@pytest.mark.parametrize("df", [df_corrupted])
def test_benchmark_cross_validation_fit_transform(df: pd.DataFrame) -> None:
    result_cv = cv.fit_transform(df)
    result = np.array(result_cv)
    result_expected = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 1.5]])
    np.testing.assert_allclose(result, result_expected, atol=1e-5)
