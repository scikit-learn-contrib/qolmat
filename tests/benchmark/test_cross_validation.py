from typing import Dict, Union

import numpy as np
import pandas as pd
import pytest

from qolmat.benchmark import cross_validation
from qolmat.benchmark.missing_patterns import EmpiricalHoleGenerator
from qolmat.imputations.imputers import ImputerRPCA

df_origin = pd.DataFrame({"col1": [0, np.nan, 2, 4, np.nan], "col2": [-1, np.nan, 0.5, 1, 1.5]})
df_imputed = pd.DataFrame({"col1": [0, 1, 2, 3.5, 4], "col2": [-1.5, 0, 1.5, 2, 1.5]})
df_mask = pd.DataFrame(
    {"col1": [False, False, True, False, False], "col2": [True, False, True, True, False]}
)
df_corrupted = df_origin.copy()
df_corrupted[df_mask] = np.nan

imputer_rpca = ImputerRPCA(tau=2, random_state=42, columnwise=True, period=1)
dict_imputers_rpca = {"rpca": imputer_rpca}
generator_holes = EmpiricalHoleGenerator(n_splits=1, ratio_masked=0.5)
dict_config_opti = {
    "rpca": {
        "lam": {
            "col1": {"min": 0.1, "max": 6, "type": "Real"},
            "col2": {"min": 1, "max": 4, "type": "Real"},
        },
        "tol": {"min": 1e-6, "max": 0.1, "type": "Real"},
        "max_iter": {"min": 99, "max": 100, "type": "Integer"},
        "norm": {"categories": ["L1", "L2"], "type": "Categorical"},
    }
}
dict_config_opti_imputer = dict_config_opti["rpca"]
hyperparams_flat = {"lam/col1": 4.7, "lam/col2": 1.5, "tol": 0.07, "max_iter": 100, "norm": "L1"}

cv = cross_validation.CrossValidation(
    imputer=imputer_rpca,
    dict_config_opti_imputer=dict_config_opti_imputer,
    hole_generator=generator_holes,
)

result_params_expected = {
    "lam1": (0.1, 6),
    "lam2": (1, 4),
    "tol": (1e-6, 0.1),
    "max_iter": (99, 100),
    "norm": ("L1", "L2"),
}


@pytest.mark.parametrize("dict_config_opti_imputer", [dict_config_opti_imputer])
@pytest.mark.parametrize("param", ["tol", "max_iter", "norm"])
def test_benchmark_cross_validation_get_dimension(
    dict_config_opti_imputer: Dict, param: str
) -> None:
    result = cross_validation.get_dimension(
        dict_bounds=dict_config_opti_imputer[param], name_dimension=param
    )
    result_expected = result_params_expected[param]
    assert result.bounds == result_expected


@pytest.mark.parametrize("dict_config_opti_imputer", [dict_config_opti_imputer])
def test_benchmark_cross_validation_get_search_space(dict_config_opti_imputer: Dict) -> None:
    list_result = cross_validation.get_search_space(dict_config_opti_imputer)
    list_expected_bounds = list(result_params_expected.values())
    for result, expected_bounds in zip(list_result, list_expected_bounds):
        assert result.bounds == expected_bounds


@pytest.mark.parametrize("hyperparams_flat", [hyperparams_flat])
def test_benchmark_cross_validation_deflat_hyperparams(
    hyperparams_flat: Dict[str, Union[float, int, str]]
) -> None:
    result_deflat = cross_validation.deflat_hyperparams(hyperparams_flat=hyperparams_flat)
    result_expected = {
        "lam": {"col1": 4.7, "col2": 1.5},
        "tol": 0.07,
        "max_iter": 100,
        "norm": "L1",
    }
    assert result_deflat == result_expected


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
    np.testing.assert_allclose(result_cv2, 1.5, atol=1e-5)
    cv.loss_norm = 1
    result_cv1 = cv.loss_function(df_origin=df1, df_imputed=df2, df_mask=df_mask)
    np.testing.assert_allclose(result_cv1, 2.5, atol=1e-5)


@pytest.mark.parametrize("df", [df_corrupted])
def test_benchmark_cross_validation_optimize_hyperparams(df: pd.DataFrame) -> None:
    result_hp = cv.optimize_hyperparams(df)
    result_expected = {
        "lam": {
            "col1": 4.799603622475375,
            "col2": 1.5503043695984915,
        },
        "tol": 0.07796932033627668,
        "max_iter": 100,
        "norm": "L1",
    }
    assert result_hp == result_expected
