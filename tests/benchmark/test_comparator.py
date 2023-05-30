import numpy as np
import pandas as pd
import pytest

from qolmat.benchmark import comparator
from qolmat.imputations.imputers import ImputerMedian, ImputerRPCA
from qolmat.benchmark.missing_patterns import EmpiricalHoleGenerator

df_origin = pd.DataFrame({"col1": [0, np.nan, 2, 4, np.nan], "col2": [-1, np.nan, 0.5, 1, 1.5]})

df_imputed = pd.DataFrame({"col1": [0, 1, 2, 3.5, 4], "col2": [-1.5, 0, 1.5, 2, 1.5]})

df_mask = pd.DataFrame(
    {"col1": [False, False, True, True, False], "col2": [True, False, True, True, False]}
)

cols_to_impute = ["col1", "col2"]
generator_holes = EmpiricalHoleGenerator(n_splits=1, ratio_masked=0.5)
dict_imputers_median = {"median": ImputerMedian()}
dict_imputers_rpca = {"rpca": ImputerRPCA(max_iter=100, tau=2)}
search_params = {"rpca": {"lam": {"min": 0.1, "max": 1, "type": "Real"}}}

comparison_median = comparator.Comparator(
    dict_models=dict_imputers_median,
    selected_columns=cols_to_impute,
    generator_holes=generator_holes,
)

comparison_rpca = comparator.Comparator(
    dict_models=dict_imputers_rpca,
    selected_columns=cols_to_impute,
    generator_holes=generator_holes,
    dict_config_opti=search_params,
)

comparison_bug = comparator.Comparator(
    dict_models=dict_imputers_median,
    selected_columns=["bug"],
    generator_holes=generator_holes,
    dict_config_opti=search_params,
)

result_expected_median = [3.0, 0.5, 0.75, 0.5, 37.88948, 39.68123]
result_expected_rpca = [4.0, 1.0, 1.0, 1.0, 37.60179, 38.98809]

comparison_dict = {"median": comparison_median, "rpca": comparison_rpca, "bug": comparison_bug}
result_expected_dict = {"median": result_expected_median, "rpca": result_expected_rpca}


@pytest.mark.parametrize("df1", [df_origin])
@pytest.mark.parametrize("df2", [df_imputed])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_benchmark_comparator_get_errors(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    result_comparison = comparison_median.get_errors(
        df_origin=df1, df_imputed=df2, df_mask=df_mask
    )
    result = list(result_comparison.values)
    result_expected = [0.25, 0.83333, 0.0625, 1.16666, 18.80089, 36.63671]
    np.testing.assert_allclose(result, result_expected, atol=1e-5)


@pytest.mark.parametrize("df1", [df_origin])
def test_benchmark_comparator_evaluate_errors_sample(df1: pd.DataFrame) -> None:
    result_comparison = comparison_median.evaluate_errors_sample(
        dict_imputers_median["median"], df1
    )
    result = comparison_rpca.evaluate_errors_sample(dict_imputers_rpca["rpca"], df1)
    result = list(result_comparison.values)
    np.testing.assert_allclose(result, result_expected_median, atol=1e-5)


@pytest.mark.parametrize("df1", [df_origin])
@pytest.mark.parametrize("imputer", ["median", "rpca", "bug"])
def test_benchmark_comparator_compare(df1: pd.DataFrame, imputer: str) -> None:
    comparison = comparison_dict[imputer]
    if imputer == "bug":
        np.testing.assert_raises(Exception, comparison.compare, df1)
    else:
        result_comparison = comparison.compare(df1)
        result = list(result_comparison.values.flatten())
        np.testing.assert_allclose(result, result_expected_dict[imputer], atol=1e-5)
