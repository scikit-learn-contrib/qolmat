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
dict_imputers = {"rpca": ImputerRPCA(max_iter=100, tau=2)}
dict_config_opti = {"rpca": {"lam": {"min": 0.1, "max": 1, "type": "Real"}}}

comparison_rpca = comparator.Comparator(
    dict_models=dict_imputers,
    selected_columns=cols_to_impute,
    generator_holes=generator_holes,
    dict_config_opti=dict_config_opti,
)

comparison_bug = comparator.Comparator(
    dict_models=dict_imputers,
    selected_columns=["bug"],
    generator_holes=generator_holes,
    dict_config_opti=dict_config_opti,
)

dict_comparison = {"rpca": comparison_rpca, "bug": comparison_bug}
index_tuples_expected = pd.MultiIndex.from_product(
    [["mae", "wmape", "KL_columnwise"], ["col1", "col2"]]
)
data_expected = [3.0, 0.5, 0.75, 0.5, 37.88948, 39.68123]
result_expected = pd.Series(data_expected, index=index_tuples_expected)


@pytest.mark.parametrize("df1", [df_origin])
@pytest.mark.parametrize("df2", [df_imputed])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_benchmark_comparator_get_errors(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    result = comparison_rpca.get_errors(df_origin=df1, df_imputed=df2, df_mask=df_mask)
    index_tuples_expected = pd.MultiIndex.from_product(
        [["mae", "wmape", "KL_columnwise"], ["col1", "col2"]]
    )
    result_expected = pd.Series(
        [0.25, 0.83333, 0.0625, 1.16666, 18.80089, 36.63671], index=index_tuples_expected
    )
    np.testing.assert_allclose(result, result_expected, atol=1e-5)


@pytest.mark.parametrize("df1", [df_origin])
def test_benchmark_comparator_evaluate_errors_sample(df1: pd.DataFrame) -> None:
    result = comparison_rpca.evaluate_errors_sample(dict_imputers["rpca"], df1)
    np.testing.assert_allclose(result, result_expected, atol=1e-5)


@pytest.mark.parametrize("df1", [df_origin])
@pytest.mark.parametrize("imputer", ["rpca", "bug"])
def test_benchmark_comparator_compare(df1: pd.DataFrame, imputer: str) -> None:
    comparison = dict_comparison[imputer]
    if imputer == "bug":
        np.testing.assert_raises(Exception, comparison.compare, df_origin)
    else:
        result = comparison.compare(df_origin)
        result_expected_DataFrame = pd.DataFrame(result_expected)
        np.testing.assert_allclose(result, result_expected_DataFrame, atol=1e-5)
