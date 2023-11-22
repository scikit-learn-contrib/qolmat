import numpy as np
import pandas as pd
from qolmat.benchmark import missing_patterns
from qolmat.benchmark.comparator import Comparator, get_errors
from qolmat.imputations import imputers


def test_get_errors():
    df_origin = pd.DataFrame({"col1": [np.nan, 2, 3], "col2": [0, -1, np.nan]})
    df_imputed = pd.DataFrame({"col1": [0, 1, 3], "col2": [0, -1, 2]})
    df_mask = pd.DataFrame({"col1": [False, True, False], "col2": [True, False, False]})
    list_metrics = ["mae", "wmape", "KL_columnwise"]
    result = get_errors(df_origin, df_imputed, df_mask, list_metrics)
    assert result.shape == (6,)
    assert len(result.index.names) == 2


def test_comparator_ealuate_errors_sample():
    df = pd.DataFrame({"col1": [np.nan, 2, 3], "col2": [0, -1, np.nan]})
    imputer_mean = imputers.ImputerMean()
    imputer_median = imputers.ImputerMedian()
    dict_models = {"mean": imputer_mean, "meadian": imputer_median}
    selected_columns = ["col1", "col2"]
    generator_holes = missing_patterns.EmpiricalHoleGenerator(n_splits=2, random_state=42)

    compar = Comparator(dict_models, selected_columns, generator_holes)
    result = compar.evaluate_errors_sample(imputer_mean, df)
    print(result)


# import numpy as np
# import pandas as pd
# import pytest

# from qolmat.benchmark import comparator
# from qolmat.imputations.imputers import ImputerMedian, ImputerRPCA
# from qolmat.benchmark.missing_patterns import EmpiricalHoleGenerator
# import hyperopt as ho

# df_origin = pd.DataFrame({"col1": [0, np.nan, 2, 4, np.nan], "col2": [-1, np.nan, 0.5, 1, 1.5]})
# df_imputed = pd.DataFrame({"col1": [0, 1, 2, 3.5, 4], "col2": [-1.5, 0, 1.5, 2, 1.5]})
# df_mask = pd.DataFrame(
#     {"col1": [False, False, True, True, False], "col2": [True, False, True, True, False]}
# )

# cols_to_impute = ["col1", "col2"]
# generator_holes = EmpiricalHoleGenerator(n_splits=1, ratio_masked=0.5)
# dict_imputers = {"rpca": ImputerRPCA(max_iterations=100, tau=2)}
# dict_config_opti = {"rpca": {"lam": ho.hp.uniform("lam", low=0.1, high=1)}}

# comparison_rpca = comparator.Comparator(
#     dict_models=dict_imputers,
#     selected_columns=cols_to_impute,
#     generator_holes=generator_holes,
#     dict_config_opti=dict_config_opti,
# )

# comparison_bug = comparator.Comparator(
#     dict_models=dict_imputers,
#     selected_columns=["bug"],
#     generator_holes=generator_holes,
#     dict_config_opti=dict_config_opti,
# )

# dict_comparison = {"rpca": comparison_rpca, "bug": comparison_bug}
# index_tuples_expected = pd.MultiIndex.from_product(
#     [["mae", "wmape", "KL_columnwise"], ["col1", "col2"]]
# )
# # data_expected = [3.0, 0.5, 0.75, 0.5, 37.88948, 39.68123]
# data_expected = [4.467175, 7.467187, 1.116794, 7.467187, 37.491336, 36.977574]
# result_expected = pd.Series(data_expected, index=index_tuples_expected)


# @pytest.mark.parametrize("df1", [df_origin])
# @pytest.mark.parametrize("df2", [df_imputed])
# @pytest.mark.parametrize("df_mask", [df_mask])
# def test_comparator_get_errors(
#     df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
# ) -> None:
#     result = comparison_rpca.get_errors(df_origin=df1, df_imputed=df2, df_mask=df_mask)
#     assert isinstance(result, pd.Series)
#     pd.testing.assert_index_equal(result.index, index_tuples_expected)
#     assert result.notna().all()


# @pytest.mark.parametrize("df", [df_origin])
# def test_comparator_evaluate_errors_sample(df: pd.DataFrame) -> None:
#     result = comparison_rpca.evaluate_errors_sample(dict_imputers["rpca"], df)
#     assert isinstance(result, pd.Series)
#     pd.testing.assert_index_equal(result.index, index_tuples_expected)
#     assert result.notna().all()


# @pytest.mark.parametrize("df", [df_origin])
# @pytest.mark.parametrize("imputer", ["rpca", "bug"])
# def test_comparator_compare(df: pd.DataFrame, imputer: str) -> None:
#     comparison = dict_comparison[imputer]
#     if imputer == "bug":
#         np.testing.assert_raises(Exception, comparison.compare, df)
#     else:
#         result = comparison.compare(df)
#         assert isinstance(result, pd.DataFrame)
#         pd.testing.assert_index_equal(result.index, index_tuples_expected)
#         assert result.notna().all().all()
