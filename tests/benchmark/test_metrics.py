# ######################
# # Evaluation metrics #
# ######################

import numpy as np
from numpy import random as npr
import pandas as pd
import pytest
import scipy

from qolmat.benchmark import metrics
from qolmat.utils.exceptions import NotEnoughSamples

df_incomplete = pd.DataFrame(
    {"col1": [0, np.nan, 2, 3, np.nan], "col2": [-1, np.nan, 0.5, 1, 1.5]}
)

df_complete = pd.DataFrame({"col1": [0, 2, 2, 3, 4], "col2": [-1, -2, 0.5, 1, 1.5]})

df_imputed = pd.DataFrame({"col1": [0, 1, 2, 3.5, 4], "col2": [-1.5, 0, 1.5, 2, 1.5]})

df_mask = pd.DataFrame(
    {"col1": [False, False, True, True, False], "col2": [True, False, True, True, False]}
)


@pytest.mark.parametrize("df1", [df_incomplete])
@pytest.mark.parametrize("df2", [df_imputed])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_mean_squared_error(df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame) -> None:
    assert metrics.mean_squared_error(df1, df1, df_mask).equals(
        pd.Series([0.0, 0.0], index=["col1", "col2"])
    )
    assert metrics.mean_squared_error(df1, df2, df_mask).equals(
        pd.Series([0.125, 0.750], index=["col1", "col2"])
    )


@pytest.mark.parametrize("df1", [df_incomplete])
@pytest.mark.parametrize("df2", [df_imputed])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_root_mean_squared_error(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    assert metrics.root_mean_squared_error(df1, df1, df_mask).equals(
        pd.Series([0.0, 0.0], index=["col1", "col2"])
    )
    assert (
        metrics.root_mean_squared_error(df1, df2, df_mask)
        .round(3)
        .equals(pd.Series([0.354, 0.866], index=["col1", "col2"]))
    )


@pytest.mark.parametrize("df1", [df_incomplete])
@pytest.mark.parametrize("df2", [df_imputed])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_mean_absolute_error(df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame) -> None:
    assert metrics.mean_absolute_error(df1, df1, df_mask).equals(
        pd.Series([0.0, 0.0], index=["col1", "col2"])
    )
    assert (
        metrics.mean_absolute_error(df1, df2, df_mask)
        .round(3)
        .equals(pd.Series([0.250, 0.833], index=["col1", "col2"]))
    )


@pytest.mark.parametrize("df1", [df_incomplete])
@pytest.mark.parametrize("df2", [df_imputed])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_mean_absolute_percentage_error(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    assert metrics.mean_absolute_percentage_error(df1, df1, df_mask).equals(
        pd.Series([0.0, 0.0], index=["col1", "col2"])
    )
    result = metrics.mean_absolute_percentage_error(df1, df2, df_mask)
    expected = pd.Series([0.083, 1.166], index=["col1", "col2"])
    np.testing.assert_allclose(result, expected, atol=1e-3)


@pytest.mark.parametrize("df1", [df_incomplete])
@pytest.mark.parametrize("df2", [df_imputed])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_weighted_mean_absolute_percentage_error(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    assert metrics.weighted_mean_absolute_percentage_error(df1, df1, df_mask).equals(
        pd.Series([0.0, 0.0], index=["col1", "col2"])
    )
    result = metrics.weighted_mean_absolute_percentage_error(df1, df2, df_mask)
    expected = pd.Series([0.1, 1.0], index=["col1", "col2"])
    np.testing.assert_allclose(result, expected, atol=1e-3)


@pytest.mark.parametrize("df1", [df_incomplete])
@pytest.mark.parametrize("df2", [df_imputed])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_wasserstein_distance(df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame) -> None:
    dist = metrics.dist_wasserstein(df1, df1, df_mask, method="columnwise")
    assert dist.equals(pd.Series([0.0, 0.0], index=["col1", "col2"]))
    dist = metrics.dist_wasserstein(df1, df2, df_mask, method="columnwise")
    assert dist.round(3).equals(pd.Series([0.250, 0.833], index=["col1", "col2"]))


@pytest.mark.parametrize("df1", [df_incomplete])
@pytest.mark.parametrize("df2", [df_imputed])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_kl_divergence_columnwise(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    result = metrics.kl_divergence(df1, df1, df_mask, method="columnwise")
    expected = pd.Series([0.0, 0.0], index=["col1", "col2"])
    np.testing.assert_allclose(result, expected, atol=1e-3)
    result = metrics.kl_divergence(df1, df2, df_mask, method="columnwise")
    expected = pd.Series([18.945, 36.637], index=["col1", "col2"])
    np.testing.assert_allclose(result, expected, atol=1e-3)


@pytest.mark.parametrize("df1", [df_incomplete])
@pytest.mark.parametrize("df2", [df_imputed])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_kl_divergence_gaussian(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    result = metrics.kl_divergence_gaussian(df1, df1, df_mask)
    np.testing.assert_allclose(result, 0, atol=1e-3)

    result = metrics.kl_divergence_gaussian(df1, df2, df_mask)
    np.testing.assert_allclose(result, 1.371, atol=1e-3)


@pytest.mark.parametrize("df1", [df_incomplete])
@pytest.mark.parametrize("df2", [df_imputed])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_kl_divergence_forest(df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame) -> None:
    result = metrics.kl_divergence_forest(df1, df1, df_mask)
    np.testing.assert_allclose(result, 0, atol=1e-3)

    result = metrics.kl_divergence_forest(df1, df2, df_mask)
    np.testing.assert_allclose(result, 6.21e-2, rtol=1e-2)


@pytest.mark.parametrize("df1", [df_incomplete])
@pytest.mark.parametrize("df2", [df_imputed])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_frechet_distance(df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame) -> None:
    result = metrics.frechet_distance(df1, df1, df_mask)
    np.testing.assert_allclose(result, 0, atol=1e-3)

    result = metrics.frechet_distance(df1, df2, df_mask)
    np.testing.assert_allclose(result, 0.253, atol=1e-3)


@pytest.mark.parametrize("df1", [df_incomplete])
@pytest.mark.parametrize("df2", [df_imputed])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_kolmogorov_smirnov_test(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    result = metrics.kolmogorov_smirnov_test(df1, df1, df_mask)
    expected = pd.Series([0, 0], index=["col1", "col2"])
    np.testing.assert_allclose(result, expected, atol=1e-3)

    result = metrics.kolmogorov_smirnov_test(df1, df2, df_mask)
    expected = pd.Series([0.5, 2 / 3], index=["col1", "col2"])
    np.testing.assert_allclose(result, expected, atol=1e-3)


@pytest.mark.parametrize("df1", [df_incomplete])
@pytest.mark.parametrize("df2", [df_imputed])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_sum_pairwise_distances(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    result = metrics.sum_pairwise_distances(df1, df2, df_mask)
    np.testing.assert_allclose(result, 28, atol=1e-3)


@pytest.mark.parametrize("df1", [df_incomplete])
@pytest.mark.parametrize("df2", [df_imputed])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_sum_energy_distances(df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame) -> None:
    sum_distances_df1 = np.sum(
        scipy.spatial.distance.cdist(
            df1[df_mask].fillna(0.0), df1[df_mask].fillna(0.0), metric="cityblock"
        )
    )
    sum_distances_df2 = np.sum(
        scipy.spatial.distance.cdist(
            df2[df_mask].fillna(0.0), df2[df_mask].fillna(0.0), metric="cityblock"
        )
    )
    sum_distances_df1_df2 = np.sum(
        scipy.spatial.distance.cdist(
            df1[df_mask].fillna(0.0), df2[df_mask].fillna(0.0), metric="cityblock"
        )
    )
    energy_distance_scipy = 2 * sum_distances_df1_df2 - sum_distances_df1 - sum_distances_df2
    energy_distance_qolmat = metrics.sum_energy_distances(df1, df2, df_mask)

    assert energy_distance_qolmat.equals(pd.Series(energy_distance_scipy, index=["All"]))


@pytest.mark.parametrize("df1", [df_incomplete])
@pytest.mark.parametrize("df2", [df_imputed])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_mean_difference_correlation_matrix_numerical_features(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    assert metrics.mean_difference_correlation_matrix_numerical_features(df1, df1, df_mask).equals(
        pd.Series([0.0, 0.0], index=["col1", "col2"])
    )
    assert metrics.mean_difference_correlation_matrix_numerical_features(
        df1, df1, df_mask, False
    ).equals(pd.Series([0.0, 0.0], index=["col1", "col2"]))

    assert metrics.mean_difference_correlation_matrix_numerical_features(df1, df2, df_mask).equals(
        pd.Series([0.0, 0.0], index=["col1", "col2"])
    )


df_incomplete_cat = pd.DataFrame(
    {"col1": ["a", np.nan, "a", "b", np.nan], "col2": ["c", np.nan, "d", "b", "d"]}
)

df_imputed_cat = pd.DataFrame(
    {"col1": ["a", "b", "a", "c", "c"], "col2": ["b", "d", "c", "d", "d"]}
)

df_mask_cat = pd.DataFrame(
    {"col1": [False, False, True, True, False], "col2": [True, False, True, True, False]}
)


@pytest.mark.parametrize("df1", [df_incomplete_cat])
@pytest.mark.parametrize("df2", [df_imputed_cat])
@pytest.mark.parametrize("df_mask", [df_mask_cat])
def test_total_variance_distance(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    result = metrics.total_variance_distance(df1, df1, df_mask)
    expected = pd.Series([0.0, 0.0], index=["col1", "col2"])
    np.testing.assert_allclose(result, expected, atol=1e-3)

    result = metrics.total_variance_distance(df1, df2, df_mask)
    expected = pd.Series([1.0, 0], index=["col1", "col2"])
    np.testing.assert_allclose(result, expected, atol=1e-3)


@pytest.mark.parametrize("df1", [df_incomplete_cat])
@pytest.mark.parametrize("df2", [df_imputed_cat])
@pytest.mark.parametrize("df_mask", [df_mask_cat])
def test_mean_difference_correlation_matrix_categorical_features(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    assert metrics.mean_difference_correlation_matrix_categorical_features(
        df1, df1, df_mask
    ).equals(pd.Series([0.0, 0.0], index=["col1", "col2"]))
    assert metrics.mean_difference_correlation_matrix_categorical_features(
        df1, df1, df_mask, False
    ).equals(pd.Series([0.0, 0.0], index=["col1", "col2"]))
    assert metrics.mean_difference_correlation_matrix_categorical_features(
        df1, df2, df_mask
    ).equals(pd.Series([0.0, 0.0], index=["col1", "col2"]))


df_incomplete_cat_num = pd.DataFrame(
    {"col1": ["a", np.nan, "a", "b", np.nan], "col2": [-1, np.nan, 0.5, 1, 1.5]}
)

df_imputed_cat_num = pd.DataFrame(
    {"col1": ["a", "b", "a", "c", "c"], "col2": [-1.5, 0, 1.5, 2, 1.5]}
)

df_mask_cat_num = pd.DataFrame(
    {"col1": [True, False, True, True, False], "col2": [True, False, True, True, False]}
)


@pytest.mark.parametrize("df1", [df_incomplete_cat_num])
@pytest.mark.parametrize("df2", [df_imputed_cat_num])
@pytest.mark.parametrize("df_mask", [df_mask_cat_num])
def test_mean_diff_corr_matrix_categorical_vs_numerical_features(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    assert metrics.mean_diff_corr_matrix_categorical_vs_numerical_features(
        df1, df1, df_mask
    ).equals(pd.Series([0.0], index=["col1"]))
    assert metrics.mean_diff_corr_matrix_categorical_vs_numerical_features(
        df1, df1, df_mask, False
    ).equals(pd.Series([0.0], index=["col1"]))
    assert metrics.mean_diff_corr_matrix_categorical_vs_numerical_features(
        df1, df2, df_mask
    ).equals(pd.Series([0.07009774198932273], index=["col1"]))


df_imputed_bad_shape = pd.DataFrame({"col1": [0, 1, 2, 3.5, 4]})


@pytest.mark.parametrize("df1", [df_incomplete])
@pytest.mark.parametrize("df2", [df_imputed_bad_shape])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_exception_raise_different_shapes(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    with pytest.raises(Exception):
        metrics.mean_difference_correlation_matrix_numerical_features(df1, df2, df_mask)
    with pytest.raises(Exception):
        metrics.frechet_distance(df1, df2, df_mask)


@pytest.mark.parametrize("df1", [df_incomplete_cat])
@pytest.mark.parametrize("df2", [df_imputed_cat])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_exception_raise_no_numerical_column_found(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    with pytest.raises(Exception):
        metrics.kolmogorov_smirnov_test(df1, df2, df_mask)
    with pytest.raises(Exception):
        metrics.mean_difference_correlation_matrix_numerical_features(df1, df2, df_mask)


@pytest.mark.parametrize("df1", [df_incomplete])
@pytest.mark.parametrize("df2", [df_imputed])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_exception_raise_no_categorical_column_found(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    with pytest.raises(Exception):
        metrics.total_variance_distance(df1, df2, df_mask)


df_incomplete_cat_num_bad = pd.DataFrame(
    {"col1": ["a", np.nan, "c", "b", np.nan], "col2": [-1, np.nan, 0.5, 0.5, 1.5]}
)


@pytest.mark.parametrize("df1", [df_incomplete_cat_num])
@pytest.mark.parametrize("df2", [df_incomplete_cat_num_bad])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_value_error_get_correlation_f_oneway_matrix(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    assert metrics.mean_diff_corr_matrix_categorical_vs_numerical_features(
        df1, df2, df_mask
    ).equals(pd.Series([np.nan], index=["col1"]))


@pytest.mark.parametrize("df1", [df_incomplete])
@pytest.mark.parametrize("df2", [df_imputed])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_distance_anticorr(df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame) -> None:
    result = metrics.distance_anticorr(df1, df2, df_mask)
    np.testing.assert_allclose(result, 1.1e-4, rtol=1e-2)


@pytest.mark.parametrize("df1", [df_incomplete])
@pytest.mark.parametrize("df2", [df_imputed])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_pattern_based_weighted_mean_metric(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    with pytest.raises(NotEnoughSamples):
        metrics.pattern_based_weighted_mean_metric(
            df1, df2, df_mask, metric=metrics.distance_anticorr, min_n_rows=5
        )

    expected = pd.Series([1.1e-4], index=["All"])
    result = metrics.pattern_based_weighted_mean_metric(
        df1, df2, df_mask, metric=metrics.distance_anticorr, min_n_rows=1
    )
    np.testing.assert_allclose(result, expected, rtol=1e-2)


rng = npr.default_rng(123)
df_gauss1 = pd.DataFrame(rng.multivariate_normal([0, 0], [[1, 0.2], [0.2, 2]], size=100))
df_gauss2 = pd.DataFrame(rng.multivariate_normal([0, 1], [[1, 0.2], [0.2, 2]], size=100))
df_mask = pd.DataFrame(np.full_like(df_gauss1, True))


def test_pattern_mae_comparison() -> None:
    def fun_mean_mae(df_gauss1, df_gauss2, df_mask) -> float:
        return metrics.mean_squared_error(df_gauss1, df_gauss2, df_mask).mean()

    result1 = fun_mean_mae(df_gauss1, df_gauss2, df_mask)
    result2 = metrics.pattern_based_weighted_mean_metric(
        df_gauss1, df_gauss2, df_mask, metric=fun_mean_mae, min_n_rows=1
    )
    np.testing.assert_allclose(result1, result2, rtol=1e-2)
