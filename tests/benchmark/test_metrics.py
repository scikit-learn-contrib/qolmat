# ######################
# # Evaluation metrics #
# ######################

import numpy as np
import pandas as pd
import pytest
import scipy

from qolmat.benchmark import metrics

df_incomplete = pd.DataFrame(
    {"col1": [0, np.nan, 2, 3, np.nan], "col2": [-1, np.nan, 0.5, 1, 1.5]}
)

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
def test_weighted_mean_absolute_percentage_error(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    assert metrics.weighted_mean_absolute_percentage_error(df1, df1, df_mask).equals(
        pd.Series([0.0, 0.0], index=["col1", "col2"])
    )
    result = metrics.weighted_mean_absolute_percentage_error(df1, df2, df_mask)
    expected = pd.Series([0.083, 1.167], index=["col1", "col2"])
    np.testing.assert_allclose(result, expected, atol=1e-3)


@pytest.mark.parametrize("df1", [df_incomplete])
@pytest.mark.parametrize("df2", [df_imputed])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_wasserstein_distance(df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame) -> None:
    dist = metrics.wasserstein_distance(df1, df1, df_mask, method="columnwise")
    assert dist.equals(pd.Series([0.0, 0.0], index=["col1", "col2"]))
    dist = metrics.wasserstein_distance(df1, df2, df_mask, method="columnwise")
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
def test_kl_divergence(df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame) -> None:
    result = metrics.kl_divergence(df1, df1, df_mask, method="gaussian")
    expected = pd.Series([0, 0], index=["col1", "col2"])
    np.testing.assert_allclose(result, expected, atol=1e-3)

    result = metrics.kl_divergence(df1, df2, df_mask, method="gaussian")
    expected = pd.Series([0.669, 0.669], index=["col1", "col2"])
    np.testing.assert_allclose(result, expected, atol=1e-3)


@pytest.mark.parametrize("df1", [df_incomplete])
@pytest.mark.parametrize("df2", [df_imputed])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_frechet_distance(df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame) -> None:
    assert (
        metrics.frechet_distance(df1, df1, df_mask)
        .round(3)
        .equals(pd.Series([-0.0, -0.0], index=[0, 1]))
    )
    assert (
        metrics.frechet_distance(df1, df2, df_mask)
        .round(3)
        .equals(pd.Series([1.11, 1.11], index=[0, 1]))
    )
    assert (
        metrics.frechet_distance(df1, df1, df_mask, True)
        .round(3)
        .equals(pd.Series([-0.0], index=["All"]))
    )
    assert (
        metrics.frechet_distance(df1, df2, df_mask, True)
        .round(3)
        .equals(pd.Series([0.253], index=["All"]))
    )


@pytest.mark.parametrize("df1", [df_incomplete])
@pytest.mark.parametrize("df2", [df_imputed])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_kolmogorov_smirnov_test(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    assert metrics.kolmogorov_smirnov_test(df1, df1, df_mask).equals(
        pd.Series([0.0, 0.0], index=["col1", "col2"])
    )
    assert metrics.kolmogorov_smirnov_test(df1, df2, df_mask).equals(
        pd.Series([0.5, 2 / 3], index=["col1", "col2"])
    )


@pytest.mark.parametrize("df1", [df_incomplete])
@pytest.mark.parametrize("df2", [df_imputed])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_sum_pairwise_distances(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    assert metrics.sum_pairwise_distances(df1, df2, df_mask).equals(
        pd.Series([64.0], index=["All"])
    )


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
