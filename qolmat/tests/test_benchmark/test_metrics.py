import pandas as pd
import numpy as np
import scipy
import pytest

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
def test_kolmogorov_smirnov_test(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    assert metrics.kolmogorov_smirnov_test(df1, df1, df_mask).equals(
        pd.Series([0.0, 0.0], index=["col1", "col2"])
    )
    print(metrics.kolmogorov_smirnov_test(df1, df2, df_mask))
    assert metrics.kolmogorov_smirnov_test(df1, df2, df_mask).equals(
        pd.Series([0.5, 2 / 3], index=["col1", "col2"])
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
    assert metrics.mean_difference_correlation_matrix_numerical_features(df1, df2, df_mask).equals(
        pd.Series([0.0, 0.0], index=["col1", "col2"])
    )


df_incomplete_cat = pd.DataFrame(
    {"col1": ["a", np.nan, "a", "b", np.nan], "col2": ["c", np.nan, "c", np.nan, "d"]}
)

df_imputed_cat = pd.DataFrame(
    {"col1": ["a", "b", "a", "c", "c"], "col2": ["e", "d", "c", "d", "d"]}
)


@pytest.mark.parametrize("df1", [df_incomplete_cat])
@pytest.mark.parametrize("df2", [df_imputed_cat])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_total_variance_distance(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    assert metrics.total_variance_distance(df1, df1, df_mask).equals(
        pd.Series([0.0, 0.0], index=["col1", "col2"])
    )
    assert metrics.total_variance_distance(df1, df2, df_mask).equals(
        pd.Series([1.0, 1.0], index=["col1", "col2"])
    )


@pytest.mark.parametrize("df1", [df_incomplete_cat])
@pytest.mark.parametrize("df2", [df_imputed_cat])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_mean_difference_correlation_matrix_categorical_features(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    assert metrics.mean_difference_correlation_matrix_categorical_features(
        df1, df2, df_mask
    ).equals(pd.Series([0.0, 0.0], index=["col1", "col2"]))


df_incomplete_cat_num = pd.DataFrame(
    {"col1": ["a", np.nan, "a", "b", np.nan], "col2": [-1, np.nan, 0.5, 1, 1.5]}
)

df_imputed_cat_num = pd.DataFrame(
    {"col1": ["a", "b", "a", "c", "c"], "col2": [-1.5, 0, 1.5, 2, 1.5]}
)

df_mask = pd.DataFrame(
    {"col1": [True, False, True, True, False], "col2": [True, False, True, True, False]}
)


@pytest.mark.parametrize("df1", [df_incomplete_cat_num])
@pytest.mark.parametrize("df2", [df_imputed_cat_num])
@pytest.mark.parametrize("df_mask", [df_mask])
def test_mean_difference_correlation_matrix_categorical_vs_numerical_features(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> None:
    assert metrics.mean_difference_correlation_matrix_categorical_vs_numerical_features(
        df1, df2, df_mask
    ).equals(pd.Series([0.07009774198932273], index=["col1"]))