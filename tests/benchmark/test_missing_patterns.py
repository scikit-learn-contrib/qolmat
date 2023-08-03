import numpy as np
import pandas as pd
import pytest

from qolmat.benchmark import missing_patterns as mp

df_complet = pd.DataFrame({"col1": [i for i in range(100)], "col2": [2 * i for i in range(100)]})

df_incomplet = df_complet.copy()
df_incomplet.iloc[99, :] = np.nan

index_group = np.concatenate([[f"g{i}" for i in range(10)] for j in range(10)])

df_complet_group = df_complet.copy()
df_complet_group = df_complet_group.set_index(index_group)
df_complet_group.index = df_complet_group.index.set_names("group")

df_incomplet_group = df_incomplet.copy()
df_incomplet_group = df_incomplet_group.set_index(pd.Index(index_group))
df_incomplet_group.index = df_incomplet_group.index.set_names("group")

list_generators = {
    "geo": mp.GeometricHoleGenerator(n_splits=2, ratio_masked=0.1, random_state=42),
    "unif": mp.UniformHoleGenerator(n_splits=2, ratio_masked=0.1, random_state=42),
    "multi": mp.MultiMarkovHoleGenerator(n_splits=2, ratio_masked=0.1, random_state=42),
    "group": mp.GroupedHoleGenerator(
        n_splits=2, ratio_masked=0.1, random_state=42, groups=("group",)
    ),
}


@pytest.mark.parametrize(
    "df, generator",
    [
        (df_incomplet, list_generators["geo"]),
        (df_incomplet, list_generators["unif"]),
        (df_incomplet, list_generators["multi"]),
        (df_incomplet_group, list_generators["group"]),
    ],
)
def test_SamplerHoleGenerator_split(df: pd.DataFrame, generator: mp._HoleGenerator) -> None:
    mask = generator.split(df)[0]
    col1_holes = mask["col1"].sum()
    col2_holes = mask["col2"].sum()
    expected_col1_holes = 10
    expected_col2_holes = 10
    np.testing.assert_allclose(col1_holes, expected_col1_holes, atol=1)
    np.testing.assert_allclose(col2_holes, expected_col2_holes, atol=1)


@pytest.mark.parametrize(
    "df, generator",
    [
        (df_complet, list_generators["geo"]),
        (df_complet, list_generators["unif"]),
        (df_complet, list_generators["multi"]),
        (df_complet_group, list_generators["group"]),
    ],
)
def test_SamplerHoleGenerator_without_real_nans(
    df: pd.DataFrame, generator: mp._HoleGenerator
) -> None:
    real_nan = np.random.choice([True, False], size=df.size, p=[0.4, 0.6]).reshape(100, 2)
    df[real_nan] = np.nan

    mask = generator.split(df)[0]

    loc_real_nans_col1 = np.where(df["col1"].isna())[0]
    loc_mask_col1 = np.where(mask["col1"])[0]

    loc_real_nans_col2 = np.where(df["col2"].isna())[0]
    loc_mask_col2 = np.where(mask["col2"])[0]

    np.testing.assert_allclose(len(set(loc_real_nans_col1) & set(loc_mask_col1)), 0)
    np.testing.assert_allclose(len(set(loc_real_nans_col2) & set(loc_mask_col2)), 0)
