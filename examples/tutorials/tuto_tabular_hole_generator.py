"""
============================================
Tutorial for hole generation in tabular data
============================================

In this tutorial, we show how to use the different hole generator classes
in a tabular data case. In particular, we show how to use the
:class:`~qolmat.benchmark.missing_patterns.UniformHoleGenerator`,
:class:`~qolmat.benchmark.missing_patterns.GeometricHoleGenerator`,
:class:`~qolmat.benchmark.missing_patterns.EmpiricalHoleGenerator`
and :class:`~qolmat.benchmark.missing_patterns.MultiMarkovHoleGenerator`
classes.
The dataset used is the the numerical `superconduct` dataset and
contains information on 21263 superconductors.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qolmat.benchmark import missing_patterns
from qolmat.utils import data, plot

# %%
# 1. Data
# ---------------------------------------------------------------
# The data contains information on 21263 superconductors.
# Originally, the first 81 columns contain extracted features and
# the 82nd column contains the critical temperature which is used as the
# target variable. The original data from which the features were extracted
# comes from http://supercon.nims.go.jp/index_en.html, which is public.
# The data does not contain missing values; so for the purpose of this notebook,
# we corrupt the data, with the ``qolmat.utils.data.add_holes`` function.
# In this way, each column has missing values.

csv_url = (
    "https://huggingface.co/datasets/polinaeterna/"
    "tabular-benchmark/resolve/main/reg_num/superconduct.csv"
)
df_data = pd.read_csv(csv_url, index_col=0)
df = data.add_holes(df_data, ratio_masked=0.2, mean_size=120)

# %%
# The dataset contains 82 columns. For simplicity,
# we only consider some.

columns = [
    "criticaltemp",
    "mean_atomic_mass",
    "mean_FusionHeat",
    "mean_ThermalConductivity",
    "mean_Valence",
]
df = df[columns]
cols_to_impute = df.columns

# %%
# Let's visualise the mask (i.e. missing values) of this dataset.
# Missing values are in white, while observed ones ae in black.

plt.figure(figsize=(15, 4))
plt.imshow(df.notna().values.T, aspect="auto", cmap="binary", interpolation="none")
plt.yticks(range(len(df.columns)), df.columns)
plt.xlabel("Samples", fontsize=12)
plt.grid(False)
plt.show()

# %%
# 2. Hole generators
# ---------------------------------------------------------------
# Given an pandas dataframe `df`, the aim of a hole generator
# is to create a mask `mask`, i.e. a pandas dataframe, of the same dimension
# as `df` with missing entries `np.nan`. The missing entries of the mask
# cannot be missing in the initial dataframe.
# For each method, we will generate 10 percent missing values, i.e.
# ``ratio_masked=0.1``, and we will generate missing values
# for all the columns in the dataframe, i.e. ``subset=df.columns``.
# Since the exercise here is simply to show how to generate missing data,
# the ``n_splits`` argument is not important (it indicates the number of times
# a mask must be generated when using the comparator).
# We therefore set it to 1.
# Let's just define a fucntion to visualise the additional
# missing values.


def visualise_missing_values(df_init, df_mask):
    """Visualise the missing values in the final dataframe
    with different colors for initial (white) and
    additional (red) missing values.
    """
    df_tot = df_init.copy()
    df_tot[df_init.notna()] = 0
    df_tot[df_init.isna()] = 2
    df_mask = np.invert(df_mask).astype("int")
    df_tot += df_mask
    colorsList = [(1, 0, 0), (0, 0, 0), (1, 1, 1)]
    custom_cmap = matplotlib.colors.ListedColormap(colorsList)
    plt.figure(figsize=(15, 4))
    plt.imshow(df_tot.values.T, aspect="auto", cmap=custom_cmap, interpolation="none")
    plt.yticks(range(len(df_tot.columns)), df_tot.columns)
    plt.xlabel("Samples", fontsize=12)
    plt.grid(False)
    plt.show()


# %%
# a. Uniform Hole Generator
# ***************************************************************
# The holes are generated randomly, using the ``resample`` method of scikit learn.
# Holels are created column by column. This metohd is implemented in the
# :class:`~qolmat.benchmark.missing_patterns.UniformHoleGenerator` class.

generator = missing_patterns.UniformHoleGenerator(n_splits=1, subset=df.columns, ratio_masked=0.1)
df_mask = generator.generate_mask(df)

print("Pourcentage of additional missing values:")
print(round((df_mask.sum() / len(df_mask)) * 100, 2))
visualise_missing_values(df, df_mask)

# %%
# Just to illustrate, imagine we have columns without missing values.
# In this case, there is no point to create hole in these columns.
# So all we need to do is pass in the argument the name of the columns
# for which we want to create gaps, for example,

unif_generator = missing_patterns.UniformHoleGenerator(
    n_splits=1,
    subset=["mean_FusionHeat", "mean_ThermalConductivity", "mean_Valence"],
    ratio_masked=0.1,
)
df_mask = unif_generator.generate_mask(df)

print("Pourcentage of additional missing values:")
print(round((df_mask.sum() / len(df_mask)) * 100, 2))
visualise_missing_values(df, df_mask)

# %%
# b. Geometric Hole Generator
# ***************************************************************
# The holes are generated following a Markov 1D process.
# Holes are created column by column. The transition matrix of the
# one-dimensional Markov process is learned from the data.
# It is therefore necessary to fit the data. Once the transition
# matrices have been calculated, we can generate the masks for each
# column to obtain the final mask for the entire dataframe.
# This metohd is implemented in the
# :class:`~qolmat.benchmark.missing_patterns.UniformHoleGenerator` class.


geom_generator = missing_patterns.GeometricHoleGenerator(
    n_splits=1, subset=cols_to_impute, ratio_masked=0.1
)
geom_generator.fit(df)
df_mask = geom_generator.generate_mask(df)

print("Pourcentage of additional missing values:")
print(round((df_mask.sum() / len(df_mask)) * 100, 2))
visualise_missing_values(df, df_mask)

# %%
# Note it is also possible to use this class even if we do not have
# a dataframe to fit. It suffices to define a ``mean_size`` parameter and
# the two attributes: ``dict_probas_out`` and ``dict_ratios``.
# In this case, there is no ``fit`` to do and we can
# directly call the ``generate_mask`` method.

geom_generator = missing_patterns.GeometricHoleGenerator(
    n_splits=1, subset=df.columns, ratio_masked=0.1
)
mean_size = 4
geom_generator.dict_probas_out = {column: 1 / mean_size for column in df.columns}
geom_generator.dict_ratios = {column: 1 / len(df.columns) for column in df.columns}
df_mask = geom_generator.generate_mask(df)

print("Pourcentage of additional missing values:")
print(round((df_mask.sum() / len(df_mask)) * 100, 2))
visualise_missing_values(df, df_mask)

# %%
# c. Empirical Hole Generator
# ***************************************************************
# The distribution of holes is learned from the data.
# The distributions of holes are learned column by column; so you need to fit
# the generator to the data.
# This metohd is implemented in the
# :class:`~qolmat.benchmark.missing_patterns.EmpiricalHoleGenerator` class.

empirical_generator_holes = missing_patterns.EmpiricalHoleGenerator(
    n_splits=1, subset=df.columns, ratio_masked=0.1
)
empirical_generator_holes.fit(df)
df_mask = empirical_generator_holes.generate_mask(df)

print("Pourcentage of additional missing values:")
print(round((df_mask.sum() / len(df_mask)) * 100, 2))
visualise_missing_values(df, df_mask)

# %%
# d. Multi Markov Hole Generator
# ***************************************************************
# The holes are generated according to a Markov process.
# Each line of the dataframe mask (np.nan) represents a state of the Markov chain.
# Note it is also more difficult to achieve exactly the required
# missing data ratio.
# This metohd is implemented in the
# :class:`~qolmat.benchmark.missing_patterns.MultiMarkovHoleGenerator` class.

multimarkov_generator = missing_patterns.MultiMarkovHoleGenerator(
    n_splits=1, subset=df.columns, ratio_masked=0.1
)
multimarkov_generator.fit(df)
df_mask = multimarkov_generator.generate_mask(df)

print("Pourcentage of additional missing values:")
print(round((df_mask.sum() / len(df_mask)) * 100, 2))
visualise_missing_values(df, df_mask)
