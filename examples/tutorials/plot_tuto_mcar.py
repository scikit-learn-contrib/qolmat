"""
============================================
Tutorial for Testing the MCAR Case
============================================

In this tutorial, we show how to test the MCAR case using the Little's test.
"""

# %%
# First import some libraries
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from scipy.stats import norm

from qolmat.analysis.holes_characterization import LittleTest
from qolmat.benchmark.missing_patterns import UniformHoleGenerator

plt.rcParams.update({"font.size": 12})


# %%
# Generating random data
# ----------------------

rng = np.random.RandomState(42)
data = rng.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=200)
df = pd.DataFrame(data=data, columns=["Column 1", "Column 2"])

q975 = norm.ppf(0.975)

# %%
# The Little's test
# ---------------------------------------------------------------
# First, we need to introduce the concept of a missing pattern. A missing pattern, also called a
# pattern, is the structure of observed and missing values in a dataset. For example, in a
# dataset with two columns, the possible patterns are: (0, 0), (1, 0), (0, 1), (1, 1). The value 1
# (0) indicates that the column value is missing (observed).
#
# The null hypothesis, H0, is: "The means of observations within each pattern are similar.".
#
# We choose to use the classic threshold of 5%. If the test p-value is below this threshold,
# we reject the null hypothesis.
#
# This notebook shows how the Little's test performs on a simplistic case and its limitations. We
# instanciate a test object with a random state for reproducibility.

test_mcar = LittleTest(random_state=rng)

# %%
# Case 1: MCAR holes (True negative)
# ==================================


hole_gen = UniformHoleGenerator(
    n_splits=1, random_state=rng, subset=["Column 2"], ratio_masked=0.2
)
df_mask = hole_gen.generate_mask(df)
df_nan = df.where(~df_mask, np.nan)

has_nan = df_mask.any(axis=1)
df_observed = df.loc[~has_nan]
df_hidden = df.loc[has_nan]

plt.scatter(df_observed["Column 1"], df_observed[["Column 2"]], label="Fully observed values")
plt.scatter(df_hidden[["Column 1"]], df_hidden[["Column 2"]], label="Values with missing C2")

plt.legend(
    loc="lower left",
    fontsize=8,
)
plt.xlabel("Column 1")
plt.ylabel("Column 2")
plt.title("Case 1: MCAR missingness mechanism")
plt.grid()
plt.show()

# %%
result = test_mcar.test(df_nan)
print(f"Test p-value: {result:.2%}")
# %%
# The p-value is quite high, therefore we don't reject H0.
# We can then suppose that our missingness mechanism is MCAR.

# %%
# Case 2: MAR holes with mean bias (True positive)
# ================================================

df_mask = pd.DataFrame({"Column 1": False, "Column 2": df["Column 1"] > q975}, index=df.index)

df_nan = df.where(~df_mask, np.nan)

has_nan = df_mask.any(axis=1)
df_observed = df.loc[~has_nan]
df_hidden = df.loc[has_nan]

plt.scatter(df_observed["Column 1"], df_observed[["Column 2"]], label="Fully observed values")
plt.scatter(df_hidden[["Column 1"]], df_hidden[["Column 2"]], label="Values with missing C2")

plt.legend(
    loc="lower left",
    fontsize=8,
)
plt.xlabel("Column 1")
plt.ylabel("Column 2")
plt.title("Case 2: MAR missingness mechanism")
plt.grid()
plt.show()

# %%

result = test_mcar.test(df_nan)
print(f"Test p-value: {result:.2%}")
# %%
# The p-value is lower than the classic threshold (5%).
# H0 is then rejected and we can suppose that our missingness mechanism is MAR.

# %%
# Case 3: MAR holes with any mean bias (False negative)
# =====================================================
#
# The specific case is designed to emphasize the Little's test limits. In the case, we generate
# holes when the absolute value of the first feature is high. This missingness mechanism is clearly
# MAR but the means between missing patterns is not statistically different.

df_mask = pd.DataFrame(
    {"Column 1": False, "Column 2": df["Column 1"].abs() > q975}, index=df.index
)

df_nan = df.where(~df_mask, np.nan)

has_nan = df_mask.any(axis=1)
df_observed = df.loc[~has_nan]
df_hidden = df.loc[has_nan]

plt.scatter(df_observed["Column 1"], df_observed[["Column 2"]], label="Fully observed values")
plt.scatter(df_hidden[["Column 1"]], df_hidden[["Column 2"]], label="Values with missing C2")

plt.legend(
    loc="lower left",
    fontsize=8,
)
plt.xlabel("Column 1")
plt.ylabel("Column 2")
plt.title("Case 3: MAR missingness mechanism undetected by the Little's test")
plt.grid()
plt.show()

# %%

result = test_mcar.test(df_nan)
print(f"Test p-value: {result:.2%}")
# %%
# The p-value is higher than the classic threshold (5%).
# H0 is not rejected whereas the missingness mechanism is clearly MAR.

# %%
# Limitations
# -----------
# In this tutoriel, we can see that Little's test fails to detect covariance heterogeneity between
# patterns.
#
# We also note that the Little's test does not handle categorical data or temporally
# correlated data.
