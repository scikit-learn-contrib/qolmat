"""
============================================
Tutorial for testing the MCAR case
============================================

In this tutorial, we show how to use the mcar test classe and it methods

Keep in my mind that, at this moment, the mcar tests are only handle tabular data.
"""
# %%
# First import some libraries
from matplotlib import pyplot as plt
import random

import numpy as np
import pandas as pd

from qolmat.audit.holes_characterization import MCARTest

# %%
# 1. The Little's test
# ---------------------------------------------------------------
# How to use the Little's test ?
# ==============================
# When we deal with missing data in our dataset it's interesting to know the nature of these holes.
# There exist three types of holes : MCAR, MAR and MNAR.
# (see the: `Rubin's missing mechanism classification
# <https://qolmat.readthedocs.io/en/latest/explanation.html>`_)
#
# The simplest case to test is the MCAR case. The most famous MCAR statistical test is the
# `Little's test <https://www.tandfonline.com/doi/abs/10.1080/01621459.1988.10478722>`_.
# Keep in mind that the Little's test is designed to test the homogeneity of means between the
# missing patterns and won't be efficient to detect the heterogeneity of covariance between missing
# patterns.
#
# This notebook shows how the Little's test performs and its limitations.

np.random.seed(11)

mcartest = MCARTest(method="little")

# %%
# Case 1 : Normal iid feature with MCAR holes
# ===========================================

matrix = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=100)
matrix.ravel()[np.random.choice(matrix.size, size=20, replace=False)] = np.nan
matrix_masked = matrix[np.argwhere(np.isnan(matrix))]
df_1 = pd.DataFrame(matrix)

plt_1 = plt.scatter(matrix[:, 0], matrix[:, 1])
plt_2 = plt.scatter(matrix_masked[:, 0], matrix_masked[:, 1])

plt.legend(
    (plt_1, plt_2),
    ("observed_values", "masked_vlues"),
    scatterpoints=1,
    loc="lower left",
    ncol=1,
    fontsize=8,
)

plt.title("Case 1 : MCAR missingness mechanism")
plt.xlabel("x values (all observed)")
plt.ylabel("y values (with missing ones)")

plt.show()

# %%

mcartest.test(df_1)
# %%
# The p-value is quite high, therefore we don't reject H_0.
# We can then suppose that our missingness mechanism is MCAR.

# %%
# Case 2 : Normal iid feature with MAR holes
# ==========================================
np.random.seed(11)

matrix = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=100)
threshold = random.uniform(0, 1)
matrix[np.argwhere(matrix[:, 0] > 1.96), 1] = np.nan
matrix_masked = matrix[np.argwhere(np.isnan(matrix))]
df_2 = pd.DataFrame(matrix)

plt_1 = plt.scatter(matrix[:, 0], matrix[:, 1])
plt_2 = plt.scatter(matrix_masked[:, 0], matrix_masked[:, 1])

plt.legend(
    (plt_1, plt_2),
    ("observed_values", "masked_vlues"),
    scatterpoints=1,
    loc="lower left",
    ncol=1,
    fontsize=8,
)

plt.title("Case 2 : MAR missingness mechanism")
plt.xlabel("x values (all observed)")
plt.ylabel("y values (with missing ones)")

plt.show()

# %%

mcartest.test(df_2)
# %%
# The p-value is lower than the classic threshold (5%).
# H_0 is then rejected and we can suppose that our missingness mechanism is MAR.

# %%
# Case 3 : Normal iid feature MAR holes
# =====================================
# The specific case is design to emphasize the Little's test limits. In the case, we generate holes
# when the value of the first feature is high. This missingness mechanism is clearly MAR but the
# means between missing patterns is not statistically different.

np.random.seed(11)

matrix = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=100)
matrix[np.argwhere(abs(matrix[:, 0]) >= 1.95), 1] = np.nan
matrix_masked = matrix[np.argwhere(np.isnan(matrix))]
df_3 = pd.DataFrame(matrix)

plt_1 = plt.scatter(matrix[:, 0], matrix[:, 1])
plt_2 = plt.scatter(matrix_masked[:, 0], matrix_masked[:, 1])

plt.legend(
    (plt_1, plt_2),
    ("observed_values", "masked_values"),
    scatterpoints=1,
    loc="lower left",
    ncol=1,
    fontsize=8,
)

plt.title("Case 3 : MAR missingness mechanism undetected by the Little's test")
plt.xlabel("x values (all observed)")
plt.ylabel("y values (with missing ones)")

plt.show()

# %%

mcartest.test(df_3)
# %%
# The p-value is higher than the classic threshold (5%).
# H_0 is not rejected whereas the missingness mechanism is clearly MAR.
