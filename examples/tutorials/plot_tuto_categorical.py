"""
==============================
Benchmark for categorical data
==============================

In this tutorial, we show how to use Qolmat to define imputation methods managing mixed type data.
We benchmark these methods on the Titanic Data Set.
It comprehends passengers features as well as if they survived the accident.
"""

from qolmat.imputations import preprocessing, imputers
from qolmat.imputations.imputers import ImputerRegressor
from qolmat.benchmark import missing_patterns
from qolmat.benchmark import comparator
from qolmat.utils import data

from sklearn.pipeline import Pipeline

# %%
# 1. Titanic dataset
# ---------------------------------------------------------------
# We get the data and focus on the explanatory variables
df = data.get_data("Titanic")
df = df.drop(columns=["survived"])

# %%
# 2. Mixed type imputation methods
# ---------------------------------------------------------------
# Qolmat supports three approaches to impute mixed type data.
# The first approach is a simple imputation by the mean, median or the most-frequent value column
# by column

imputer_simple = imputers.ImputerSimple()

# %%
# The second approach relies on the class WrapperTransformer which wraps a numerical imputation
# method (e.g. RPCA) in a preprocessing transformer with fit_transform and inverse_transform
# methods providing an embedding of the data.

cols_num = df.select_dtypes(include="number").columns
cols_cat = df.select_dtypes(exclude="number").columns
imputer_rpca = imputers.ImputerRpcaNoisy()
ohe = preprocessing.OneHotEncoderProjector(
    handle_unknown="ignore",
    handle_missing="return_nan",
    use_cat_names=True,
    cols=cols_cat,
)
bt = preprocessing.BinTransformer(cols=cols_num)
wrapper = Pipeline(steps=[("OneHotEncoder", ohe), ("BinTransformer", bt)])
imputer_wrap_rpca = preprocessing.WrapperTransformer(imputer_rpca, wrapper)

# %%
# The third approach uses ImputerRegressor which imputes iteratively each column using the other
# ones. The function make_robust_MixteHGB provides an underlying model able to:
# - adress both numerical targets (regression) and categorical targets (classification)
# - manage categorical features though one hot encoding
# - manage missing features (native to the HistGradientBoosting)

pipestimator = preprocessing.make_robust_MixteHGB(avoid_new=True)
imputer_hgb = ImputerRegressor(estimator=pipestimator, handler_nan="none")
imputer_wrap_hgb = preprocessing.WrapperTransformer(imputer_hgb, bt)

#  %%
# 3. Mixed type model selection
# ---------------------------------------------------------------
# Let us now compare these three aproaches by measuring their ability to impute uniformly
# distributed holes.

dict_imputers = {
    "Simple": imputer_simple,
    "HGB": imputer_wrap_hgb,
    "RPCA": imputer_wrap_rpca,
}
cols_to_impute = df.columns
ratio_masked = 0.1
generator_holes = missing_patterns.UniformHoleGenerator(
    n_splits=2,
    subset=cols_to_impute,
    ratio_masked=ratio_masked,
    sample_proportional=False,
)
metrics = ["rmse", "accuracy"]

comparison = comparator.Comparator(
    dict_imputers,
    cols_to_impute,
    generator_holes=generator_holes,
    metrics=metrics,
    max_evals=2,
)
results = comparison.compare(df)

# %%
# On numerical variables, the imputation based on the HistGradientBoosting (HGB) model globally
# achieves lower Root-square Mean Squared Errors (RMSE).
results.loc["rmse"].style.highlight_min(color="lightgreen", axis=1)

# %%
# The HGB imputation methods globaly reaches a better accuracy on the categorical data.
results.loc["accuracy"].style.highlight_max(color="lightgreen", axis=1)
