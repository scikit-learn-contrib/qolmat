import numpy as np
import pandas as pd
import pytest
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from qolmat.imputations.estimators import (
    BinTransformer,
    MixteHGBM,
    make_pipeline_mixte_preprocessing,
    make_robust_MixteHGB,
)

# Sample data for testing
X_cat = np.random.choice(["A", "B", "C"], size=(100, 3))
values = np.random.rand(100, 3)
X_num = np.random.rand(100, 3)
X = np.concatenate([X_num, X_cat], axis=1)
df_X = pd.DataFrame(X)
y_numeric = np.random.rand(100)
y_string = np.random.choice(["A", "B", "C"], size=100)


@pytest.fixture
def mixte_hgb_model():
    return MixteHGBM()


@pytest.fixture
def robust_mixte_hgb_model():
    return make_robust_MixteHGB()


def test_estimator(mixte_hgb_model):
    check_estimator(mixte_hgb_model)


def test_fit_predict(mixte_hgb_model):
    # Test fitting and predicting with numeric target
    X_train, X_test, y_train, y_test = train_test_split(
        X_num, y_numeric, test_size=0.2, random_state=42
    )
    mixte_hgb_model.fit(X_train, y_train)
    y_pred = mixte_hgb_model.predict(X_test)
    assert mean_squared_error(y_test, y_pred) >= 0

    # Test fitting and predicting with string target
    X_train, X_test, y_train, y_test = train_test_split(
        X_num, y_string, test_size=0.2, random_state=42
    )
    mixte_hgb_model.fit(X_train, y_train)
    y_pred = mixte_hgb_model.predict(X_test)
    assert len(y_pred) == len(X_test)


# Testing BinTransformer


@pytest.fixture
def bin_transformer():
    return BinTransformer()


def test_fit_transform(bin_transformer):
    X = np.array([1, 2, 3, np.nan, 5])
    transformed_X = bin_transformer.fit_transform(X)
    assert np.array_equal(transformed_X, np.array([1, 2, 3, np.nan, 5]), equal_nan=True)


def test_transform(bin_transformer):
    bin_transformer.dict_df_bins_ = {
        0: pd.DataFrame({"value": [1, 2, 3, 4, 5], "min": [-np.inf, 1.5, 2.5, 3.5, 4.5]})
    }
    X = np.array([4.2, -1, 3.0, 4.5, 12])
    transformed_X = bin_transformer.transform(X)
    assert np.array_equal(transformed_X, np.array([4, 1, 3, 5, 5]))


def test_fit_transform_with_series(bin_transformer):
    X = pd.Series([1, 2, 3, np.nan, 5])
    transformed_X = bin_transformer.fit_transform(X)
    pd.testing.assert_series_equal(transformed_X, pd.Series([1, 2, 3, np.nan, 5]))


def test_transform_with_series(bin_transformer):
    bin_transformer.dict_df_bins_ = {
        0: pd.DataFrame({"value": [1, 2, 3, 4, 5], "min": [0.5, 1.5, 2.5, 3.5, 4.5]})
    }
    X = pd.Series([1, 2, 3, 4, 5])
    transformed_X = bin_transformer.transform(X)
    pd.testing.assert_series_equal(transformed_X, pd.Series([1, 2, 3, 4, 5], dtype=float))


# Testing make_pipeline_mixte_preprocessing


@pytest.fixture
def preprocessing_pipeline():
    return make_pipeline_mixte_preprocessing()


def test_preprocessing_pipeline(preprocessing_pipeline):
    # Ensure the pipeline is constructed correctly
    assert isinstance(preprocessing_pipeline, BaseEstimator)

    # Test with numerical features
    X_num = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
    transformed_X = preprocessing_pipeline.fit_transform(X_num)
    assert isinstance(transformed_X, pd.DataFrame)
    assert transformed_X.shape[1] == X_num.shape[1]

    # Test with categorical features
    X_cat = pd.DataFrame([["a", "b"], ["c", "d"], ["e", "f"]])
    transformed_X = preprocessing_pipeline.fit_transform(X_cat)
    assert isinstance(transformed_X, pd.DataFrame)
    assert transformed_X.shape[1] > X_cat.shape[1]

    # Test with mixed features
    X_mixed = pd.DataFrame([[1, "a"], [2, "b"], [3, "c"]])
    transformed_X = preprocessing_pipeline.fit_transform(X_mixed)
    assert isinstance(transformed_X, pd.DataFrame)
    assert transformed_X.shape[1] > X_mixed.shape[1]


# Testing make_robust_MixteHGB


def test_make_robust_MixteHGB(robust_mixte_hgb_model):
    # Ensure the pipeline is constructed correctly
    assert isinstance(robust_mixte_hgb_model, Pipeline)

    # Ensure the preprocessor in the pipeline is of type ColumnTransformer
    assert isinstance(robust_mixte_hgb_model.named_steps["preprocessor"], ColumnTransformer)

    # Test fitting and predicting with numeric target
    X_train, X_test, y_train, y_test = train_test_split(
        df_X, y_numeric, test_size=0.2, random_state=42
    )
    robust_mixte_hgb_model.fit(X_train, y_train)
    y_pred = robust_mixte_hgb_model.predict(X_test)
    assert mean_squared_error(y_test, y_pred) >= 0

    # Test fitting and predicting with string target
    X_train, X_test, y_train, y_test = train_test_split(
        df_X, y_string, test_size=0.2, random_state=42
    )
    robust_mixte_hgb_model.fit(X_train, y_train)
    y_pred = robust_mixte_hgb_model.predict(X_test)
    assert len(y_pred) == len(X_test)


if __name__ == "__main__":
    pytest.main([__file__])