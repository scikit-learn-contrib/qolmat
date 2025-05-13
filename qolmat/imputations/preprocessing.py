"""Script for preprocessing functions."""

import copy
from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np
import pandas as pd
from category_encoders.one_hot import OneHotEncoder
from numpy.typing import NDArray
from sklearn import utils as sku
from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import InputTags
from sklearn.utils.validation import (
    check_is_fitted,
)

# from typing_extensions import Self
from qolmat.utils import utils


class MixteHGBM(RegressorMixin, BaseEstimator):
    """MixteHGBM class.

    This is a custom scikit-learn estimator implementing a mixed model using
    HistGradientBoostingClassifier for string target data and
    HistGradientBoostingRegressor for numeric target data.
    """

    def __init__(self):
        super().__init__()

    def set_model_parameters(self, **args_model):
        """Set the arguments of the underlying model.

        Parameters
        ----------
        **args_model : dict
            Additional keyword arguments to be passed to the underlying models.

        """
        self.args_model = args_model

    def fit(self, X: NDArray, y: NDArray) -> "MixteHGBM":
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors.
        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.

        """
        X, y = sku.validation.validate_data(
            self,
            X,
            y,
            accept_sparse=False,
            ensure_all_finite="allow-nan",
            reset=True,
            dtype=["float", "int", "string", "categorical", "object"],
        )
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]
        if hasattr(self, "args_model"):
            args_model = self.args_model
        else:
            args_model = {}
        if pd.api.types.is_string_dtype(y):
            model = HistGradientBoostingClassifier(**args_model)
        elif pd.api.types.is_numeric_dtype(y):
            model = HistGradientBoostingRegressor(**args_model)
        else:
            raise TypeError("Unknown label type")

        self.model_ = model.fit(X, y)
        return self

    def predict(self, X: NDArray) -> NDArray:
        """Predict using the fitted model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : array-like, shape (n_samples,)
            Predicted target values.

        """
        sku.validation.validate_data(
            self,
            X,
            accept_sparse=False,
            ensure_all_finite="allow-nan",
            reset=False,
            dtype=["float", "int", "string", "categorical", "object"],
        )
        check_is_fitted(self, "is_fitted_")
        y_pred = self.model_.predict(X)
        return y_pred

    def __sklearn_tags__(self):
        """Indicate if the class allows inputs with categorical data and nans.

        It modifies the behaviour of the functions checking data.
        """
        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags(
            two_d_array=True, categorical=True, string=True, allow_nan=True
        )
        tags.target_tags.single_output = False
        tags.non_deterministic = True
        return tags


class BinTransformer(TransformerMixin, BaseEstimator):
    """BinTransformer class.

    Learn the possible values of the provided numerical feature,
    allowing to transform new values to the closest existing one.
    """

    def __init__(self, cols: Optional[List] = None):
        super().__init__()
        self.cols = cols

    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> "BinTransformer":
        """Fit the BinTransformer to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the unique values.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        self : object
            Fitted transformer.

        """
        sku.validation.validate_data(
            self,
            X,
            accept_sparse=False,
            ensure_all_finite="allow-nan",
            reset=True,
            dtype=["float", "int", "string", "categorical", "object"],
        )
        df = utils._validate_input(X)
        self.feature_names_in_ = df.columns
        self.n_features_in_ = len(df.columns)
        self.dict_df_bins_: Dict[Hashable, pd.DataFrame] = {}
        if self.cols is None:
            cols = df.select_dtypes(include="number").columns
        else:
            cols = self.cols
        for col in cols:
            values = df[col]
            values = values.dropna()
            df_bins = pd.DataFrame({"value": np.sort(values.unique())})
            df_bins["min"] = (df_bins["value"] + df_bins["value"].shift()) / 2
            self.dict_df_bins_[col] = df_bins.fillna(-np.inf)
        return self

    def transform(self, X: NDArray) -> NDArray:
        """Transform X to existing values learned during fit.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            The data to transform.

        Returns
        -------
        X_out : ndarray of shape (n_samples,)
            Transformed input.

        """
        sku.validation.validate_data(
            self,
            X,
            accept_sparse=False,
            ensure_all_finite="allow-nan",
            reset=False,
            dtype=["float", "int", "string", "categorical", "object"],
        )
        df = utils._validate_input(X)
        check_is_fitted(self)
        # if (
        #     not hasattr(self, "feature_names_in_")
        #     or df.columns.to_list() != self.feature_names_in_.to_list()
        # ):
        #     raise ValueError(
        #         f"Feature names in X {df.columns} don't match with "
        #         f"expected {self.feature_names_in_}"
        #     )
        df_out = df.copy()
        for col in df:
            values = df[col]
            if col in self.dict_df_bins_.keys():
                df_bins = self.dict_df_bins_[col]
                bins_X = np.digitize(values, df_bins["min"]) - 1
                values_out = df_bins.loc[bins_X, "value"]
                values_out.index = values.index
                df_out[col] = values_out.where(values.notna(), np.nan)
        if isinstance(X, np.ndarray):
            return df_out.values
        return df_out

    def inverse_transform(self, X: NDArray) -> NDArray:
        """Transform X to existing values learned during fit.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        X_out : ndarray of shape (n_samples,)
            Transformed input.

        """
        return self.transform(X)

    def __sklearn_tags__(self):
        """Indicate if the class allows inputs with categorical data and nans.

        It modifies the behaviour of the functions checking data.
        """
        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags(
            two_d_array=True, categorical=True, string=True, allow_nan=True
        )
        tags.target_tags.single_output = False
        tags.non_deterministic = True
        return tags


class OneHotEncoderProjector(OneHotEncoder):
    """Class for one-hot encoding of categorical features.

    It inherits from the class OneHotEncoder imported from category_encoders.
    The decoding function accepts non boolean values (as it is the case for
    the sklearn OneHotEncoder). In this case the decoded value corresponds to
    the largest dummy value.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reverse_dummies(self, X: pd.DataFrame, mapping: Dict) -> pd.DataFrame:
        """Convert dummy variable into numerical variables.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        mapping: list-like
              Mapping of column to be transformed to its
              new columns and value represented

        Returns
        -------
        numerical: DataFrame

        """
        out_cols = X.columns.tolist()
        mapped_columns = []
        for switch in mapping:
            col = switch.get("col")
            mod = switch.get("mapping")
            insert_at = out_cols.index(mod.columns[0])
            X.insert(insert_at, col, 0)
            positive_indexes = mod.index[mod.index > 0]
            max_code = X[mod.columns].max(axis=1)
            for existing_col, val in zip(mod.columns, positive_indexes):
                X.loc[X[existing_col] == max_code, col] = val
                mapped_columns.append(existing_col)
            X = X.drop(mod.columns, axis=1)
            out_cols = X.columns.tolist()

        return X


class WrapperTransformer(TransformerMixin, BaseEstimator):
    """Wrap a transformer.

    Wrapper with reversible transformers designed to embed the data.
    """

    def __init__(
        self, transformer: TransformerMixin, wrapper: TransformerMixin
    ):
        super().__init__()
        self.transformer = transformer
        self.wrapper = wrapper

    def fit(
        self, X: NDArray, y: Optional[NDArray] = None
    ) -> "WrapperTransformer":
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : NDArray
            Input array.
        y : Optional[NDArray], optional
            _description_, by default None

        Returns
        -------
        Self
            The object itself.

        """
        X_transformed = copy.deepcopy(X)
        X_transformed = self.wrapper.fit_transform(X_transformed)
        X_transformed = self.transformer.fit(X_transformed)
        return self

    def fit_transform(self, X: NDArray) -> NDArray:
        """Fit the model according to the given training data and transform it.

        Parameters
        ----------
        X : NDArray
            Input array.

        Returns
        -------
        NDArray
            Transformed array.

        """
        X_transformed = copy.deepcopy(X)
        X_transformed = self.wrapper.fit_transform(X_transformed)
        X_transformed = self.transformer.fit_transform(X_transformed)
        X_transformed = self.wrapper.inverse_transform(X_transformed)
        return X_transformed

    def transform(self, X: NDArray) -> NDArray:
        """Transform X.

        Parameters
        ----------
        X : NDArray
            Input array.

        Returns
        -------
        NDArray
            Transformed array.

        """
        X_transformed = copy.deepcopy(X)
        X_transformed = self.wrapper.transform(X_transformed)
        X_transformed = self.transformer.transform(X_transformed)
        X_transformed = self.wrapper.inverse_transform(X_transformed)
        return X_transformed


def make_pipeline_mixte_preprocessing(
    scale_numerical: bool = False, avoid_new: bool = False
) -> Pipeline:
    """Create a preprocessing pipeline managing mixed type data.

    It does this by one hot encoding categorical data.

    Parameters
    ----------
    scale_numerical : bool, default=False
        Whether to scale numerical features.
    avoid_new : bool, default=False
        Whether to forbid new numerical values.

    Returns
    -------
    preprocessor : Pipeline
        Preprocessing pipeline

    """
    transformers: List[Tuple] = []
    if scale_numerical:
        transformers += [
            ("num", StandardScaler(), selector(dtype_include=np.number))
        ]

    ohe = OneHotEncoder(handle_unknown="ignore", use_cat_names=True)
    transformers += [("cat", ohe, selector(dtype_exclude=np.number))]
    col_transformer = ColumnTransformer(
        transformers=transformers, remainder="passthrough"
    )
    col_transformer = col_transformer.set_output(transform="pandas")
    preprocessor = Pipeline(steps=[("col_transformer", col_transformer)])

    if avoid_new:
        preprocessor.steps.append(("bins", BinTransformer()))
    return preprocessor


def make_robust_MixteHGB(
    scale_numerical: bool = False, avoid_new: bool = False
) -> Pipeline:
    """Create a robust pipeline for MixteHGBM.

    Create a preprocessing pipeline managing mixed type data
    by one hot encoding categorical features.
    This estimator is intended for use in ImputerRegressor
    to deal with mixed type data.

    Note that from sklearn 1.4 HistGradientBoosting Natively Supports
    Categorical DTypes in DataFrames, so that this pipeline is not
    required anymore.


    Parameters
    ----------
    scale_numerical : bool, default=False
        Whether to scale numerical features.
    avoid_new : bool, default=False
        Whether to forbid new numerical values.

    Returns
    -------
    robust_MixteHGB : object
        A robust pipeline for MixteHGBM.

    """
    preprocessor = make_pipeline_mixte_preprocessing(
        scale_numerical=scale_numerical, avoid_new=avoid_new
    )
    robust_MixteHGB = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("estimator", MixteHGBM()),
        ]
    )

    return robust_MixteHGB
