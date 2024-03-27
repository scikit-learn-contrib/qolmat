import copy
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)

from category_encoders.one_hot import OneHotEncoder


from typing_extensions import Self
from numpy.typing import NDArray


class MixteHGBM(RegressorMixin, BaseEstimator):
    """
    A custom scikit-learn estimator implementing a mixed model using
    HistGradientBoostingClassifier for string target data and
    HistGradientBoostingRegressor for numeric target data.

    Parameters:
    -----------
    allow_new : bool, default=True
        Whether to allow new categories in numerical target data. If false the predictions are
        mapped to the closest existing value.
    """

    def __init__(self, allow_new=True):
        super().__init__()
        self.allow_new = allow_new

    def set_model_parameters(self, **args_model):
        """
        Sets the arguments of the underlying model.

        Parameters:
        -----------
        **kwargs : dict
            Additional keyword arguments to be passed to the underlying models.
        """
        self.args_model = args_model

    def fit(self, X: NDArray, y: NDArray) -> Self:
        """
        Fit the model according to the given training data.

        Parameters:
        -----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors.
        y : array-like, shape (n_samples,)
            Target values.

        Returns:
        --------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True, force_all_finite="allow-nan")
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
        """
        Predict using the fitted model.

        Parameters:
        -----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Samples.

        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Predicted target values.
        """
        X = check_array(X, accept_sparse=True, force_all_finite="allow-nan")
        check_is_fitted(self, "is_fitted_")
        y_pred = self.model_.predict(X)
        return y_pred

    def _more_tags(self):
        """
        This method indicates that this class allows inputs with categorical data and nans. It
        modifies the behaviour of the functions checking data.
        """
        return {"X_types": ["2darray", "categorical", "string"], "allow_nan": True}


class BinTransformer(TransformerMixin, BaseEstimator):
    """
    Learns the possible values of the provided numerical feature, allowing to transform new values
    to the closest existing one.
    """

    def __init__(self, cols: Optional[List] = None):
        super().__init__()
        self.cols = cols

    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> Self:
        """
        Fit the BinTransformer to X.

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
        X = check_array(X, accept_sparse=False, force_all_finite="allow-nan", ensure_2d=False)
        df = pd.DataFrame(X)
        self.dict_df_bins_ = dict()
        cols = df.columns if self.cols is None else self.cols
        for col in cols:
            values = df[col]
            values = values.dropna()
            df_bins = pd.DataFrame({"value": np.sort(values.unique())})
            df_bins["min"] = (df_bins["value"] + df_bins["value"].shift()) / 2
            self.dict_df_bins_[col] = df_bins.fillna(-np.inf)
        return self

    def transform(self, X: NDArray) -> NDArray:
        """
        Transform X to existing values learned during fit.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            The data to transform.

        Returns
        -------
        X_out : ndarray of shape (n_samples,)
            Transformed input.
        """
        X_arr = check_array(X, accept_sparse=False, force_all_finite="allow-nan", ensure_2d=False)
        df = pd.DataFrame(X_arr)
        list_values_out = []
        for col in df:
            values = df[col]
            if col in self.dict_df_bins_.keys():
                df_bins = self.dict_df_bins_[col]
                bins_X = np.digitize(values, df_bins["min"]) - 1
                values_out = df_bins.loc[bins_X, "value"].values
                values_out = np.where(np.isnan(values), np.nan, values_out)
            else:
                values_out = values
            list_values_out.append(values_out)
        X_out = np.vstack(list_values_out).T
        X_out = X_out.reshape(X_arr.shape)
        if isinstance(X, pd.DataFrame):
            X_out = pd.DataFrame(X_out, index=X.index, columns=X.columns)
        elif isinstance(X, pd.Series):
            X_out = pd.Series(X_out, index=X.index)
        return X_out

    def inverse_transform(self, X: NDArray) -> NDArray:
        """
        Transform X to existing values learned during fit.

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

    def _more_tags(self):
        """
        This method indicates that this class allows inputs with categorical data and nans. It
        modifies the behaviour of the functions checking data.
        """
        return {"X_types": ["2darray"], "allow_nan": True}


class WrapperTransformer(TransformerMixin, BaseEstimator):
    """
    Wraps a transformer with reversible transformers designed to embed the data.
    """

    def __init__(self, transformer: TransformerMixin, wrapper: TransformerMixin):
        super().__init__()
        self.transformer = transformer
        self.wrapper = wrapper

    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> Self:
        X_transformed = copy.deepcopy(X)
        X_transformed = self.wrapper.fit_transform(X_transformed)
        X_transformed = self.transformer.fit(X_transformed)
        return self

    def fit_transform(self, X: NDArray) -> Self:
        X_transformed = copy.deepcopy(X)
        X_transformed = self.wrapper.fit_transform(X_transformed)
        # print("Shape after transformation:", X_transformed.shape)
        X_transformed = self.transformer.fit_transform(X_transformed)
        X_transformed = self.wrapper.inverse_transform(X_transformed)
        return X_transformed

    def transform(self, X: NDArray) -> Self:
        X_transformed = copy.deepcopy(X)
        X_transformed = self.wrapper.transform(X_transformed)
        X_transformed = self.transformer.transform(X_transformed)
        X_transformed = self.wrapper.inverse_transform(X_transformed)
        return X_transformed


def make_pipeline_mixte_preprocessing(
    scale_numerical: bool = True,
) -> BaseEstimator:
    """
    Create a preprocessing pipeline managing mixed type data by one hot encoding categorical data.


    Parameters:
    -----------
    scale_numerical : bool, default=True
        Whether to scale numerical features.

    Returns:
    --------
    preprocessor : Pipeline
        Preprocessing pipeline
    """
    if scale_numerical:
        transformers = [("num", StandardScaler(), selector(dtype_include=np.number))]
    else:
        transformers = []
    transformers.append(
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", use_cat_names=True),
            selector(dtype_exclude=np.number),
        )
    )
    preprocessor = ColumnTransformer(transformers=transformers).set_output(transform="pandas")
    return preprocessor


def make_robust_MixteHGB(scale_numerical: bool = True, allow_new: bool = True) -> Pipeline:
    """
    Create a robust pipeline for MixteHGBM by one hot encoding categorical features.
    This estimator is intended for use in ImputerRegressor to deal with mixed type data.

    Note that from sklearn 1.4 HistGradientBoosting Natively Supports Categorical DTypes in
    DataFrames, so that this pipeline is not required anymore.


    Parameters:
    -----------
    scale_numerical : bool, default=True
        Whether to scale numerical features.
    allow_new : bool, default=True
        Whether to allow new categories.

    Returns:
    --------
    robust_MixteHGB : object
        A robust pipeline for MixteHGBM.
    """
    preprocessor = make_pipeline_mixte_preprocessing(
        scale_numerical=scale_numerical,
    )
    robust_MixteHGB = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("estimator", MixteHGBM(allow_new=allow_new)),
        ]
    )

    return robust_MixteHGB
