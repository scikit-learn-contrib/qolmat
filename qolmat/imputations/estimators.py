import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
)
from sklearn.utils.validation import (
    check_X_y,
    _check_feature_names_in,
    _num_samples,
    check_array,
    _check_y,
    check_is_fitted,
)

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
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
        self.df_bins_ = None
        if hasattr(self, "args_model"):
            args_model = self.args_model
        else:
            args_model = {}
        if pd.api.types.is_string_dtype(y):
            model = HistGradientBoostingClassifier(**args_model)
        elif pd.api.types.is_numeric_dtype(y):
            model = HistGradientBoostingRegressor(**args_model)
            if not self.allow_new:
                df_bins = pd.DataFrame({"value": np.sort(np.unique(y))})
                df_bins["min"] = (df_bins["value"] + df_bins["value"].shift()) / 2
                self.df_bins_ = df_bins.fillna(-np.inf)

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
        if self.df_bins_ is not None:
            bins_y = np.digitize(y_pred, self.df_bins_["min"]) - 1
            y_pred = self.df_bins_.loc[bins_y, "value"].values
        return y_pred

    def _more_tags(self):
        """
        This method indicates that this class allows inputs with categorical data and nans. It
        modifies the behaviour of the functions checking data.
        """
        return {"X_types": ["2darray", "categorical", "string"], "allow_nan": True}

    # def _validate_input(self, X: NDArray) -> pd.DataFrame:
    #     """
    #     Checks that the input X can be converted into a DataFrame, and returns the corresponding
    #     dataframe.

    #     Parameters
    #     ----------
    #     X : NDArray
    #         Array-like to process

    #     Returns
    #     -------
    #     pd.DataFrame
    #         Formatted dataframe, if the input had no column names then the dataframe columns are
    #         integers
    #     """
    #     check_array(X, force_all_finite="allow-nan", dtype=None)
    #     if not isinstance(X, pd.DataFrame):
    #         X_np = np.array(X)
    #         if len(X_np.shape) == 0:
    #             raise ValueError
    #         if len(X_np.shape) == 1:
    #             X_np = X_np.reshape(-1, 1)
    #         df = pd.DataFrame(X_np, columns=[i for i in range(X_np.shape[1])])
    #         df = df.infer_objects()
    #     else:
    #         df = X
    #     # df = df.astype(float)

    #     return df


def make_robust_MixteHGB(scale_numerical: bool = True, allow_new: bool = True) -> Pipeline:
    """
    Create a robust pipeline for MixteHGBM by one hot encoding categorical features.
    This estimator is intended for use in ImputerRegressor to deal with mixed type data.

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
    if scale_numerical:
        transformers = [("num", StandardScaler(), selector(dtype_include=np.number))]
    else:
        transformers = []
    transformers.append(
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            selector(dtype_exclude=np.number),
        )
    )
    preprocessor = ColumnTransformer(transformers=transformers)
    robust_MixteHGB = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("estimator", MixteHGBM(allow_new=allow_new)),
        ]
    )

    return robust_MixteHGB
