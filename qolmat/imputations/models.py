from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from qolmat.imputations.rpca.pcp_rpca import RPCA
from qolmat.imputations.rpca.temporal_rpca import TemporalRPCA, OnlineTemporalRPCA
from qolmat.benchmark import utils
import os
import sys


class ImputeColumnWise:
    def __init__(
        self,
        groups=[],
    ) -> None:
        self.groups = groups

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:

        col_to_impute = df.columns
        imputed = df.copy()
        for col in col_to_impute:
            # df_col = df[col].reset_index()
            imputed[col] = self.fit_transform_col(df[col]).values
        imputed.fillna(0, inplace=True)
        return imputed

    def get_hyperparams(self):
        return {}


class ImputeByMean(ImputeColumnWise):
    def __init__(
        self,
        groups=[],
    ) -> None:
        super().__init__(groups=groups)

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        col = signal.name
        signal = signal.reset_index()
        # imputed = utils.custom_groupby(signal, self.groups)[[col_to_impute]].apply(lambda x: x.fillna(x.mean()))
        imputed = signal[col].fillna(
            utils.custom_groupby(signal, self.groups)[col].transform("mean")
        )
        return imputed


class ImputeByMedian(ImputeColumnWise):
    def __init__(
        self,
        groups=[],
    ) -> None:
        super().__init__(groups=groups)

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        col = signal.name
        signal = signal.reset_index()
        # imputed = utils.custom_groupby(signal, self.groups)[[col_to_impute]].apply(lambda x: x.fillna(x.mean()))
        imputed = signal[col].fillna(
            utils.custom_groupby(signal, self.groups)[col].transform("median")
        )
        # imputed = signal[col].groupby(self.groups).transform("median")
        return imputed


class RandomImpute(ImputeColumnWise):
    def __init__(
        self,
    ) -> None:
        pass

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        col = signal.name
        imputed = signal.reset_index()
        number_missing = imputed[col].isnull().sum()
        obs = imputed.loc[imputed[col].notnull(), col].values
        imputed.loc[imputed[col].isnull(), col] = np.random.choice(
            obs, number_missing, replace=True
        )
        return imputed[col]


class ImputeLOCF(ImputeColumnWise):
    def __init__(
        self,
        groups=[],
    ) -> None:
        super().__init__(groups=groups)

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        col = signal.name
        imputed = signal.reset_index()
        imputed = utils.custom_groupby(imputed, self.groups)[col].transform(
            lambda x: x.ffill()
        )
        return imputed.fillna(np.nanmedian(imputed))


class ImputeNOCB(ImputeColumnWise):
    def __init__(
        self,
        groups=[],
    ) -> None:
        super().__init__(groups=groups)

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        col = signal.name
        imputed = signal.reset_index()
        imputed = utils.custom_groupby(imputed, self.groups)[col].transform(
            lambda x: x.bfill()
        )
        return imputed.fillna(np.nanmedian(imputed))


class ImputeKNN(ImputeColumnWise):
    def __init__(self, **kwargs) -> None:
        for name, value in kwargs.items():
            setattr(self, name, value)

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        col = signal.name
        signal = signal.reset_index()
        imputed = np.asarray(signal[col]).reshape(-1, 1)
        imputer = KNNImputer(n_neighbors=self.k)
        imputed = imputer.fit_transform(imputed)
        imputed = pd.Series([a[0] for a in imputed], index=signal.index)
        return imputed.fillna(np.nanmedian(imputed))

    def get_hyperparams(self) -> Dict[str, int]:
        return {"k": self.k}


class ImputeByInterpolation(ImputeColumnWise):
    def __init__(self, **kwargs) -> None:
        for name, value in kwargs.items():
            setattr(self, name, value)

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        col = signal.name
        signal = signal.reset_index()
        signal = signal.set_index("datetime")
        return signal[col].interpolate(method=self.method)


class ImputeBySpline(ImputeColumnWise):
    def __init__(
        self,
    ) -> None:
        pass

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        return signal.interpolate(option="spline")


class ImputeRPCA:
    def __init__(self, method, multivariate=False, **kwargs) -> None:
        self.multivariate = multivariate

        if method == "PCP":
            self.rpca = RPCA()
        elif method == "temporal":
            self.rpca = TemporalRPCA()
        elif method == "online":
            self.rpca = OnlineTemporalRPCA()
        for name, value in kwargs.items():
            setattr(self.rpca, name, value)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:

        if self.multivariate:
            imputed, _, _ = self.rpca.fit_transform(signal=df.values)
            imputed = pd.DataFrame(imputed, columns=df.columns)
        else:
            imputed = pd.DataFrame()
            for col in df.columns:
                imputed_signal, _, _ = self.rpca.fit_transform(signal=df[col].values)
                imputed[col] = imputed_signal
        imputed.index = df.index

        return imputed

    def get_hyperparams(self):
        return self.rpca.__dict__


class ImputeIterative:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        iterative_imputer = IterativeImputer(**(self.kwargs))
        res = iterative_imputer.fit_transform(df.values)
        imputed = pd.DataFrame(columns=df.columns)
        for ind, col in enumerate(imputed.columns):
            imputed[col] = res[:, ind]
        imputed.index = df.index
        return imputed

    def get_hyperparams(self):
        return self.__dict__["kwargs"]


class ImputeRegressor:
    def __init__(self, model, cols_to_impute=[], **kwargs) -> None:
        self.model = model
        self.cols_to_impute = cols_to_impute
        for name, value in kwargs.items():
            setattr(self, name, value)

    def fit_transform(self, df: pd.DataFrame) -> pd.Series:
        df_imp = df.copy()
        if self.cols_to_impute == []:
            self.cols_to_impute = df.columns.tolist()
        X = df[[col for col in df.columns if col not in self.cols_to_impute]]
        for col in self.cols_to_impute:
            y = df[col]
            is_na = y.isna()
            self.model.fit(X[~is_na], y[~is_na])
            df_imp.loc[is_na, col] = self.model.predict(X[is_na])

        return df_imp

    def get_hyperparams(self):
        return self.__dict__

    def create_features(self, df):
        return None


class ImputeStochasticRegressor:
    def __init__(self, model, cols_to_impute=[], **kwargs) -> None:
        self.model = model
        self.cols_to_impute = cols_to_impute
        for name, value in kwargs.items():
            setattr(self, name, value)

    def fit_transform(self, df: pd.DataFrame) -> pd.Series:
        df_imp = df.copy()
        if self.cols_to_impute == []:
            self.cols_to_impute = df.columns.tolist()
        X = df[[col for col in df.columns if col not in self.cols_to_impute]]
        for col in self.cols_to_impute:
            y = df[col]
            is_na = y.isna()
            self.model.fit(X[~is_na], y[~is_na])
            y_pred = self.model.predict(X)
            std_error = (y_pred[~is_na] - y[~is_na]).std()
            random_pred = np.random.normal(size=len(y), loc=y_pred, scale=std_error)
            df_imp.loc[is_na, col] = random_pred[is_na]

        return df_imp

    def get_hyperparams(self):
        return self.__dict__

    def create_features(self, df):
        return None
