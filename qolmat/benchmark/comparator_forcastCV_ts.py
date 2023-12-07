from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from qolmat.utils import data
from qolmat.benchmark import metrics
from IPython.display import display

import cmdstanpy
import logging

logging.getLogger("cmdstanpy").disabled = True


def features_engineering(df_train: pd.DataFrame, df_buffer: pd.DataFrame, df_test: pd.DataFrame):
    df_concat = pd.concat([df_train, df_buffer, df_test])
    df_test_copy = df_test.copy()
    for col in df_train.columns:
        rolling_means = (
            df_concat[col]
            .groupby(df_concat.index.dayofyear)
            .rolling(3, min_periods=1)
            .mean()
            .shift(1)
        )
        rolling_means = rolling_means.reset_index(level=0, drop=True)
        df_test_copy[col + "_mean"] = rolling_means.loc[df_test.index]
    return df_train, df_test_copy


class CompareForecastTS:
    def __init__(
        self,
        df_data: pd.DataFrame,
        dict_imputers: Dict[str, Any],
        dict_predictors: Dict[str, Any],
        target: str,
        horizon_length: int = 30,
        test_size: int = 30,
        ratio_masked: float = 0.3,
        mean_size: int = 40,
        nb_splits: int = 2,
    ):
        self.df_data = df_data
        self.dict_imputers = dict_imputers
        self.dict_predictors = dict_predictors
        self.target = target
        self.horizon_length = horizon_length
        self.test_size = test_size
        self.ratio_masked = ratio_masked
        self.mean_size = mean_size
        self.nb_splits = nb_splits

    def generate_train_test_splits(self):
        dfs_splits = []
        tscv = TimeSeriesSplit(n_splits=self.nb_splits, test_size=self.test_size)
        for train_index, test_index in tscv.split(self.df_data):
            df_train = self.df_data.iloc[train_index[: -self.horizon_length]]
            df_buffer = self.df_data.iloc[train_index[-self.horizon_length :]]
            df_test = self.df_data.iloc[test_index]
            df_test = df_test[[self.target]]
            dfs_splits.append((df_train, df_buffer, df_test))
        return dfs_splits

    def generate_holes_mcar(
        self, dfs_split: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        dfs_split_hole_mcar = []
        for (df_train, df_buffer, df_test) in dfs_split:
            # Copy
            df_train_mcar = df_train.copy()
            # Add Holes and create new column for the holes position
            df_train_mcar[self.target] = data.add_holes(
                pd.DataFrame(df_train[self.target]),
                ratio_masked=self.ratio_masked,
                mean_size=self.mean_size,
            )
            df_train_mcar[self.target + "_mask"] = 1 - df_train_mcar[[self.target]].isna() * 1
            dfs_split_hole_mcar.append((df_train_mcar, df_buffer, df_test))
        return dfs_split_hole_mcar

    def compare(self):

        dfs_forcast = []
        all_forecast_metrics = []

        # Splits and Hole Generator
        dfs_split = self.generate_train_test_splits()
        dfs_split_hole_mcar = self.generate_holes_mcar(dfs_split)
        # Metrics
        fun_metric = metrics.get_metric("mae")
        fun_metricCor = metrics.get_metric("dist_corr_pattern")

        for split_number, (df_train_hole, df_buffer, df_test) in enumerate(dfs_split_hole_mcar):
            # DataFrame of Metrics
            df_forecast_metric = pd.DataFrame(
                columns=self.dict_imputers.keys(), index=self.dict_predictors.keys()
            )
            df_train_nonan, df_buffer_nonan, df_test_nonan = dfs_split[split_number]

            # ## XGboost With Nan ##
            # predictor_xgboost_with_nan = PredictorXgboost_with_nan()
            # predictions_df, value_to_set = predictor_xgboost_with_nan.metric_xgboost_no_impute(
            #   df_train_hole, df_buffer, df_test)
            # df_forecast_metric.loc["XGBoost", "xgboost_no_impute"] = value_to_set # Save Metrics
            # ##

            for num_imputer, (name_imputer, imputer) in enumerate(self.dict_imputers.items()):
                # Impute and compute metrics
                df_train_imputer = imputer.fit_transform(df_train_hole)
                df_forecast_metric.loc["1 - Metric Imputer MAE", name_imputer] = fun_metric(
                    df_train_nonan[[self.target]],
                    df_train_imputer[[self.target]],
                    df_train_hole[[self.target]].isna(),
                ).values[0]
                df_forecast_metric.loc["2 - Metric Imputer Correlation", name_imputer] = np.abs(
                    fun_metricCor(
                        df_train_nonan[[self.target]],
                        df_train_imputer[[self.target]],
                        df_train_hole[[self.target]].isna(),
                    ).values[0]
                )

                # Add Features Engineering
                df_train_mcar_FE, df_test_FE = features_engineering(
                    df_train=df_train_imputer, df_buffer=df_buffer, df_test=df_test
                )
                for name_predictor, predictor in self.dict_predictors.items():
                    ## Predict and compute metrics
                    predictor.fit(df_train_mcar_FE)
                    forecast = predictor.predict(df_test_FE)
                    forecast.index, forecast.name = df_test_FE.index, self.target
                    df_forecast = pd.DataFrame(forecast)
                    dfs_forcast.append(df_forecast)
                    df_forecast_metric.loc[name_predictor, name_imputer] = fun_metric(
                        df_test_FE[[self.target]],
                        df_forecast,
                        ~df_test_FE[[self.target]].isna(),
                    ).values[0]
                    ##

                    if num_imputer == 0:
                        ## Predict without hole and compute metrics
                        df_train_nonan_FE, df_test_nonan_FE = features_engineering(
                            df_train=df_train_nonan,
                            df_buffer=df_buffer_nonan,
                            df_test=df_test_nonan,
                        )
                        predictor.fit(df_train_nonan_FE)
                        forecast = predictor.predict(df_test_nonan_FE)
                        forecast.index, forecast.name = df_test_FE.index, self.target
                        df_forecast = pd.DataFrame(forecast)
                        dfs_forcast.append(df_forecast)
                        df_forecast_metric.loc[name_predictor, "witout imputation"] = fun_metric(
                            df_test_nonan_FE[[self.target]],
                            df_forecast,
                            ~df_test_nonan_FE[[self.target]].isna(),
                        ).values[0]
                        ##

            print("Split:", split_number + 1)
            display(df_forecast_metric.style.highlight_min(color="green", axis=1))
            all_forecast_metrics.append(df_forecast_metric)
        return all_forecast_metrics


class PredictorProphet:
    def __init__(self, target, daily_seasonality=True, yearly_seasonality=True):
        self.target = target
        self.daily_seasonality = daily_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.model = None

    def fit(self, df):
        prophet_data = df.reset_index()
        prophet_data.rename(columns={"datetime": "ds", self.target: "y"}, inplace=True)
        self.model = Prophet(
            daily_seasonality=self.daily_seasonality, yearly_seasonality=self.yearly_seasonality
        )
        self.model.fit(prophet_data)

    def predict(self, df):
        if self.model is None:
            raise Exception("The model is not fit.")
        X_predict = self.model.make_future_dataframe(periods=len(df), freq="D")
        forecast = self.model.predict(X_predict)
        forecast = forecast[-len(df) :]["yhat"]
        return forecast


class PredictorESM:
    def __init__(self, target, smoothing_level=0.01, smoothing_seasonal=0.8):
        self.target = target
        self.smoothing_level = smoothing_level
        self.smoothing_seasonal = smoothing_seasonal
        self.model = None

    def fit(self, df):
        self.model = ExponentialSmoothing(
            df[self.target], seasonal="add", seasonal_periods=365, freq="D"
        )
        self.model = self.model.fit(
            smoothing_level=self.smoothing_level, smoothing_seasonal=self.smoothing_seasonal
        )

    def predict(self, df):
        if self.model is None:
            raise Exception("The model is not fit.")
        return self.model.forecast(len(df[self.target]))


class PredictorLinearRegression:
    def __init__(self, target):
        self.target = target
        self.model = LinearRegression()

    def fit(self, df):
        X_train = df.drop(self.target, axis=1)
        y_train = df[self.target]
        self.model.fit(X_train, y_train)

    def predict(self, df):
        if self.model is None:
            raise Exception("The model is not fit.")
        X_predict = df.drop([self.target + "_mean", self.target], axis=1)
        X_predict = X_predict.rename(columns=lambda x: x.replace("_mean", ""))
        forecast = self.model.predict(X_predict)
        forecast = pd.DataFrame(forecast, columns=[self.target])
        return forecast


class PredictorXgboost:
    def __init__(self, target):
        self.target = target
        self.model = XGBRegressor()

    def fit(self, df):
        X_train = df.drop(self.target, axis=1)
        y_train = df[self.target]
        self.model.fit(X_train, y_train)

    def predict(self, df):
        if self.model is None:
            raise Exception("The model is not fit.")
        X_predict = df.drop([self.target + "_mean", self.target], axis=1)
        X_predict = X_predict.rename(columns=lambda x: x.replace("_mean", ""))
        forecast = self.model.predict(X_predict)
        forecast = pd.DataFrame(forecast, columns=[self.target])
        return forecast


class PredictorXgboost_with_nan:
    def __init__(self, target):
        self.target = target
        self.model = XGBRegressor()

    def metric_xgboost_no_impute(
        self, df_train: pd.DataFrame, df_buffer: pd.DataFrame, df_test: pd.DataFrame
    ):

        df_train_FE, df_test_FE = features_engineering(df_train, df_buffer, df_test)
        X_predict = df_test_FE.drop([self.target + "_mean", self.target], axis=1)
        X_predict = X_predict.rename(columns=lambda x: x.replace("_mean", ""))
        X_predict = X_predict[["DEWP", "PRES", "time_cos"]]

        model = XGBRegressor()
        model.fit(df_train_FE.drop([self.target], axis=1), df_train_FE[self.target])
        predictions_xgboost = model.predict(X_predict)
        predictions_df = pd.DataFrame(predictions_xgboost, columns=[self.target])
        predictions_df.index = X_predict.index
        df_origin = pd.DataFrame(df_test_FE[self.target])
        df_mask = ~df_origin.isna()
        fun_metric = metrics.get_metric("mae")
        value_to_set = fun_metric(df_origin, predictions_df, df_mask).values[0]
        return predictions_df, value_to_set
