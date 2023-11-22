from typing import Any, Dict, List, Tuple
from datetime import timedelta
import pandas as pd

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
    for col in df_train.columns:
        rolling_means = (
            df_concat[col]
            .groupby(df_concat.index.dayofyear)
            .rolling(3, min_periods=1)
            .mean()
            .shift(1)
        )
        rolling_means = rolling_means.reset_index(level=0, drop=True)
        df_test[col + "_mean"] = rolling_means.loc[df_test.index]
    return df_train, df_test


class CompareForecastTS:
    def __init__(
        self,
        df_data: pd.DataFrame,
        dict_imputers: Dict[str, Any],
        dict_predictors: Dict[str, Any],
        horizon_length: int = 30,
        test_size: int = 30,
        ratio_masked: float = 0.3,
        mean_size: int = 40,
        nb_splits: int = 2,
    ):
        self.df_data = df_data
        self.dict_imputers = dict_imputers
        self.dict_predictors = dict_predictors
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
            df_test = df_test.drop(["DEWP", "PRES", "time_cos"], axis=1)
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
            df_train_mcar["TEMP"] = data.add_holes(
                pd.DataFrame(df_train["TEMP"]),
                ratio_masked=self.ratio_masked,
                mean_size=self.mean_size,
            )
            df_train_mcar["TEMP_mask"] = 1 - df_train_mcar[["TEMP"]].isna().astype(int)
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
        fun_metricKL = metrics.get_metric("KL_columnwise")

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

            for name_imputer, imputer in self.dict_imputers.items():
                # Impute and compute metrics
                df_train_imputer = imputer.fit_transform(df_train_hole)
                df_forecast_metric.loc["1 - Metric Imputer MAE", name_imputer] = fun_metric(
                    pd.DataFrame(df_train_nonan["TEMP"]),
                    pd.DataFrame(df_train_imputer["TEMP"]),
                    pd.DataFrame(df_train_hole["TEMP"].isna()),
                ).values[0]
                df_forecast_metric.loc["2 - Metric Imputer KL", name_imputer] = fun_metricKL(
                    pd.DataFrame(df_train_nonan["TEMP"]),
                    pd.DataFrame(df_train_imputer["TEMP"]),
                    pd.DataFrame(df_train_hole["TEMP"].isna()),
                ).values[0]

                # Add Features Engineering
                df_train_mcar_FE, df_test_FE = features_engineering(
                    df_train=df_train_imputer, df_buffer=df_buffer, df_test=df_test
                )
                for name_predictor, predictor in self.dict_predictors.items():
                    # Predict and compute metrics
                    predictor.fit(df_train_mcar_FE)
                    forecast = predictor.predict(df_test_FE)
                    forecast.index, forecast.name = df_test_FE.index, "TEMP"
                    df_forecast = pd.DataFrame(forecast)
                    dfs_forcast.append(df_forecast)
                    df_forecast_metric.loc[name_predictor, name_imputer] = fun_metric(
                        pd.DataFrame(df_test_FE["TEMP"]),
                        df_forecast,
                        pd.DataFrame(~df_test_FE["TEMP"].isna()),
                    ).values[0]
                    all_forecast_metrics.append(df_forecast_metric)
            print("Split:", split_number + 1)
            display(df_forecast_metric.style.highlight_min(color="green", axis=1))
        return all_forecast_metrics


class PredictorProphet:
    def __init__(self, daily_seasonality=True, yearly_seasonality=True):
        self.daily_seasonality = daily_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.model = None

    def fit(self, df):
        prophet_data = df.reset_index()
        prophet_data.rename(columns={"datetime": "ds", "TEMP": "y"}, inplace=True)
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
    def __init__(self, smoothing_level=0.01, smoothing_seasonal=0.8):
        self.smoothing_level = smoothing_level
        self.smoothing_seasonal = smoothing_seasonal
        self.model = None

    def fit(self, df):
        self.model = ExponentialSmoothing(
            df["TEMP"], seasonal="add", seasonal_periods=365, freq="D"
        )
        self.model = self.model.fit(
            smoothing_level=self.smoothing_level, smoothing_seasonal=self.smoothing_seasonal
        )

    def predict(self, df):
        if self.model is None:
            raise Exception("The model is not fit.")
        return self.model.forecast(len(df["TEMP"]))


class PredictorLinearRegression:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, df):
        X_train = df.drop("TEMP", axis=1)
        y_train = df["TEMP"]
        self.model.fit(X_train, y_train)

    def predict(self, df):
        if self.model is None:
            raise Exception("The model is not fit.")
        X_predict = df.drop(["TEMP_mean", "TEMP"], axis=1)
        X_predict = X_predict.rename(columns=lambda x: x.replace("_mean", ""))
        X_predict = X_predict[["DEWP", "PRES", "time_cos", "TEMP_mask"]]
        forecast = self.model.predict(X_predict)
        forecast = pd.DataFrame(forecast, columns=["TEMP"])
        return forecast


class PredictorXgboost:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, df):
        X_train = df.drop("TEMP", axis=1)
        y_train = df["TEMP"]
        self.model.fit(X_train, y_train)

    def predict(self, df):
        if self.model is None:
            raise Exception("The model is not fit.")
        X_predict = df.drop(["TEMP_mean", "TEMP"], axis=1)
        X_predict = X_predict.rename(columns=lambda x: x.replace("_mean", ""))
        X_predict = X_predict[["DEWP", "PRES", "time_cos", "TEMP_mask"]]
        forecast = self.model.predict(X_predict)
        forecast = pd.DataFrame(forecast, columns=["TEMP"])
        return forecast


class PredictorXgboost_with_nan:
    def __init__(self):
        self.model = LinearRegression()

    def metric_xgboost_no_impute(
        self, df_train: pd.DataFrame, df_buffer: pd.DataFrame, df_test: pd.DataFrame
    ):

        df_train_FE, df_test_FE = features_engineering(df_train, df_buffer, df_test)
        X_predict = df_test_FE.drop(["TEMP_mean", "TEMP"], axis=1)
        X_predict = X_predict.rename(columns=lambda x: x.replace("_mean", ""))
        X_predict = X_predict[["DEWP", "PRES", "time_cos"]]

        model = XGBRegressor()
        model.fit(df_train_FE.drop(["TEMP"], axis=1), df_train_FE["TEMP"])
        predictions_xgboost = model.predict(X_predict)
        predictions_df = pd.DataFrame(predictions_xgboost, columns=["TEMP"])
        predictions_df.index = X_predict.index
        df_origin = pd.DataFrame(df_test_FE["TEMP"])
        df_mask = ~df_origin.isna()
        fun_metric = metrics.get_metric("mae")
        value_to_set = fun_metric(df_origin, predictions_df, df_mask).values[0]
        return predictions_df, value_to_set
