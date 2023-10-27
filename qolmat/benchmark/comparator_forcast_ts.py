from typing import Any, Dict, List
import pandas as pd

from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from qolmat.utils import data
from qolmat.benchmark import metrics
from IPython.display import display

import cmdstanpy
import logging

logging.getLogger("cmdstanpy").disabled = True


def forecasters(
    dict_imputers: Dict[str, Any], dfs_train_imputed: Dict[str, Any], df_test: pd.DataFrame
):
    dict_forecast_prophet = {}
    dict_forecast_esm = {}
    dict_forecast_ols = {}
    dict_forecast_xgboost = {}

    for name_imputer in dict_imputers.keys():

        df_train_model = dfs_train_imputed[name_imputer]
        prophet_data = df_train_model.reset_index()
        prophet_data.rename(columns={"datetime": "ds", "TEMP": "y"}, inplace=True)

        X_train = df_train_model.drop("TEMP", axis=1)
        y_train = df_train_model["TEMP"]

        X_predict = df_test.drop(["TEMP_mean", "TEMP"], axis=1)
        X_predict = X_predict.rename(columns=lambda x: x.replace("_mean", ""))
        X_predict = X_predict[["DEWP", "PRES", "time_cos", "TEMP_mask"]]

        modelesm = ExponentialSmoothing(
            df_train_model["TEMP"], seasonal="add", seasonal_periods=365, freq="D"
        )
        model_prophet = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model_ols = LinearRegression()
        model_xgboost = XGBRegressor()

        model_prophet.fit(prophet_data)
        model_esm = modelesm.fit(smoothing_level=0.01, smoothing_seasonal=0.8)
        model_ols.fit(X_train, y_train)
        model_xgboost.fit(X_train, y_train)

        future = model_prophet.make_future_dataframe(periods=len(df_test), freq="D")
        forecast = model_prophet.predict(future)
        forecast_prophet = forecast[-len(df_test) :]["yhat"]
        forecast_esm = model_esm.forecast(len(df_test["TEMP"]))
        forecast_ols = model_ols.predict(X_predict)
        forecast_xgboost = model_xgboost.predict(X_predict)

        forecast_ols = pd.DataFrame(forecast_ols, columns=["TEMP"])
        forecast_xgboost = pd.DataFrame(forecast_xgboost, columns=["TEMP"])
        forecast_prophet.index = df_test.index
        forecast_ols.index = X_predict.index
        forecast_xgboost.index = X_predict.index
        forecast_prophet.name = "TEMP"
        forecast_esm.name = "TEMP"

        dict_forecast_esm[name_imputer] = pd.DataFrame(forecast_esm)
        dict_forecast_prophet[name_imputer] = pd.DataFrame(forecast_prophet)
        dict_forecast_ols[name_imputer] = forecast_ols
        dict_forecast_xgboost[name_imputer] = forecast_xgboost
    return dict_forecast_prophet, dict_forecast_esm, dict_forecast_ols, dict_forecast_xgboost


def holes_mcar_features_engineering(
    df_train: pd.DataFrame, df_test: pd.DataFrame, ratio_masked: float, mean_size: int
):
    df_train_mcar = df_train.copy()
    df_train_mcar["TEMP"] = data.add_holes(
        pd.DataFrame(df_train["TEMP"]), ratio_masked=ratio_masked, mean_size=mean_size
    )
    df_train_mcar["TEMP_mask"] = 1 - df_train_mcar[["TEMP"]].isna() * 1

    for date in df_test.index:
        day_month = date.strftime("%m-%d")
        date_mean = df_train_mcar[df_train_mcar.index.strftime("%m-%d") == day_month].mean()
        df_test.loc[date, "TEMP_mask_mean"] = date_mean["TEMP_mask"]
    return df_train_mcar, df_test


def forecasts_metrics(
    df_test: pd.DataFrame,
    dict_imputers: Dict[str, Any],
    dict_dict_prediction: Dict[str, Dict[str, Any]],
):
    df_origin = pd.DataFrame(df_test["TEMP"])
    df_mask = ~df_origin.isna()

    df_forecast_metric = pd.DataFrame(
        columns=dict_imputers.keys(), index=dict_dict_prediction.keys()
    )
    for name_forecast, dict_forecast in dict_dict_prediction.items():
        for name_imputer, df_forecast in dict_forecast.items():
            fun_metric = metrics.get_metric("mae")
            df_forecast_metric.loc[name_forecast, name_imputer] = fun_metric(
                df_origin, df_forecast, df_mask
            ).values[0]

    return df_forecast_metric


def compare_forecast(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    dict_imputers: Dict[str, Any],
    ration_masked: float,
    mean_size: int,
) -> pd.DataFrame:

    df_train_mcar, df_test = holes_mcar_features_engineering(
        df_train, df_test, ratio_masked=ration_masked, mean_size=mean_size
    )
    dfs_train_imputed = {
        name: imp.fit_transform(df_train_mcar) for name, imp in dict_imputers.items()
    }
    (
        dict_forecast_prophet,
        dict_forecast_esm,
        dict_forecast_ols,
        dict_forecast_xgboost,
    ) = forecasters(dict_imputers, dfs_train_imputed, df_test)
    dict_dict_prediction = {
        "prophet": dict_forecast_prophet,
        "esm": dict_forecast_esm,
        "ols": dict_forecast_ols,
        "xgboost": dict_forecast_xgboost,
    }
    df_forecast_metric = forecasts_metrics(df_test, dict_imputers, dict_dict_prediction)
    return df_forecast_metric


def iter_compare_forecast(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    dict_imputers: Dict[str, Any],
    ration_masked: float,
    mean_size: int,
    nb_iteration: int,
):
    all_forecast_metrics = []

    dict_forecast_xgboost_no_impute = {}

    X_predict = df_test.drop(["TEMP_mean", "TEMP"], axis=1)
    X_predict = X_predict.rename(columns=lambda x: x.replace("_mean", ""))
    X_predict = X_predict[["DEWP", "PRES", "time_cos"]]

    model = XGBRegressor()
    model.fit(df_train.drop(["TEMP"], axis=1), df_train["TEMP"])
    predictions_xgboost = model.predict(X_predict)
    predictions_df = pd.DataFrame(predictions_xgboost, columns=["TEMP"])
    predictions_df.index = X_predict.index
    for name_imputer in dict_imputers.keys():
        dict_forecast_xgboost_no_impute[name_imputer] = predictions_df

    df_origin = pd.DataFrame(df_test["TEMP"])
    df_mask = ~df_origin.isna()
    fun_metric = metrics.get_metric("mae")
    value_to_set = fun_metric(df_origin, predictions_df, df_mask).values[0]

    for iter in range(nb_iteration):
        df_forecast_metric = compare_forecast(
            df_train, df_test, dict_imputers, ration_masked, mean_size
        )
        df_forecast_metric.loc["xgboost", "xgboost_no_impute"] = value_to_set
        all_forecast_metrics.append(df_forecast_metric)
        print("Iteration:", iter + 1)
        display(df_forecast_metric.style.highlight_min(color="green", axis=1))

    combined_df = pd.concat(all_forecast_metrics)
    print(f"Moyenne des {iter+1} itérations:")
    avg_forecast_metric = combined_df.groupby(combined_df.index).mean()

    return avg_forecast_metric
