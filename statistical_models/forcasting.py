import pandas as pd
from statistical_models.arima import arima_forecast
from statistical_models.auto_arima import auto_arima_forecast
from statistical_models.auto_sarima import auto_sarima_forecast
from statistical_models.sarima import sarima_forecast

if __name__ == "__main__":
    dataset_path = '../dataset/exchange_rate/exchange_rate.csv'
    df = pd.read_csv(dataset_path)

    # --------use arima---------
    arima_forecast(df, forecast_steps=10)

    # --------use auto-arima---------
    auto_arima_forecast(df, forecast_steps=10)

    # --------use sarima---------
    sarima_forecast(df, forecast_steps=10)

    # --------use auto-sarima---------
    auto_sarima_forecast(df, forecast_steps=10)