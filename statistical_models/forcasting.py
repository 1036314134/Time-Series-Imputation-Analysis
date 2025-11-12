import pandas as pd
from statistical_models.arima import arima_forecast

if __name__ == "__main__":
    dataset_path = '../dataset/exchange_rate/exchange_rate.csv'
    df = pd.read_csv(dataset_path)

    arima_forecast(df, forecast_steps=10, order=(2, 1, 2))