import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

from dataset.draw import show_pred_result
from statistical_models.arima import data_preprocessing

warnings.filterwarnings("ignore")


def sarima_forecast(
    df,
    forecast_steps=10,
    order=(1, 1, 1),
    seasonal_order=(1, 0, 1, 7),
    plot=True
):
    """
    使用指定的 SARIMA 参数进行时间序列预测。
    使用 statsmodels 实现，适配 numpy==1.26.4。

    参数：
        df: DataFrame，第一列为时间戳，最后一列为目标变量
        forecast_steps: 预测步数
        order: (p, d, q)
        seasonal_order: (P, D, Q, m)，其中 m 为季节周期

    返回：
        forecast_df: 包含预测结果及置信区间的 DataFrame
        model_fit: 训练后的模型对象
    """

    # ==== 1. 数据预处理 ====
    ts, pred_truth, freq = data_preprocessing(df, forecast_steps)

    # ==== 2. 拟合模型 ====
    model = SARIMAX(
        ts,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    model_fit = model.fit(disp=False)

    print(f"✅ 模型训练完成: order={order}, seasonal_order={seasonal_order}")
    print(f"AIC = {model_fit.aic:.2f}")

    # ==== 3. 预测 ====
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    forecast_index = pd.date_range(
        start=ts.index[-1],
        periods=forecast_steps + 1,
        freq=freq
    )[1:]

    forecast_df = pd.DataFrame({
        "forecast": forecast_mean.values,
        "lower_ci": forecast_ci.iloc[:, 0].values,
        "upper_ci": forecast_ci.iloc[:, 1].values
    }, index=forecast_index)

    forecast_series = forecast_df.iloc[-forecast_steps:, 0]

    # -------- draw figure --------
    if plot:
        show_pred_result(pred_truth, forecast_series, "Auto-SARIMA", df.columns[-1])

    return forecast_df, model_fit


# ==== 示例使用 ====
if __name__ == "__main__":
    # 构造模拟时间序列
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    y = 10 + np.sin(np.arange(200) / 10) * 2 + np.random.normal(0, 0.5, 200)
    df = pd.DataFrame({"date": dates, "value": y})

    # 手动设定参数
    forecast_df, model_fit = sarima_forecast(
        df,
        forecast_steps=14,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 7)
    )

    print("\n--- 预测结果 ---")
    print(forecast_df.head(10))