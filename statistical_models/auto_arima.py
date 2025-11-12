import pandas as pd
import numpy as np
import warnings
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from dataset.draw import show_pred_result
from statistical_models.arima import data_preprocessing

warnings.filterwarnings("ignore")


def auto_arima_forecast(df, forecast_steps=10, max_p=3, max_d=2, max_q=3, plot=True):
    """
    自动确定 ARIMA(p, d, q) 最优参数，并进行预测。
    仅使用 statsmodels，不依赖 pmdarima。

    参数：
        df: DataFrame，第一列为时间戳，最后一列为目标变量
        forecast_steps: 预测步数
        max_p, max_d, max_q: 搜索范围上限

    返回：
        forecast_df: 包含预测结果的 DataFrame
        best_order: 最优 (p, d, q)
    """

    # ==== 1. 准备数据 ====
    ts, pred_truth, freq = data_preprocessing(df, forecast_steps)

    # ==== 2. 搜索最优参数 ====
    best_aic = np.inf
    best_order = None
    best_model = None

    for p, d, q in product(range(max_p + 1), range(max_d + 1), range(max_q + 1)):
        try:
            model = ARIMA(ts, order=(p, d, q))
            result = model.fit()
            if result.aic < best_aic:
                best_aic = result.aic
                best_order = (p, d, q)
                best_model = result
        except Exception:
            continue

    print(f"✅ 最优 ARIMA 参数: (p, d, q) = {best_order}, AIC = {best_aic:.2f}")

    # ==== 3. 进行预测 ====
    forecast = best_model.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    forecast_index = pd.date_range(
        start=ts.index[-1], periods=forecast_steps + 1, freq=pd.infer_freq(ts.index)
    )[1:]

    forecast_df = pd.DataFrame({
        'forecast': forecast_mean.values,
        'lower_ci': forecast_ci.iloc[:, 0].values,
        'upper_ci': forecast_ci.iloc[:, 1].values
    }, index=forecast_index)

    forecast_series = forecast_df.iloc[-forecast_steps:, 0]

    # -------- draw figure --------
    if plot:
        show_pred_result(pred_truth, forecast_series, "AUTO-ARIMA", df.columns[-1])

    return forecast_df, best_order


# ==== 示例使用 ====
if __name__ == "__main__":
    # 生成示例时间序列
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    y = np.sin(np.arange(100) / 5) + np.random.normal(0, 0.3, 100)
    df = pd.DataFrame({'date': dates, 'value': y})

    forecast_df, order = auto_arima_forecast(df, forecast_steps=10)
    print(forecast_df)
