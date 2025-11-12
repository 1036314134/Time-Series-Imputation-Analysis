import pandas as pd
import numpy as np
import warnings
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX
from dataset.draw import show_pred_result
from statistical_models.arima import data_preprocessing

warnings.filterwarnings("ignore")


def auto_sarima_forecast(
    df,
    forecast_steps=10,
    max_p=2,
    max_d=1,
    max_q=2,
    max_P=1,
    max_D=1,
    max_Q=1,
    seasonal=True,
    m=None,
    plot=True
):
    """
    自动确定最优 SARIMA(p,d,q)(P,D,Q,m) 参数，并进行预测。
    使用 statsmodels 实现，无需 pmdarima。

    参数：
        df: DataFrame，第一列为时间戳，最后一列为目标变量
        forecast_steps: 预测步数
        max_p, max_d, max_q: 非季节部分最大参数
        max_P, max_D, max_Q: 季节部分最大参数
        seasonal: 是否启用季节性
        m: 季节周期长度（None 则自动推断）

    返回：
        forecast_df: 包含预测结果及置信区间的 DataFrame
        best_order: (p, d, q)
        best_seasonal_order: (P, D, Q, m)
    """

    # ==== 1. 数据预处理 ====
    ts, pred_truth, freq = data_preprocessing(df, forecast_steps)

    # 若 m 未指定，则自动推断季节长度
    if seasonal and m is None:
        if freq.upper().startswith("M"):
            m = 12   # 月频 → 年季节性
        elif freq.upper().startswith("W"):
            m = 52   # 周频 → 年季节性
        elif freq.upper().startswith("D"):
            m = 7    # 日频 → 周季节性
        elif freq.upper().startswith("H"):
            m = 24   # 小时频 → 日季节性
        else:
            m = 1    # 默认无季节

    # ==== 2. 参数搜索 ====
    print("正在搜索最优 SARIMA 参数，请稍候...")
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    best_model = None

    for p, d, q in product(range(max_p + 1), range(max_d + 1), range(max_q + 1)):
        for P, D, Q in product(range(max_P + 1), range(max_D + 1), range(max_Q + 1)):
            seasonal_order = (P, D, Q, m) if seasonal else (0, 0, 0, 0)
            try:
                model = SARIMAX(
                    ts,
                    order=(p, d, q),
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False)
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = (p, d, q)
                    best_seasonal_order = seasonal_order
                    best_model = result
            except Exception:
                continue

    print(f"✅ 最优 SARIMA 参数: order={best_order}, seasonal_order={best_seasonal_order}, AIC={best_aic:.2f}")

    # ==== 3. 预测 ====
    forecast = best_model.get_forecast(steps=forecast_steps)
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

    return forecast_df, best_order, best_seasonal_order


# ==== 示例使用 ====
if __name__ == "__main__":
    # 生成示例时间序列（带季节性）
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    y = 10 + np.sin(np.arange(200) / 10) * 2 + np.random.normal(0, 0.5, 200)
    df = pd.DataFrame({"date": dates, "value": y})

    forecast_df, order, seasonal_order = auto_sarima_forecast(df, forecast_steps=14,)

    print("\n--- 预测结果 ---")
    print(forecast_df)