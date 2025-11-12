import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")


def auto_arima_forecast(df, forecast_steps=10, seasonal=False, m=1, plot=True):
    """
    使用 auto_arima 自动确定最优参数并进行 ARIMA 时序预测

    参数：
        df : pd.DataFrame
            第一列为时间戳，最后一列为需要预测的目标序列
        forecast_steps : int
            预测未来步数
        seasonal : bool
            是否考虑季节性（默认 False）
        m : int
            每个季节的周期（当 seasonal=True 时生效，例如12代表一年12个月）
        plot : bool
            是否绘制预测图像

    返回：
        result_df : pd.DataFrame
            包含历史值与预测值的DataFrame
    """

    # -------- 数据准备 --------
    time_col = df.columns[0]
    target_col = df.columns[-1]

    # 转换时间戳列为 datetime
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)

    # 自动推断频率，避免警告
    try:
        df = df.asfreq(pd.infer_freq(df.index))
    except:
        df = df.asfreq('D')

    ts = df[target_col].astype(float)

    # -------- 自动选择最优 (p, d, q) 参数 --------
    print("正在搜索最优 ARIMA 参数，请稍候...")
    stepwise_model = auto_arima(
        ts,
        seasonal=seasonal,
        m=m,
        trace=True,        # 打印搜索过程
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    print("\n✅ 最优 ARIMA 参数:", stepwise_model.order)
    if seasonal:
        print("✅ 最优季节性参数:", stepwise_model.seasonal_order)

    # -------- 用最优参数拟合模型 --------
    model = ARIMA(ts, order=stepwise_model.order)
    model_fit = model.fit()

    # -------- 预测 --------
    forecast = model_fit.forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=ts.index[-1], periods=forecast_steps + 1, freq=ts.index.freq)[1:]
    forecast_series = pd.Series(forecast, index=forecast_index, name='forecast')

    # -------- 合并结果 --------
    result_df = pd.concat([ts, forecast_series], axis=0)

    # -------- 绘图 --------
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(ts, label='历史数据', color='blue')
        plt.plot(forecast_series, label='预测值', color='red', linestyle='--')
        plt.title(f"ARIMA{stepwise_model.order} 模型预测结果")
        plt.xlabel("时间")
        plt.ylabel(target_col)
        plt.legend()
        plt.grid(True)
        plt.show()

    return result_df