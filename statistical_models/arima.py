import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from dataset.draw import show_pred_result

def data_preprocessing(df, forecast_steps):
    df = df.copy()

    # 提取时间戳与目标列
    time_col = df.columns[0]
    target_col = df.columns[-1]

    # 转换时间戳列为 datetime
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])

    # 设置时间戳为索引
    df.set_index(time_col, inplace=True)

    # 切出训练数据与测试数据
    data = df[target_col].astype(float)
    ts = data.iloc[:-forecast_steps]
    pred_truth = data.iloc[-forecast_steps:]

    # 自动推断频率
    freq = pd.infer_freq(ts.index)
    if freq is None:
        freq = "D"  # 默认日频率
    ts = ts.asfreq(freq)

    return ts, pred_truth, freq


def arima_forecast(df, forecast_steps=10, order=(1, 1, 1), plot=True):
    """
    使用 ARIMA 模型对时间序列进行预测

    参数:
        df : pd.DataFrame
            第一列为时间戳，最后一列为需要预测的目标序列
        forecast_steps : int
            预测步数（默认 10）
        order : tuple
            ARIMA 模型参数 (p, d, q)，默认 (1, 1, 1)
        plot : bool
            是否绘制预测图像

    返回:
        forecast_df : pd.DataFrame
            包含历史值与预测值的DataFrame
    """
    # -------- 数据准备 --------
    ts, pred_truth, freq = data_preprocessing(df, forecast_steps)

    # -------- 构建 ARIMA 模型 --------
    print(f"正在训练 ARIMA 模型，参数 order={order} ...")
    model = ARIMA(ts, order=order)
    model_fit = model.fit()

    # -------- 输出模型摘要 --------
    print("\n模型拟合摘要：")
    print(model_fit.summary())

    # -------- forcasting --------
    forecast = model_fit.forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=ts.index[-1], periods=forecast_steps + 1, freq='D')[1:]
    forecast_series = pd.Series(forecast, index=forecast_index, name='forecast')

    # -------- 合并结果 --------
    result_df = pd.concat([ts, forecast_series], axis=0)

    # -------- draw figure --------
    if plot:
        show_pred_result(pred_truth, forecast_series, "ARIMA", df.columns[-1])

    return result_df


# ==== 示例使用 ====
if __name__ == "__main__":
    # 生成示例时间序列
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    y = np.sin(np.arange(100) / 5) + np.random.normal(0, 0.3, 100)
    df = pd.DataFrame({'date': dates, 'value': y})

    forecast_df, order = arima_forecast(df, forecast_steps=10)
    print(forecast_df)