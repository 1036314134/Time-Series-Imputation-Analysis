import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

from dataset.draw import show_pred_result

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

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
    # 提取时间戳与目标列
    time_col = df.columns[0]
    target_col = df.columns[-1]

    # 转换时间戳列为 datetime
    df[time_col] = pd.to_datetime(df[time_col])

    # 设置时间戳为索引
    df = df.set_index(time_col)[target_col]

    # 自动推断时间序列频率
    df = df.asfreq(pd.infer_freq(df.index))

    # 切出训练数据与测试数据
    ts = df.iloc[:-forecast_steps]
    pred_truth = df.iloc[-forecast_steps:]

    # -------- 构建 ARIMA 模型 --------
    print(f"正在训练 ARIMA 模型，参数 order={order} ...")
    model = ARIMA(ts, order=order)
    model_fit = model.fit()

    # -------- 输出模型摘要 --------
    print("\n模型拟合摘要：")
    print(model_fit.summary())

    # -------- 预测 --------
    forecast = model_fit.forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=ts.index[-1], periods=forecast_steps + 1, freq='D')[1:]
    forecast_series = pd.Series(forecast, index=forecast_index, name='forecast')

    # -------- 合并结果 --------
    forecast_df = pd.concat([ts, forecast_series], axis=0)

    # -------- 绘图 --------
    if plot:
        show_pred_result(pred_truth, forecast_series, "ARIMA", target_col)

    return forecast_df

