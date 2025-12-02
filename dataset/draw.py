import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 尝试加载常见中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def show_specify_line(df, columns, colour="green"):
    """
    将数据中指定的序列展示出来
    """
    for column in columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[column], linestyle='-', color=colour)
        plt.title(f'{column}')
        plt.grid(False)

    plt.show()


def show_together(df, need_columns, colour):
    """
    一列展示多条挑序列的原始数据
    """
    # 创建一个大的图形对象
    fig, axes = plt.subplots(nrows=len(need_columns), ncols=1, figsize=(6, 3 * len(need_columns)))

    # 循环绘制每列数据的折线图
    for ax, column in zip(axes, need_columns):
        ax.plot(df.index, df[column], linestyle='-', color=colour, label='Origin')
        ax.set_title(f'{column}')
        ax.grid(False)
        ax.legend()

    # 调整布局以避免重叠
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def show_change(df_origin, df_cleaned, method, columns=None):
    """
    依次展示单挑序列的原始数据与修复数据
    """
    if columns is None:
        columns = ["OT"]
    for column in columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df_origin.index, df_origin[column], label='origin data', linestyle='-', color='green')
        # plt.plot(df_cleaned.index, df_cleaned[column], label='imputed data', linestyle='-', color='blue')
        plt.plot(df_cleaned.index, df_cleaned[column], label='reconstructing data', linestyle='-', color='blue')
        plt.title(f"{method}填补结果")
        plt.ylabel(f'{column}')
        plt.legend()
        plt.grid(False)

    plt.show()


def show_pred_result(pred_truth, forecast_series, method, column):
    plt.figure(figsize=(10, 5))
    plt.plot(pred_truth, label='真实值', color='blue')
    plt.plot(forecast_series, label='预测值', color='red', linestyle='--')
    plt.title(f"{method} 模型预测结果")
    plt.xlabel("时间")
    plt.ylabel(column)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    np.random.seed(42)
    # dataset_path = '../dataset/exchange_rate/test301-600.csv'
    # dataset_path = '../dataset/exchange_rate/test601-900.csv'
    # df = pd.read_csv(dataset_path)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    y1 = np.sin(np.arange(100) / 5) + np.random.normal(0, 0.3, 100)
    y2 = y1.copy() + 0.1
    y3 = y1.copy()
    df1 = pd.DataFrame({'date': dates, 'OT': y1})
    df2 = pd.DataFrame({'date': dates, 'OT': y2})
    df3 = pd.DataFrame({'date': dates, 'OT': y3})
    df3.iloc[56, 1] = 3
    # show_specify_line(df, ['value'], colour='red')
    show_change(df1, df2, 1)
    show_change(df1, df3, 1)