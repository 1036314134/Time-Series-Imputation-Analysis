import matplotlib.pyplot as plt

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
        plt.legend()
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
        plt.plot(df_cleaned.index, df_cleaned[column], label='imputed data', linestyle='-', color='blue')
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

    




