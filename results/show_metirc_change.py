import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import json


def dict_to_csv(data: dict, filepath: str):
    """
    将字典写入 CSV 文件
    每一行：key, value(json)
    """
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for k, v in data.items():
            writer.writerow([k, json.dumps(v, ensure_ascii=False)])


def csv_to_dict(filepath: str) -> dict:
    """
    从 CSV 文件读取并还原字典
    """
    data = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["key"]] = json.loads(row["value"])
    return data


def show_metirc_change(metrics, xlabel_name, ylabel_name, label):
    # ======================
    # 1. 横坐标：缺失率
    # ======================
    missing_rates = np.arange(0.0, 0.55, 0.05)

    # ======================
    # 2. 各算法指标（示例数据，替换为你的）
    # ======================

    # ======================
    # 3. 线型 & 点型（不重复）
    # ======================
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', 'P']

    # ======================
    # 4. 画图（关键参数都在这里）
    # ======================
    plt.figure(figsize=(10, 8))  # 稍大画布，缩小后仍清晰

    for i, (algo_name, values) in enumerate(metrics.items()):
        plt.plot(
            missing_rates,
            values,
            linestyle=line_styles[i % len(line_styles)],
            marker=markers[i % len(markers)],
            linewidth=3.5,  # ❗ 更粗的折线
            markersize=12,  # ❗ 更大的点
            markeredgewidth=1.2,  # 点边框更清楚
            label=algo_name
        )

    # ======================
    # 5. 坐标轴 & 标题（全部加大字体）
    # ======================
    plt.xlabel(xlabel_name, fontsize=24)
    plt.ylabel(ylabel_name, fontsize=24)
    plt.title(label, fontsize=28)

    plt.xticks(missing_rates, fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(fontsize=20)

    # ❌ 不加网格线，保持干净
    plt.tight_layout()
    plt.show()