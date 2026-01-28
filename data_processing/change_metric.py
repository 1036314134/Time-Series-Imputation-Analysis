import numpy as np
import pandas as pd
from data_processing.calculate_metric import compute_metrics, compare_ot_metrics, evaluate_ot_similarity
from dataset.draw import show_change
from statsmodels.tsa.seasonal import STL
from scipy.signal import periodogram


# def sample_contiguous_segments(
#     n,
#     ratio,
#     min_len,
#     max_len,
#     start_idx,
#     end_idx,
#     random_state
# ):
#     """
#     在指定范围内采样若干连续片段，使得总长度约为 ratio * (end_idx - start_idx)
#
#     Parameters
#     ----------
#     n : int
#         原始序列长度
#     ratio : float
#         修改比例（相对于可修改区间长度）
#     min_len : int
#         单个连续片段的最小长度
#     max_len : int
#         单个连续片段的最大长度
#     start_idx : int
#         允许修改的起始位置（包含）
#     end_idx : int or None
#         允许修改的终止位置（不包含），None 表示 n
#     random_state : int or None
#         随机种子
#
#     Returns
#     -------
#     indices : np.ndarray
#         排好序的待修改下标
#     """
#
#     if random_state is not None:
#         np.random.seed(random_state)
#
#     if end_idx is None:
#         end_idx = n
#
#     # 合法性检查
#     start_idx = max(0, start_idx)
#     end_idx = min(n, end_idx)
#
#     if start_idx >= end_idx:
#         raise ValueError("start_idx must be smaller than end_idx")
#
#     effective_length = end_idx - start_idx
#     target_total_len = int(np.round(ratio * effective_length))
#
#     if target_total_len <= 0:
#         return np.array([], dtype=int)
#
#     selected = set()
#     attempts = 0
#     max_attempts = 10 * target_total_len
#
#     while len(selected) < target_total_len and attempts < max_attempts:
#         attempts += 1
#
#         seg_len = np.random.randint(min_len, max_len + 1)
#
#         if seg_len > effective_length:
#             continue
#
#         seg_start = np.random.randint(start_idx, end_idx - seg_len + 1)
#         seg_indices = range(seg_start, seg_start + seg_len)
#
#         for idx in seg_indices:
#             if len(selected) < target_total_len:
#                 selected.add(idx)
#             else:
#                 break
#
#     return np.array(sorted(selected))
#
#
# def modify_ot_for_target_metric(
#     df,
#     target_metric,
#     ratio=0.1,
#     min_len=5,
#     max_len=50,
#     start_idx=0,
#     end_idx=None,
#     random_state=0
# ):
#     """
#     使用连续区间扰动，在尽量不影响其他指标的前提下，
#     定向改变某一个时间序列指标
#     """
#     new_df = df.copy()
#     ot = new_df.iloc[:, -1].values.copy()
#     n = len(ot)
#
#     # Calculate dataset size and partition position
#     if end_idx == None:
#         end_idx = int(len(ot) * 0.8) - 96  # Test set range
#
#     # 采样连续区间索引
#     idx = sample_contiguous_segments(
#         n=n,
#         ratio=ratio,
#         min_len=min_len,
#         max_len=max_len,
#         start_idx=start_idx,
#         end_idx=end_idx,
#         random_state=random_state
#     )
#
#     if len(idx) == 0:
#         return new_df
#
#     std = np.std(ot)
#     t = idx
#
#     # 针对不同指标的“定向扰动”
#     if target_metric == "acf1":
#         # 局部时间反转：破坏短期依赖
#         ot[t] = ot[t[::-1]]
#
#     elif target_metric == "trend":
#         # 局部低频漂移：改变趋势强度
#         drift = np.linspace(0, 0.5 * std, len(t))
#         ot[t] += drift
#
#     elif target_metric == "seasonal":
#         # 固定周期季节扰动
#         period = 24
#         amp = 0.3 * std
#         ot[t] += amp * np.sin(2 * np.pi * t / period)
#
#     elif target_metric == "cycle":
#         # 单一频率循环增强
#         freq = 1 / 50
#         amp = 0.5 * std
#         ot[t] += amp * np.sin(2 * np.pi * freq * t)
#
#     elif target_metric == "spectral_entropy":
#         # 高频噪声注入
#         noise = np.random.normal(0, 0.8 * std, size=len(t))
#         ot[t] += noise
#
#     elif target_metric == "hurst":
#         # 连续区间符号翻转（破坏长期相关）
#         ot[t] = -ot[t]
#
#     elif target_metric == "adf_kpss":
#         # 连续区间趋势漂移（增强非平稳性）
#         drift = np.linspace(0, std, len(t))
#         ot[t] += drift
#
#     elif target_metric == "ljungbox":
#         # 区间内反转（增强多阶扰动）
#         ot[t] = ot[t[::-1]]
#
#     else:
#         raise ValueError(f"Unsupported target metric: {target_metric}")
#
#     new_df.iloc[:, -1] = ot
#     return new_df


def _modify_trend(x, segment, strength, period=None):
    stl = STL(x, period=period, robust=True)
    res = stl.fit()

    trend = res.trend
    seasonal = res.seasonal
    resid = res.resid

    # def deform_trend_shape(trend, segment, strength):
    #     t = np.arange(len(trend))
    #     t_seg = t[segment]
    #
    #     # 低频 smooth oscillation
    #     freq = 1 / (len(t_seg) * 0.8)
    #     oscillation = np.sin(2 * np.pi * freq * t_seg)
    #
    #     # 强度控制
    #     amp = strength * np.std(trend[segment])
    #
    #     trend_new = trend.copy()
    #     trend_new[segment] += amp * oscillation
    #
    #     return trend_new


    # strength 控制趋势向“平坦”退化的程度
    trend_new = trend.copy()

    # trend_new = deform_trend_shape(trend_new, segment, strength)

    trend_mean = np.mean(trend[segment])
    trend_new[segment] = (
        (1 - strength) * trend[segment]
        + strength * trend_mean
    )

    x_new = trend_new + seasonal + resid
    return x_new


def _modify_seasonality(x, segment, strength, period):
    stl = STL(x, period=period, robust=True)
    res = stl.fit()
    seasonal = res.seasonal
    seasonal_new = seasonal.copy()
    seasonal_new[segment] *= (1 - strength)

    return res.trend + seasonal_new + res.resid


def _modify_cycle(x, segment, strength):
    seg = x[segment]
    freqs, power = periodogram(seg)

    main_freq = np.argmax(power[1:]) + 1

    fft = np.fft.rfft(seg)
    fft[main_freq] *= (1 - strength)

    seg_new = np.fft.irfft(fft, n=len(seg))

    x_new = x.copy()
    x_new[segment] = seg_new
    return x_new


def _modify_dependence(x, segment, strength):
    seg = x[segment]
    n = len(seg)

    block = max(5, int((1 - strength) * n))
    indices = np.arange(n)

    np.random.shuffle(indices)
    seg_new = seg[indices]

    x_new = x.copy()
    x_new[segment] = seg_new
    return x_new


def _modify_spectrum_entropy(x, segment, strength):
    seg = x[segment]
    noise = np.random.normal(
        0,
        strength * np.std(seg),
        size=len(seg)
    )
    x_new = x.copy()
    x_new[segment] = seg + noise
    return x_new


def modify_ot_by_metric(
    df,
    metric,
    start_idx=0,
    end_idx=None,
    strength=0.1,
    period: int | None = None,
):
    """
    df: 原始 dataframe
    metric: 目标指标名
    start_idx, end_idx: 修改区间
    strength: 指标改变强度（0~1）
    period: STL 周期（可选）
    """
    df_new = df.copy()
    ot = df["OT"].values

    if end_idx == None:
        end_idx = int(len(ot) * 0.8) - 96  # Test set range

    segment = slice(start_idx, end_idx)
    ot_seg = ot[segment]

    if metric in ["trend"]:
        ot_new = _modify_trend(ot, segment, strength, period)

    elif metric in ["seasonal"]:
        ot_new = _modify_seasonality(ot, segment, strength, period)

    elif metric in ["cycle"]:
        ot_new = _modify_cycle(ot, segment, strength)

    elif metric in ["acf", "ljung_box"]:
        ot_new = _modify_dependence(ot, segment, strength)

    elif metric in ["spectral_entropy"]:
        ot_new = _modify_spectrum_entropy(ot, segment, strength)

    else:
        raise ValueError(f"Unsupported metric: {metric}")

    df_new["OT"] = ot_new
    return df_new



if __name__ == '__main__':
    dataset_paths = ['None',
                     '../dataset/exchange_rate/exchange_rate.csv',
                     '../dataset/weather/weather.csv', #30
                     '../dataset/ETT-small/ETTh1.csv', #24
                     '../dataset/ETT-small/ETTh2.csv',
                     '../dataset/ETT-small/ETTm1.csv',
                     '../dataset/ETT-small/ETTm2.csv',
                     '../dataset/illness/national_illness.csv' #52
    ]

    dataset_path = dataset_paths[3]
    df_origin = pd.read_csv(dataset_path)


    # metrics = ["trend", "seasonal", "cycle", "acf", "spectral_entropy"]
    metrics = ["trend", "seasonal", "cycle"]
    for metric in metrics:

        change_strength_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        for change_strength in change_strength_list:

            df_change = modify_ot_by_metric(df_origin, metric=metric, strength=change_strength, period=24)
            df_change.to_csv(dataset_path[:-4] + "_" + metric + "_" + str(change_strength) + ".csv", index=False)

            print(metric + "_" + str(change_strength))
            compare_ot_metrics(df_before=df_origin, df_after=df_change, metric_func=compute_metrics, period=24)
            metrics = evaluate_ot_similarity(df_origin, df_change)
            print(metrics)
            # show_change(df_origin, df_change, metric)
