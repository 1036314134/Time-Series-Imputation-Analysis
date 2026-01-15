import pandas as pd
import numpy as np

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import acf
from scipy.signal import periodogram
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
warnings.simplefilter("ignore", InterpolationWarning)


def hurst_exponent_rs(ts, min_lag=10, max_lag=None):
    """
    计算时间序列的 Hurst 指数（基于 R/S 分析），并避免 log(0) 等问题。

    参数:
        ts (array-like): 目标时间序列
        min_lag (int): 最小子区间大小，通常 >= 10
        max_lag (int): 最大子区间大小，默认为 len(ts)//2

    返回:
        H (float): Hurst 指数
    """
    ts = np.asarray(ts)
    n = len(ts)

    # 必须有足够长度
    if n < min_lag * 2:
        return np.nan

    if max_lag is None:
        max_lag = n // 2  # 避免子区间过长

    # 选取不同子区间大小
    lags = np.unique(np.floor(np.linspace(min_lag, max_lag, 20)).astype(int))
    rs_vals = []

    for lag in lags:
        # 划分若干子区间
        segments = n // lag
        if segments < 1:
            continue

        rs_segments = []

        for i in range(segments):
            subseries = ts[i*lag:(i+1)*lag]
            mean = np.mean(subseries)
            dev = subseries - mean

            R = np.max(dev) - np.min(dev)
            S = np.std(subseries)

            # 忽略 S == 0 的区间
            if S > 0:
                rs_segments.append(R / S)

        # 如果没有有效子区间跳过
        if len(rs_segments) == 0:
            continue

        rs_vals.append(np.mean(rs_segments))

    rs_vals = np.array(rs_vals)
    lags_valid = lags[:len(rs_vals)]

    # 避免 log(0) 和无效值
    mask = (rs_vals > 0) & (lags_valid > 0)
    if np.sum(mask) < 2:
        return np.nan

    # 对数坐标拟合
    log_lags = np.log(lags_valid[mask])
    log_rs = np.log(rs_vals[mask])

    # 线性拟合
    slope, _ = np.polyfit(log_lags, log_rs, 1)

    # Hurst 指数
    H = slope
    return H


# 谱熵
def spectral_entropy(ts):
    freqs, psd = periodogram(ts)
    psd_norm = psd / np.sum(psd)
    psd_norm = psd_norm[np.nonzero(psd_norm)]
    return -np.sum(psd_norm * np.log(psd_norm))


def calculate_ts_metric(df, period = 12):
    # 提取 OT 时间序列
    ot = df.iloc[:, -1].dropna()  # 最后一列为 OT

    # 1. ACF at lag=1
    acf_vals = acf(ot, nlags=5, fft=True)
    acf_1 = acf_vals[1]
    print("ACF@lag1:", acf_1)

    # 2. 平稳性检验 (ADF, KPSS)
    adf_res = adfuller(ot)
    print("ADF statistic, p-value:", adf_res[0], adf_res[1])

    kpss_res = kpss(ot, regression='c')
    print("KPSS statistic, p-value:", kpss_res[0], kpss_res[1])

    # 3. Hurst 指数
    hurst_val = hurst_exponent_rs(ot)
    print("Hurst exponent:", hurst_val)

    # 4. 谱熵
    spec_entropy = spectral_entropy(ot)
    print("Spectral entropy:", spec_entropy)

    # 5. Trend/Seasonal strength
    stl = STL(ot, period=period, robust=True)
    res = stl.fit()

    # 趋势、季节性、残差
    trend = res.trend
    seasonal = res.seasonal
    resid = res.resid

    # 趋势强度 Ft
    var_resid = resid.var()
    var_trend_resid = (trend + resid).var()
    trend_strength = max(0.0, 1.0 - var_resid / var_trend_resid)
    print("Trend strength:", trend_strength)

    # 季节性强度 Fs
    var_seasonal_resid = (seasonal + resid).var()
    seasonal_strength = max(0.0, 1.0 - var_resid / var_seasonal_resid)
    print("Seasonal strength:", seasonal_strength)

    # 循环强度
    # 这里我们用频谱峰值作为循环性的一种衡量：
    # 取功率谱中的最大能量比值
    freqs, pow_spec = periodogram(ot.values)
    dominant_power = pow_spec.max()
    total_power = pow_spec.sum()
    cycle_strength = dominant_power / total_power if total_power > 0 else np.nan
    print("Cycle strength (dominant / total):", cycle_strength)

    # 6.Ljung - Box
    # 对多个滞后进行 Ljung-Box 检验
    lb_test = acorr_ljungbox(ot, lags=[5, 10, 20], return_df=True)
    print("Ljung-Box Test:\n", lb_test)

    # 若需要分别提取统计量和 p-value：
    # lb_stat = lb_test['lb_stat']
    # lb_pval = lb_test['lb_pvalue']
    #
    # print("Ljung-Box p-values:", lb_test['lb_pvalue'].values)
