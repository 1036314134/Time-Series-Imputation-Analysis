import pandas as pd
import numpy as np

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import acf, pacf
from scipy.signal import periodogram
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
warnings.simplefilter("ignore", InterpolationWarning)
from scipy.stats import entropy, wasserstein_distance, ks_2samp
from sklearn.metrics import mutual_info_score



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



def compute_ot_mae_mse(df_origin, df_imputed, col_name="OT"):
    """
    计算真实数据与填补数据中 OT 序列的 MAE 和 MSE

    Parameters
    ----------
    df_origin : pd.DataFrame
        真实数据
    df_imputed : pd.DataFrame
        填补后数据
    col_name : str
        需要对比的列名，默认 'OT'

    Returns
    -------
    mae : float
    mse : float
    """
    y_true = df_origin[col_name].values
    y_pred = df_imputed[col_name].values

    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    print("MAE:", mae)
    print("MSE:", mse)

    # return mae, mse


def evaluate_ot_similarity(
    df_true,
    df_imputed,
    col_name="OT",
    n_bins=50,
    max_lag=20
):
    """
    计算真实 OT 序列与填补 OT 序列之间的多种差异指标

    Parameters
    ----------
    df_true : pd.DataFrame
        真实数据
    df_imputed : pd.DataFrame
        填补后数据
    col_name : str
        对比的列名（默认 OT）
    n_bins : int
        估计分布时使用的直方图 bin 数
    max_lag : int
        ACF / PACF 的最大滞后阶数

    Returns
    -------
    metrics : dict
        各类指标组成的字典
    """

    # =========================
    # 1. 数据对齐 & 清洗
    # =========================
    aligned = pd.concat(
        [df_true[col_name], df_imputed[col_name]],
        axis=1,
        keys=["true", "imputed"]
    ).dropna()

    if len(aligned) == 0:
        raise ValueError("对齐后没有可用于计算的 OT 样本")

    x = aligned["true"].values
    y = aligned["imputed"].values

    metrics = {}

    # =========================
    # 2. 点对点误差指标
    # =========================
    diff = x - y
    metrics["MAE"] = np.mean(np.abs(diff))
    metrics["MSE"] = np.mean(diff ** 2)

    # =========================
    # 3. 分布差异指标
    # =========================
    # 统一 bin（非常重要）
    hist_range = (min(x.min(), y.min()), max(x.max(), y.max()))

    px, _ = np.histogram(x, bins=n_bins, range=hist_range, density=True)
    py, _ = np.histogram(y, bins=n_bins, range=hist_range, density=True)

    # 避免 log(0)
    eps = 1e-10
    px += eps
    py += eps

    # KL 散度
    metrics["KL"] = entropy(px, py)

    # JS 散度
    m = 0.5 * (px + py)
    metrics["JS"] = 0.5 * entropy(px, m) + 0.5 * entropy(py, m)

    # Wasserstein 距离
    metrics["Wasserstein"] = wasserstein_distance(x, y)

    # Kolmogorov–Smirnov 距离
    ks_stat, _ = ks_2samp(x, y)
    metrics["KS"] = ks_stat

    # =========================
    # 4. 时序结构差异指标
    # =========================
    # ACF
    acf_x = acf(x, nlags=max_lag, fft=True)
    acf_y = acf(y, nlags=max_lag, fft=True)
    metrics["ACF_MSE"] = np.mean((acf_x - acf_y) ** 2)

    # PACF
    pacf_x = pacf(x, nlags=max_lag, method="yw")
    pacf_y = pacf(y, nlags=max_lag, method="yw")
    metrics["PACF_MSE"] = np.mean((pacf_x - pacf_y) ** 2)

    # =========================
    # 5. 互信息（离散化后）
    # =========================
    x_disc = np.digitize(x, bins=np.histogram_bin_edges(x, bins=n_bins))
    y_disc = np.digitize(y, bins=np.histogram_bin_edges(y, bins=n_bins))

    metrics["Mutual_Information"] = mutual_info_score(x_disc, y_disc)

    return metrics
