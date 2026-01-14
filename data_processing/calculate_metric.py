import pandas as pd
import numpy as np

from scipy.signal import periodogram
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL

# -------------------------------
# --- 读取并准备数据 -------
# -------------------------------

# 假设 df 是你的 DataFrame，每行是时间戳，每列是一个序列
df = pd.read_csv('your_file.csv', parse_dates=True, index_col=0)

# 提取 OT 时间序列
ot = df.iloc[:, -1].dropna()  # 最后一列为 OT

# -------------------------------
# --- 自相关 & 偏自相关 ---
# -------------------------------

# 自相关系数（ACF）
# nlags 控制最多计算多少 lag 的自相关
nlags = 40
acf_vals = acf(ot, nlags=nlags, fft=True)
print("ACF values:", acf_vals)

# 偏自相关系数（PACF）
pacf_vals = pacf(ot, nlags=nlags)
print("PACF values:", pacf_vals)

# -------------------------------
# --- Ljung-Box 检验统计量 ---
# -------------------------------

# 对多个滞后进行 Ljung-Box 检验
lb_test = acorr_ljungbox(ot, lags=[10, 20, 30], return_df=True)
print("Ljung-Box Test:\n", lb_test)

# 若需要分别提取统计量和 p-value：
# lb_stat = lb_test['lb_stat']
# lb_pval = lb_test['lb_pvalue']

# -------------------------------
# --- 频谱 / 功率谱密度 ---
# -------------------------------

# 计算 periodogram
freqs, pow_spec = periodogram(ot.values)
print("Spectrum frequencies:", freqs[:10])
print("Spectrum power:", pow_spec[:10])

# 可以用 matplotlib 绘图（若需要）
import matplotlib.pyplot as plt
plt.figure()
plt.semilogy(freqs, pow_spec)
plt.title("Power Spectrum (OT)")
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.show()

# -------------------------------
# --- STL 分解 + 趋势 & 季节性强度 ---
# -------------------------------

# 注意：必须指定数据的频率 period（例如 7, 12, 24 等）
# （你可以根据你数据的时间戳推断 period，也可以手动指定）
# 这里假设 daily 数据的一年季节性 period = 365
period = 12  # 例如季节性周期 (月度数据为 12), 你要根据具体数据调整
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

# -------------------------------
# --- 循环性强度（可选） ---
# -------------------------------

# 这里我们用频谱峰值作为循环性的一种衡量：
# 取功率谱中的最大能量比值
dominant_power = pow_spec.max()
total_power = pow_spec.sum()
cycle_strength = dominant_power / total_power if total_power > 0 else np.nan
print("Cycle strength (dominant / total):", cycle_strength)

# -------------------------------
# --- 输出结构总结 ---
# -------------------------------

result = {
    'acf': acf_vals,
    'pacf': pacf_vals,
    'ljung_box': lb_test,
    'spectrum': (freqs, pow_spec),
    'trend_strength': trend_strength,
    'seasonal_strength': seasonal_strength,
    'cycle_strength': cycle_strength
}

print("Final results keys:", result.keys())
