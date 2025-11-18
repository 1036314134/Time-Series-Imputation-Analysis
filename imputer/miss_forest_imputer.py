import time
import numpy as np
import pandas as pd
from missforest import MissForest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def miss_forest(incomp_data, n_estimators=10, max_iter=3, max_features='sqrt',
                          seed=42, logs=True, verbose=True):
    """
    Perform imputation using the MissForest algorithm (RandomForest-based imputation),
    while preserving the first column (timestamp) unmodified.

    Parameters
    ----------
    incomp_data : pandas.DataFrame
        The input DataFrame with missing values. The first column is assumed to be a timestamp.
    n_estimators : int, optional
        Number of trees in Random Forest models (default=10).
    max_iter : int, optional
        Maximum number of MissForest iterations (default=3).
    max_features : {'auto', 'sqrt', 'log2', float, int}, optional
        Number of features to consider when looking for best splits (default='sqrt').
    seed : int, optional
        Random seed (default=42).
    logs : bool, optional
        Whether to log timing info (default=True).
    verbose : bool, optional
        Whether to print details (default=True).

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the timestamp column preserved and other columns imputed.
    """

    # ===== 输入检查 =====
    if not isinstance(incomp_data, pd.DataFrame):
        incomp_data = pd.DataFrame(incomp_data)

    # 分离时间戳列
    timestamp_col = incomp_data.columns[0]
    timestamp = incomp_data.iloc[:, 0]
    feature_df = incomp_data.iloc[:, 1:]

    if verbose:
        print(f"(IMPUTATION) MISS FOREST with TIMESTAMP")
        print(f"\tMatrix shape (excluding timestamp): {feature_df.shape}")
        print(f"\tn_estimators: {n_estimators}, max_iter: {max_iter}, max_features: {max_features}, seed: {seed}\n")

    start_time = time.time()

    # ===== 构建随机森林模型 =====
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=seed,
        n_jobs=-1
    )
    rgr = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=seed,
        n_jobs=-1
    )

    # ===== 使用 MissForest 填补 =====
    mf_imputer = MissForest(clf=clf, rgr=rgr, max_iter=max_iter)
    recov_features = mf_imputer.fit_transform(feature_df)

    # 恢复为 DataFrame
    recov_features_df = pd.DataFrame(recov_features, columns=feature_df.columns, index=feature_df.index)

    # 合并时间戳与填补结果
    recov_df = pd.concat([timestamp, recov_features_df], axis=1)

    if logs and verbose:
        elapsed = time.time() - start_time
        print(f"> logs: imputation MISS FOREST - Execution Time: {elapsed:.4f} seconds\n")

    return recov_df


def miss_forest_with_timestamp(df, n_estimators=10, max_iter=3, max_features='sqrt',
                     seed=42, logs=True, verbose=True):
    """
    Random Forest (MissForest) imputation with automatic timestamp feature engineering.

    Input:
        df : DataFrame (first column must be timestamp)
    Output:
        DataFrame with missing values imputed, timestamp preserved.
    """
    # ------------------------ #
    # 1. 拷贝数据，避免原始数据被修改
    # ------------------------ #
    df = df.copy()

    # ------------------------ #
    # 2. 识别时间戳列（默认第一列）
    # ------------------------ #
    ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col])

    # 保存时间戳（不让它参与 MissForest）
    timestamp = df[ts_col]

    # ------------------------ #
    # 3. 时间特征工程（让随机森林能学习时间模式）
    # ------------------------ #
    df["timestamp_int"] = df[ts_col].astype("int64") // 10 ** 9  # 秒级整数时间
    df["hour"] = df[ts_col].dt.hour
    df["weekday"] = df[ts_col].dt.weekday
    df["dayofyear"] = df[ts_col].dt.dayofyear
    df["month"] = df[ts_col].dt.month
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["period_idx"] = np.arange(len(df))  # 时间序列步位置

    # 用于 MissForest 的数据（去掉原始 timestamp）
    df_for_impute = df.drop(columns=[ts_col])

    # ------------------------ #
    # 4. MissForest 随机森林填补（你之前的版本整合）
    # ------------------------ #
    if verbose:
        print(f"(IMPUTATION) MISS FOREST")
        print(f"\tMatrix: {df_for_impute.shape[0]} rows, {df_for_impute.shape[1]} cols")
        print(f"\tn_estimators: {n_estimators}")
        print(f"\tmax_iter: {max_iter}")
        print(f"\tmax_features: {max_features}")
        print(f"\tseed: {seed}\n")

    start_time = time.time()

    # 自定义 RF 模型
    clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, random_state=seed)
    rgr = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, random_state=seed)

    # 调用 MissForest
    mf_imputer = MissForest(clf=clf, rgr=rgr, max_iter=max_iter)
    recov_data = mf_imputer.fit_transform(df_for_impute)

    # 变成 DataFrame（保持列顺序）
    recov_df = pd.DataFrame(recov_data, columns=df_for_impute.columns)

    end_time = time.time()

    if logs and verbose:
        print(f"> logs: MissForest Execution Time: {end_time - start_time:.4f} seconds")

    # ------------------------ #
    # 5. 恢复原始 timestamp，去掉时间特征
    # ------------------------ #
    recov_df.insert(0, ts_col, timestamp)

    # 最后保留：timestamp + 原始数值列
    orig_cols = [ts_col] + [c for c in df.columns if c not in
                            ["timestamp_int", "hour", "weekday", "dayofyear",
                             "month", "is_weekend", "period_idx", ts_col]]

    return recov_df[orig_cols]
