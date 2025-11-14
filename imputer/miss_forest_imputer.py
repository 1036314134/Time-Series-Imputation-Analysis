import time
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