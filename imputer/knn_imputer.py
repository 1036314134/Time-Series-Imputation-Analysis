import pandas as pd
from sklearn.impute import KNNImputer

def knn_impute(df_missing, n_neighbors=5):
    """
    使用KNN对DataFrame中的缺失值进行填补。
    参数
    ----
    df : pd.DataFrame
        输入包含缺失值的数据，第一列为时间戳，其余列为数值特征。
    n_neighbors : int, 默认=5
        KNN的邻居数量。
    返回
    ----
    pd.DataFrame
        填补后的DataFrame，保留时间戳和原始列名。
    """
    # 拷贝避免修改原始数据
    df_copy = df_missing.copy(deep=True)

    # 分离时间戳列
    timestamp_col = df_copy.iloc[:, 0]
    features = df_copy.iloc[:, 1:]

    # 使用KNNImputer填补缺失值
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(features)

    # 转回DataFrame并恢复列名
    imputed_df = pd.DataFrame(imputed_array, columns=features.columns, index=df_copy.index)

    # 合并回时间戳列
    result_df = pd.concat([timestamp_col, imputed_df], axis=1)

    return result_df