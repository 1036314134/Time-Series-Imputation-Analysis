import pandas as pd
from xgboost import XGBRegressor


def xgboost_impute(df_missing, n_estimators=10, seed=42):
    # 拷贝避免修改原始数据
    df_copy = df_missing.copy(deep=True)

    # 分离时间戳列
    timestamp_col = df_copy.iloc[:, 0]
    features = df_copy.iloc[:, 1:]

    for column in features.columns:
        model = XGBRegressor(n_estimators=n_estimators, random_state=seed)

        non_missing = features.loc[df_missing[column].notna()]
        missing = features.loc[df_missing[column].isna()]

        x_train = non_missing.drop(columns=[column])
        y_train = non_missing[column]
        x_missing = missing.drop(columns=[column])

        # Fit the model
        model.fit(x_train, y_train)

        # Predict missing values
        predictions = model.predict(x_missing)

        # Assign the predicted values
        features.loc[features[column].isna(), column] = predictions

    # 合并回时间戳列
    result_df = pd.concat([timestamp_col, features], axis=1)

    return result_df
