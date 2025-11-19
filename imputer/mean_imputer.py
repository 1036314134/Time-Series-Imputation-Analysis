def mean_impute(raw_feature_df):
    imputed_feature_df = raw_feature_df.copy()
    n_features = imputed_feature_df.shape[1]

    for i in range(n_features):
        impute_value = imputed_feature_df.iloc[:, i].mean()
        imputed_feature_df.iloc[:, i] = imputed_feature_df.iloc[:, i].fillna(impute_value)

    return imputed_feature_df