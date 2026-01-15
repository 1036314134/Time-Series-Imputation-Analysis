def delete_missing(raw_feature_df):
    imputed_feature_df = raw_feature_df.copy()
    imputed_feature_df = imputed_feature_df.dropna(axis=0, how='any')
    return imputed_feature_df