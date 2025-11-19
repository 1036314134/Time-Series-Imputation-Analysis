from sklearn.impute import KNNImputer


def knn_impute(raw_feature_df, n_neighbors=5):
    """
    Use KNN to impute missing values in the data.

    param:
    ----------
    raw_feature_df : numpy.array
        The input matrix with contamination (missing values represented as NaNs).
    n_neighbors : int, init=5
        The number of neighbors of KNN.

    return:
    ----------
    pd.DataFrame
        The imputed DataFrame retains the timestamps and original column names.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_feature_array = imputer.fit_transform(raw_feature_df)

    return imputed_feature_array