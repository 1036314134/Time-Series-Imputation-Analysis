import pandas as pd

def mean_impute(missing_set: pd.DataFrame) -> pd.DataFrame:
    imputed_set = missing_set.copy(deep=True)
    n_features = imputed_set.shape[1] - 1

    for i in range(n_features):
        impute_value = imputed_set.iloc[:, i + 1].mean()
        imputed_set.iloc[:, i + 1] = imputed_set.iloc[:, i + 1].fillna(impute_value)

    return imputed_set