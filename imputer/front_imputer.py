import pandas as pd

def front_impute(missing_set):
    imputed_set = missing_set.copy()
    n_features = imputed_set.shape[1] - 1

    for i in range(n_features):
        imputed_set.iloc[:, i + 1] = imputed_set.iloc[:, i + 1].ffill()

    return imputed_set