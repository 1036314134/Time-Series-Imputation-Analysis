import numpy as np
import pandas as pd
from create_missing.data_processing import introduce_missing_segments
from imputer.trmf_imputer import trmf_impute

if __name__ == "__main__":
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    y = np.sin(np.arange(100) / 5) + np.random.normal(0, 0.3, 100)
    df = pd.DataFrame({'date': dates, 'OT': y})
    df_missing = introduce_missing_segments(df, 0.5, 'OT', random_seed=5)
    print(df_missing)
    df_imputed = trmf_impute(df_missing)
    print(df_imputed)
