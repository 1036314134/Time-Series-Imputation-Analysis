import numpy as np
import pandas as pd


class My_Data:
    def __init__(self, df):
        self.raw_data_df = df.copy(deep=True)
        self.timestamp = self.raw_data_df.iloc[:, 0]
        self.raw_feature_df = self.raw_data_df.iloc[:, 1:]
        self.raw_feature_array = np.array(self.raw_feature_df)
        self.imputed_feature_array = None
        self.imputed_feature_df = None
        self.imputed_data_df = None


    def change_feature_to_df(self):
        if self.imputed_feature_array is not None:
            self.imputed_feature_df = pd.DataFrame(self.imputed_feature_array, columns=self.raw_feature_df.columns, index=self.raw_feature_df.index)
        self.imputed_data_df = pd.concat([self.timestamp, self.imputed_feature_df], axis=1)