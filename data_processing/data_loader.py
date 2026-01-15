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
        self.normalizer_tpye = None
        self.normalization_params = None


    def change_feature_to_df(self):
        if self.imputed_feature_array is not None:
            self.imputed_feature_df = pd.DataFrame(self.imputed_feature_array, columns=self.raw_feature_df.columns, index=self.raw_feature_df.index)
        self.imputed_data_df = pd.concat([self.timestamp, self.imputed_feature_df], axis=1)
        self.imputed_data_df = self.imputed_data_df.dropna(axis=0, how='any').reset_index(drop=True)


    def normalize(self, normalizer="z_score"):
        self.normalizer_type = normalizer  # 保存归一化方法
        self.normalization_params = {}  # 保存归一化参数

        if normalizer == "min_max":
            # Compute the min and max for each series (column-wise), ignoring NaN
            ts_min = np.nanmin(self.raw_feature_array, axis=0)
            ts_max = np.nanmax(self.raw_feature_array, axis=0)
            # Compute the range for each series, and handle cases where the range is 0
            range_ts = ts_max - ts_min
            range_ts[range_ts == 0] = 1  # Prevent division by zero for constant series
            # Apply min-max normalization
            self.raw_feature_array = (self.raw_feature_array - ts_min) / range_ts
            # 保存参数用于反归一化
            self.normalization_params['min'] = ts_min
            self.normalization_params['max'] = ts_max
            self.normalization_params['range'] = range_ts
        elif normalizer == "z_lib":
            from scipy.stats import zscore
            self.raw_feature_array = zscore(self.raw_feature_array, axis=0)
        elif normalizer == "m_lib":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            self.raw_feature_array = scaler.fit_transform(self.raw_feature_array)
            # 保存缩放器用于反归一化
            self.normalization_params['scaler'] = scaler
        else:
            mean = np.nanmean(self.raw_feature_array, axis=0)
            std_dev = np.nanstd(self.raw_feature_array, axis=0)
            # Avoid division by zero: set std_dev to 1 where it is zero
            std_dev[std_dev == 0] = 1
            # Apply z-score normalization
            self.raw_feature_array = (self.raw_feature_array - mean) / std_dev
            # 保存参数用于反归一化
            self.normalization_params['mean'] = mean
            self.normalization_params['std'] = std_dev

    def denormalize(self):
        if not hasattr(self, 'normalizer_type') or not hasattr(self, 'normalization_params'):
            raise ValueError("Normalization parameters not found. Please normalize the data first.")
        if self.normalizer_type == "min_max":
            ts_min = self.normalization_params['min']
            range_ts = self.normalization_params['range']
            # 反向 min-max 归一化: X = X_norm * range + min
            self.imputed_feature_array = self.imputed_feature_array * range_ts + ts_min
        elif self.normalizer_type == "z_lib":
            # scipy.stats.zscore 的反向操作需要原始的均值和标准差
            # 如果没有保存，这里无法完全恢复（zscore 本身不保存参数）
            if 'mean' in self.normalization_params and 'std' in self.normalization_params:
                mean = self.normalization_params['mean']
                std_dev = self.normalization_params['std']
                self.imputed_feature_array = self.imputed_feature_array * std_dev + mean
            else:
                print("Warning: z_lib normalization parameters not saved. Cannot denormalize accurately.")
        elif self.normalizer_type == "m_lib":
            scaler = self.normalization_params['scaler']
            # 使用 sklearn 的 inverse_transform 方法
            self.imputed_feature_array = scaler.inverse_transform(self.imputed_feature_array)
        else:  # z_score (default)
            mean = self.normalization_params['mean']
            std_dev = self.normalization_params['std']
            # 反向 z-score 归一化: X = X_norm * std + mean
            self.imputed_feature_array = self.imputed_feature_array * std_dev + mean