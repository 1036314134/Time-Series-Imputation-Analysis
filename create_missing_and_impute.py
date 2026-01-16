import pandas as pd
from data_processing.calculate_metric import calculate_ts_metric
from data_processing.data_loader import My_Data
from dataset.draw import show_specify_line, show_change


def use_missing_creator(df_last,
                        missing_rate,
                        missing_columns,
                        if_figure=False,
                        if_write=True,
                        min_seg_len=1,
                        max_seg_len=100):
    """
    Inject missing values into complete data.

    param:
    ----------
    df_last: DataFrame
        Dataframes that need to be injected with missing values
    missing_rate: float
        Target proportion of missing data in the data
    missing_columns: list
        Create missing attribute name
    if_figure: boolean
        Control whether to display the attribute sequence after injection is missing
    if_write: boolean
        Control whether to write the results to a CSV file

    return:
    ----------
    df_missing: DataFrame
        Dataframes that have missing values
    """
    # Remove timestamps
    features = df_last.shape[1] - 1

    # The file name is determined based on the number of missing attributes in the manufacturing process
    if len(missing_columns) == features:
        write_name = "all"
    elif len(missing_columns) == 1:
        write_name = missing_columns[0]
    else:
        write_name = "covariate"

    # Calculate dataset size and partition position
    num_train = int(len(df_last) * 0.7) # Training set size
    num_test = int(len(df_last) * 0.2) # Validation set size
    num_val = len(df_last) - num_train - num_test # Test set size
    slice_1 = [0, num_train] # Training set range
    slice_2 = [num_train - 96, num_train + num_val] # Validation set range
    slice_3 = [len(df_last) - num_test - 96, len(df_last)] # Test set range

    # Confirm whether to write missing values to the file.
    if if_write:
        df_missing = df_last.copy(deep=True)
        for column in missing_columns:
            # Determine the random seed based on the attribute name.
            seed = seek_random_seed(column)
            # Inject error
            from data_processing.create_missing import introduce_missing_segments
            df_missing = introduce_missing_segments(df_missing, missing_rate, column, min_seg_len=min_seg_len,
                                                    max_seg_len=max_seg_len, end=slice_3[0]-1, random_seed=seed)
        df_missing.to_csv(dataset_path[:-4] + "_" + write_name + "_" + str(missing_rate) + ".csv", index=False)
    # If no data is written, then data is read from the file.
    else:
        df_missing = pd.read_csv(dataset_path[:-4] + "_" + write_name + "_" + str(missing_rate) + ".csv")

    # whether to display the attribute sequence
    if if_figure:
        show_specify_line(df_missing, missing_columns, colour='red')

    return df_missing


def use_imputer(data, missing_rate, missing_column, imputer_name, if_write=True):
    if if_write:
        if imputer_name == "delete":
            from imputer.delete import delete_missing
            data.imputed_feature_df = delete_missing(data.raw_feature_df)
        elif imputer_name == "mean":
            from imputer.mean_imputer import mean_impute
            data.imputed_feature_df = mean_impute(data.raw_feature_df)
        elif imputer_name == "front":
            from imputer.front_imputer import front_impute
            data.imputed_feature_df = front_impute(data.raw_feature_df)
        elif imputer_name == "knn":
            from imputer.knn_imputer import knn_impute
            data.imputed_feature_array = knn_impute(data.raw_feature_df)
        elif imputer_name == "xgboost":
            from imputer.xgboost_imputer import xgboost_impute
            data.imputed_feature_array = xgboost_impute(data.raw_feature_array)
        elif imputer_name == "miss_forest":
            from imputer.miss_forest_imputer import miss_forest_impute
            data.imputed_feature_array = miss_forest_impute(data.raw_feature_array)
        elif imputer_name == "iim":
            from imputer.iim_imputer import iim_impute
            data.normalize(normalizer="min_max")
            data.imputed_feature_array = iim_impute(data.raw_feature_array.T).T
            data.denormalize()
        elif imputer_name == "trmf":
            from imputer.trmf_imputer import trmf_impute
            data.normalize(normalizer="min_max")
            data.imputed_feature_array = trmf_impute(data.raw_feature_array.T).T
            data.denormalize()
        elif imputer_name == "miss_net":
            from imputer.miss_net_imputer import miss_net_impute
            data.imputed_feature_array = miss_net_impute(data.raw_feature_array.T).T
        else:
            print("No impute algorithm was used.")
            data.imputed_data_df = data.raw_data_df
        data.change_feature_to_df()
        data.imputed_data_df.to_csv(dataset_path[:-4] + "_" + missing_column + "_" + str(missing_rate) + "_" + imputer_name + ".csv", index=False)
    else:
        data.imputed_data_df = pd.read_csv(dataset_path[:-4] + "_" + missing_column + "_" + str(missing_rate) + "_" + imputer_name + ".csv")

    return data.imputed_data_df


def seek_random_seed(column):
    if column == "OT":
        return 5
    elif column == "1":
        return 10
    elif column == "2":
        return 20
    elif column == "3":
        return 30
    elif column == "4":
        return 40
    elif column == "5":
        return 50
    elif column == "6":
        return 60
    else:
        return 0


if __name__ == '__main__':
    dataset_path = 'dataset/exchange_rate/exchange_rate.csv'
    # dataset_path = 'dataset/weather/weather.csv'
    # dataset_path = 'dataset/ETT-small/ETTh1.csv'
    df_origin = pd.read_csv(dataset_path)
    # show_specify_line(df_origin, ['OT'], colour='green')
    # calculate_ts_metric(df_origin, period=365)
    df_last = df_origin

    # missing_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    missing_rate_list = [0.1]
    for  missing_rate in missing_rate_list:
        print(missing_rate)
        # ==== create missing values ====
        df_missing = use_missing_creator(df_last, missing_rate, ["OT"], if_figure=False, if_write=False,
                                         min_seg_len=1, max_seg_len=10)
        df_last = df_missing
        # calculate_ts_metric(df_missing, period=24)

        # ==== use imputer ====
        # methods = ["delete", "mean", "front", "knn", "xgboost", "miss_forest", "iim", "trmf"]
        # methods = ["mean", "front", "knn", "xgboost"]
        methods = ["mean", "front"]
        for method in methods:
            data = My_Data(df_missing)
            df_imputed = use_imputer(data, missing_rate, "OT", method, if_write=False)
            # show_specify_line(df_imputed, ['OT'], colour='#F3D266')
            show_change(df_origin, df_imputed, method)
            print(str(missing_rate) + "_" + method)
            calculate_ts_metric(df_imputed, period=365)

