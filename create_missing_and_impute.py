import pandas as pd
from dataset.draw import show_specify_line, show_change


def use_missing_creator(df_last, missing_rate, missing_columns, if_figure=False, if_write=True):
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
            from create_missing.data_processing import introduce_missing_segments
            df_missing = introduce_missing_segments(df_missing, missing_rate, column, end=slice_3[0]-1, random_seed=seed)
        df_missing.to_csv(dataset_path[:-4] + "_missing_" + write_name + "_" + str(missing_rate) + ".csv", index=False)
    # If no data is written, then data is read from the file.
    else:
        df_missing = pd.read_csv(dataset_path[:-4] + "_missing_" + write_name + "_" + str(missing_rate) + ".csv")

    # whether to display the attribute sequence
    if if_figure:
        show_specify_line(df_missing, missing_columns, colour='red')

    return df_missing


def use_imputer(df_missing, missing_rate, missing_column, imputer_name, if_write=True):
    if if_write:
        if imputer_name == "mean":
            from imputer.mean_imputer import mean_impute
            df_imputed = mean_impute(df_missing)
        elif imputer_name == "front":
            from imputer.front_imputer import front_impute
            df_imputed = front_impute(df_missing)
        elif imputer_name == "knn":
            from imputer.knn_imputer import knn_impute
            df_imputed = knn_impute(df_missing)
        elif imputer_name == "xgboost":
            from imputer.xgboost_imputer import xgboost_impute
            df_imputed = xgboost_impute(df_missing)
        elif imputer_name == "miss_forest":
            from imputer.miss_forest_imputer import miss_forest
            df_imputed = miss_forest(df_missing)
        elif imputer_name == "iim":
            from imputer.iim_imputer import iim_impute
            df_imputed = iim_impute(df_missing)
        else:
            print("No impute algorithm was used.")
            df_imputed = df_missing
        df_imputed.to_csv(dataset_path[:-4] + "_missing_" + missing_column + "_" + str(missing_rate) + "_imputed_by_" + imputer_name + ".csv", index=False)
    else:
        df_imputed = pd.read_csv(dataset_path[:-4] + "_missing_" + missing_column + "_" + str(missing_rate) + "_imputed_by_" + imputer_name + ".csv")

    return df_imputed


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
    df_origin = pd.read_csv(dataset_path)
    # show_specify_line(df_origin, ['OT'])
    df_last = df_origin

    # missing_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    missing_rate_list = [0.5]
    for  missing_rate in missing_rate_list:
        print(missing_rate)
        # create missing values
        df_missing = use_missing_creator(df_last, missing_rate, ["OT"], if_figure=False, if_write=False)
        # df_missing = use_missing_creator(df_last, missing_rate, ["0", "1", "2", "3", "4", "5", "6"], if_figure=False, if_write=False)
        # df_missing = use_missing_creator(df_last, missing_rate, ["0", "1", "2", "3", "4", "5", "6", "OT"], if_figure=False, if_write=False)
        df_last = df_missing

        # use imputer
        # methods = ["mean", "front", "knn", "xgboost", "miss_forest", "iim"]
        methods = ["iim"]
        for method in methods:
            df_imputed = use_imputer(df_missing, missing_rate, "OT", method, if_write=True)
            # df_imputed = use_imputer(df_missing, missing_rate, "covariate", method, if_write=False)
            # df_imputed = use_imputer(df_missing, missing_rate, "all", method, if_write=False)
            show_change(df_origin, df_imputed, method)