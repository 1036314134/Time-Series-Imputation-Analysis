import pandas as pd

from create_missing.data_processing import introduce_missing_segments
from dataset.draw import show_specify_line, show_change
from imputer.front_imputer import front_impute
from imputer.iim_imputer import iim_impute
from imputer.knn_imputer import knn_impute
from imputer.mean_imputer import mean_impute
from imputer.miss_forest_imputer import miss_forest
from imputer.xgboost_imputer import xgboost_impute


def use_missing_creator(df_last, missing_rate, missing_columns, if_figure, if_write=True):
    # 排除时间戳
    features = df_last.shape[1] - 1

    # 根据制造缺失的属性数量确定文件名字
    if len(missing_columns) == features:
        write_name = "all"
    elif len(missing_columns) == 1:
        write_name = missing_columns[0]
    else:
        write_name = "covariate"

    # 确认是否将缺失值写入文件
    if if_write:
        df_missing = df_last.copy(deep=True)
        for column in missing_columns:
            # 为每个属性写入同样的注入方式
            seed = seek_random_seed(column)
            # 写入错误
            df_missing = introduce_missing_segments(df_missing, missing_rate, column, random_seed=seed)
        df_missing.to_csv(dataset_path[:-4] + "_missing_" + write_name + "_" + str(missing_rate) + ".csv", index=False)
    # 从文件中读取数据
    else:
        df_missing = pd.read_csv(dataset_path[:-4] + "_missing_" + write_name + "_" + str(missing_rate) + ".csv")

    # 是否展示图片
    if if_figure:
        show_specify_line(df_missing, missing_columns, colour='red')

    return df_missing


def use_imputer(df_missing, missing_rate, missing_column, imputer_name, if_write=True):
    if if_write:
        if imputer_name == "mean":
            df_imputed = mean_impute(df_missing)
        elif imputer_name == "front":
            df_imputed = front_impute(df_missing)
        elif imputer_name == "knn":
            df_imputed = knn_impute(df_missing)
        elif imputer_name == "xgboost":
            df_imputed = xgboost_impute(df_missing)
        elif imputer_name == "miss_forest":
            df_imputed = miss_forest(df_missing)
        elif imputer_name == "iim":
            df_imputed = iim_impute(df_missing)
        else:
            print("No impute algorithm was used.")
            df_imputed = df_missing
        df_imputed.to_csv(dataset_path[:-4] + "_missing_" + missing_column + "_" + str(missing_rate) + "_imputed_by_" + imputer_name + ".csv", index=False)
    else:
        df_imputed = pd.read_csv(dataset_path[:-4] + "_missing_" + missing_column + "_" + str(missing_rate) + "_imputed_by_" + imputer_name + ".csv")

    return df_imputed


def seek_random_seed(column):
    if column == "1":
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
    elif column == "OT":
        return 70
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
        df_missing = use_missing_creator(df_last, missing_rate, ["OT"], False, if_write=False)
        # df_missing = use_missing_creator(df_last, missing_rate, ["0", "1", "2", "3", "4", "5", "6"], False, if_write=False)
        # df_missing = use_missing_creator(df_last, missing_rate, ["0", "1", "2", "3", "4", "5", "6", "OT"], False, if_write=False)
        df_last = df_missing

        # use imputer
        methods = ["mean", "front", "knn", "xgboost", "miss_forest", "iim"]
        # methods = ["iim"]
        for method in methods:
            df_imputed = use_imputer(df_missing, missing_rate, "OT", method, if_write=False)
            # df_imputed = use_imputer(df_missing, missing_rate, "covariate", method, if_write=False)
            # df_imputed = use_imputer(df_missing, missing_rate, "all", method, if_write=False)
            show_change(df_origin, df_imputed, method)