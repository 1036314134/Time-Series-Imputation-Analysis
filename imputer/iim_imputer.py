import time
import numpy as np
import pandas as pd
from imputer.imputegap.wrapper.AlgoPython.IIM.runnerIIM import impute_with_algorithm


def iim_impute(incomp_data, number_neighbor=5, algo_code="iim 1", logs=True, verbose=True):
    """
    Perform imputation using the Iterative Imputation Method (IIM) algorithm.

    Parameters
    ----------
    incomp_data : pandas.DataFrame
        The input DataFrame with missing values. The first column is assumed to be a timestamp.
    number_neighbor : int
        The number of neighbors to use for the K-Nearest Neighbors (KNN) classifier (default is 10).
    algo_code : str
        The specific action code for the IIM output. This determines the behavior of the algorithm.
    logs : bool, optional
        Whether to log the execution time (default is True).
    verbose : bool, optional
        Whether to display the contamination information (default is True).

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values recovered.

    Notes
    -----
    The IIM algorithm works by utilizing K-Nearest Neighbors (KNN) to estimate missing values in time series data.
    Depending on the provided `algo_code`, different versions of the algorithm may be executed.

    The function logs the total execution time if `logs` is set to True.

    References
    ----------
    A. Zhang, S. Song, Y. Sun and J. Wang, "Learning Individual Models for Imputation," 2019 IEEE 35th International Conference on Data Engineering (ICDE), Macao, China, 2019, pp. 160-171, doi: 10.1109/ICDE.2019.00023.
    keywords: {Data models;Adaptation models;Computational modeling;Predictive models;Numerical models;Aggregates;Regression tree analysis;Missing values;Data imputation}
    """
    # 分离时间戳列
    timestamp = incomp_data.iloc[:, 0]
    feature_df = incomp_data.iloc[:, 1:]
    # 转变为numpy
    feature_np = np.array(feature_df)

    start_time = time.time()

    recov_features = impute_with_algorithm(algo_code, feature_np, number_neighbor, verbose=verbose)

    # 恢复为 DataFrame
    recov_features_df = pd.DataFrame(recov_features, columns=feature_df.columns, index=feature_df.index)

    # 合并时间戳与填补结果
    recov_df = pd.concat([timestamp, recov_features_df], axis=1)

    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation iim - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_df
