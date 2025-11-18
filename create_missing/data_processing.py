import pandas as pd
import numpy as np
import random


def introduce_missing_segments(
        df,
        target_missing_ratio,
        column,
        min_seg_len=1,
        max_seg_len=100,
        start=0,
        end=-1,
        random_seed=None
):
    """
    在已有缺失值的基础上，为 DataFrame 的指定列增加连续缺失段，
    直到该列的缺失比例达到 target_missing_ratio，
    同时不改动该列最后 protect_last_n 个元素。

    参数：
    ----------
    df : pd.DataFrame
        输入的 DataFrame
    column : str
        要操作的列名
    target_missing_ratio : float
        目标缺失比例（0~1之间）
    min_seg_len : int
        每段连续缺失的最小长度
    max_seg_len : int
        每段连续缺失的最大长度
    start : int
        注入缺失起始位置
    end : int
        注入缺失结束位置
    random_seed : int or None
        随机种子，保证可重复性

    返回：
    ----------
    pd.DataFrame : 修改后的 DataFrame 副本
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    df_new = df.copy()

    # 仅在这部分数据上操作
    df_working = df_new.iloc[start:end]

    # 当前已有缺失数量（仅统计工作区）
    current_missing_count = df_working[column].isna().sum()
    target_missing_count = int(len(df_working) * target_missing_ratio)

    # 若当前缺失已达到或超过目标，则不操作
    if current_missing_count >= target_missing_count:
        print("The current missing percentage has reached or exceeded the target percentage, so no addition is needed.")
        return df_new

    # 计算需要新增的缺失数量
    to_add_missing = target_missing_count - current_missing_count

    # 找出可供制造缺失的索引（非缺失、非保护区域）
    valid_indices = df_working.index[~df_working[column].isna()].to_list()
    added_missing = 0
    new_missing_indices = set()

    while added_missing < to_add_missing and valid_indices:
        # 随机选择起始索引
        start_idx = random.choice(valid_indices)
        # 随机选择缺失段长度
        seg_len = random.randint(min_seg_len, max_seg_len)

        # 连续索引段
        seg_indices = [i for i in range(start_idx, start_idx + seg_len)
                       if i in df_working.index and pd.notna(df_new.loc[i, column])]

        if not seg_indices:
            continue

        # 防止超过目标
        if added_missing + len(seg_indices) > to_add_missing:
            seg_indices = seg_indices[:to_add_missing - added_missing]

        new_missing_indices.update(seg_indices)
        added_missing += len(seg_indices)

        # 更新可用索引
        valid_indices = [i for i in valid_indices if i not in new_missing_indices]

    # 应用缺失（仅工作区）
    df_new.loc[list(new_missing_indices), column] = np.nan

    return df_new