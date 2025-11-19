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
    Adds consecutive missing segments to a specified column and row of a DataFrame, building upon existing missing values,
    until the missing percentage of that column reaches `target_missing_ratio`.

    param:
    ----------

    df: DataFrame
        The input DataFrame
    column: str
        The name of the column to operate on
    target_missing_ratio: float
        The target missing percentage (between 0 and 1)
    min_seg_len: int
        The minimum length of each consecutive missing segment
    max_seg_len: int
        The maximum length of each consecutive missing segment
    start: int
        The starting position of the missing segment injection
    end: int
        The ending position of the missing segment injection
    random_seed : int or None
        A random seed to ensure repeatability

    Return:
    ----------
    df_new: DataFrame
        A copy of the modified DataFrame
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    df_new = df.copy()

    # Determine the injection error location
    df_working = df_new.iloc[start:end]

    # Current number of missing items (workspace only)
    current_missing_count = df_working[column].isna().sum()
    target_missing_count = int(len(df_working) * target_missing_ratio)

    # If the current missing value has reached or exceeded the target, no action will be taken.
    if current_missing_count >= target_missing_count:
        print("The current missing percentage has reached or exceeded the target percentage, so no addition is needed.")
        return df_new

    # Calculate the number of missing values that need to be added.
    to_add_missing = target_missing_count - current_missing_count

    # Find the indexes that can be used to create a missing index (work area and not missing).
    valid_indices = df_working.index[~df_working[column].isna()].to_list()
    added_missing = 0
    new_missing_indices = set()

    while added_missing < to_add_missing and valid_indices:
        # Randomly select the starting index
        start_idx = random.choice(valid_indices)
        # Randomly select the length of the missing segment
        seg_len = random.randint(min_seg_len, max_seg_len)

        # Continuous index segment
        seg_indices = [i for i in range(start_idx, start_idx + seg_len)
                       if i in df_working.index and pd.notna(df_new.loc[i, column])]

        if not seg_indices:
            continue

        # Prevent exceeding the target
        if added_missing + len(seg_indices) > to_add_missing:
            seg_indices = seg_indices[:to_add_missing - added_missing]

        new_missing_indices.update(seg_indices)
        added_missing += len(seg_indices)

        # Update available indexes
        valid_indices = [i for i in valid_indices if i not in new_missing_indices]

    # Application missing (workspace only)
    df_new.loc[list(new_missing_indices), column] = np.nan

    return df_new