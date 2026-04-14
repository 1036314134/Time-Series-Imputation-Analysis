from __future__ import annotations
import importlib.util
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.read_data import load_data_one_series


def _load_timesfm_forecastor():
    module_path = Path(__file__).resolve().parent / "timesfm_2p0_500m_forecastor.py"
    module_name = "timesfm_2p0_500m_forecastor_module"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "timesfm_2p0_500m_forecastor"):
        raise AttributeError("timesfm_2p0_500m_forecastor.py does not define timesfm_2p0_500m_forecastor.")

    return module.timesfm_2p0_500m_forecastor


timesfm_2p0_500m_forecastor = _load_timesfm_forecastor()


def set_random_seed(random_seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def sliding_window_forecast_test(
    csv_relative_path: str,
    lookback_window: int,
    forecast_window: int,
    num_samples: int = 100,
    freq: str | None = None,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if lookback_window <= 0:
        raise ValueError("lookback_window must be a positive integer.")

    if forecast_window <= 0:
        raise ValueError("forecast_window must be a positive integer.")

    set_random_seed(random_seed)

    series_df = load_data_one_series(csv_relative_path).reset_index(drop=True)
    timestamp_col = series_df.columns[0]
    value_col = series_df.columns[1]

    total_length = len(series_df)
    if total_length == 0:
        raise ValueError("The input series is empty.")

    test_start = math.floor(total_length * 0.4)
    if test_start <= 0:
        raise ValueError("The series is too short to create a test split with historical context.")

    test_df = series_df.iloc[test_start:].reset_index(drop=True)
    if test_df.empty:
        raise ValueError("The test set is empty after splitting the series.")

    results = []
    window_records = []
    test_length = len(test_df)

    for window_index, offset in enumerate(range(0, test_length, forecast_window), start=1):
        print("forecasting window: ", window_index)
        history_end = test_start + offset
        history_start = max(0, history_end - lookback_window)
        history_df = series_df.iloc[history_start:history_end].reset_index(drop=True)

        if history_df.empty:
            raise ValueError("Historical context is empty for the current sliding window.")

        current_window_size = min(forecast_window, test_length - offset)
        prediction_start = test_start + offset
        prediction_end = prediction_start + current_window_size - 1

        forecast_df = timesfm_2p0_500m_forecastor(
            dataframe=history_df,
            forecast_length=forecast_window,
            num_samples=num_samples,
            freq=freq,
        ).iloc[:current_window_size].reset_index(drop=True)

        truth_df = test_df.iloc[offset:offset + current_window_size].reset_index(drop=True)
        truth_timestamps = truth_df[timestamp_col].reset_index(drop=True)
        pred_timestamps = forecast_df[timestamp_col].reset_index(drop=True)

        timestamp_mismatch = truth_timestamps.astype(str) != pred_timestamps.astype(str)
        if timestamp_mismatch.any():
            forecast_df[timestamp_col] = truth_timestamps

        window_result = pd.DataFrame(
            {
                "timestamp": truth_timestamps,
                "true_value": pd.to_numeric(truth_df[value_col], errors="coerce"),
                "pred_value": pd.to_numeric(forecast_df[value_col], errors="coerce"),
            }
        )
        results.append(window_result)
        window_records.append(
            {
                "window_id": window_index,
                "lookback_start": history_start,
                "lookback_end": history_end - 1,
                "forecast_start": prediction_start,
                "forecast_end": prediction_end,
            }
        )

    result_df = pd.concat(results, ignore_index=True)
    window_df = pd.DataFrame(window_records)
    return result_df, window_df


def save_dataframe_to_csv(dataframe: pd.DataFrame, output_csv: str | Path) -> None:
    output_path = Path(output_csv)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def test_timesfm_2p0_500m_forecastor(
    csv_relative_path,
    lookback_window: int = 2880,
    forecast_window: int = 720,
    num_samples: int = 100,
    freq: str | None = None,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return sliding_window_forecast_test(
        csv_relative_path=csv_relative_path,
        lookback_window=lookback_window,
        forecast_window=forecast_window,
        num_samples=num_samples,
        freq=freq,
        random_seed=random_seed,
    )


if __name__ == "__main__":
    result_df, window_df = test_timesfm_2p0_500m_forecastor(csv_relative_path="dataset/exchange_rate/exchange_rate.csv")
    print(f"Generated {len(result_df)} predictions across {len(window_df)} windows.")
    save_dataframe_to_csv(result_df, "results/exchange_rate_predictions_timesfm_2p0_500m.csv")
    save_dataframe_to_csv(window_df, "results/exchange_rate_windows_timesfm_2p0_500m.csv")
