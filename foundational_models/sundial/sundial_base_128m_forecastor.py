from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM

LOCAL_MODEL_DIR = str(Path(__file__).resolve().parent / "sundial_base_128m")
_MODEL_CACHE = {}


def _infer_future_timestamps(timestamp_series, forecast_length):
    timestamp_series = timestamp_series.reset_index(drop=True)

    if len(timestamp_series) == 0:
        raise ValueError("Input dataframe must contain at least one row.")

    parsed_timestamps = pd.to_datetime(timestamp_series, errors="coerce")
    is_datetime_series = parsed_timestamps.notna().all()

    if is_datetime_series:
        datetime_index = pd.DatetimeIndex(parsed_timestamps)
        inferred_freq = pd.infer_freq(datetime_index)

        if inferred_freq is not None:
            start_timestamp = datetime_index[-1] + pd.tseries.frequencies.to_offset(inferred_freq)
            future_timestamps = pd.date_range(
                start=start_timestamp,
                periods=forecast_length,
                freq=inferred_freq,
            )
            return pd.Series(future_timestamps, name=timestamp_series.name)

        if len(datetime_index) < 2:
            raise ValueError("At least two timestamps are required when frequency cannot be inferred.")

        step = datetime_index[-1] - datetime_index[-2]
        future_timestamps = [datetime_index[-1] + step * (i + 1) for i in range(forecast_length)]
        return pd.Series(future_timestamps, name=timestamp_series.name)

    if len(timestamp_series) < 2:
        raise ValueError("At least two timestamps are required for non-datetime indices.")

    step = timestamp_series.iloc[-1] - timestamp_series.iloc[-2]
    future_timestamps = [timestamp_series.iloc[-1] + step * (i + 1) for i in range(forecast_length)]
    return pd.Series(future_timestamps, name=timestamp_series.name)


def sundial_forecastor(dataframe, forecast_length, num_samples=100, device=None):
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if dataframe.shape[1] != 2:
        raise ValueError("Input dataframe must contain exactly two columns.")

    if forecast_length <= 0:
        raise ValueError("forecast_length must be a positive integer.")

    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")

    input_df = dataframe.iloc[:, [0, 1]].copy().reset_index(drop=True)
    timestamp_col = input_df.columns[0]
    value_col = input_df.columns[1]

    input_df[value_col] = pd.to_numeric(input_df[value_col], errors="coerce")
    if input_df[value_col].isna().any():
        raise ValueError("The value column must be numeric and cannot contain NaN after conversion.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = str(device)

    print(f"Using device: {device}")

    model = _MODEL_CACHE.get(device)
    if model is None:
        print(f"Loading Sundial model once for device={device} from: {LOCAL_MODEL_DIR}")
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_DIR,
            trust_remote_code=True,
        ).to(device)
        model.eval()
        _MODEL_CACHE[device] = model

    input_tensor = torch.tensor(
        input_df[value_col].to_numpy(),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_new_tokens=forecast_length,
            num_samples=num_samples,
        )

    forecast_mean = output.mean(dim=1).squeeze(0).detach().cpu().numpy()
    future_timestamps = _infer_future_timestamps(input_df[timestamp_col], forecast_length)

    return pd.DataFrame(
        {
            timestamp_col: future_timestamps,
            value_col: forecast_mean,
        }
    )
