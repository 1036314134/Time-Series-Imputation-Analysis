from pathlib import Path

import pandas as pd
import torch

try:
    import chronos
except ImportError as exc:
    raise ImportError("chronos package is required. Install chronos-forecasting.") from exc


LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "chronos_2"
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


def _resolve_device(device):
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"

    device_str = str(device).lower()
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is requested but not available. Falling back to CPU.")
        return "cpu"

    if device_str.startswith("cuda"):
        return str(device)
    return "cpu"


def _resolve_local_model_dir(model_dir: Path) -> Path:
    if not model_dir.exists() or not any(model_dir.iterdir()):
        raise FileNotFoundError(
            f"Local model directory not found or empty: {model_dir}. "
            "Run download_chronos_2.py first."
        )
    return model_dir


def _load_chronos_2_model(model_dir: Path, resolved_device: str):
    model_dir = _resolve_local_model_dir(model_dir)

    if not hasattr(chronos, "Chronos2Pipeline"):
        raise AttributeError(
            "Current chronos package does not provide Chronos2Pipeline. "
            "Please upgrade chronos-forecasting."
        )

    pipeline_cls = chronos.Chronos2Pipeline
    device_map = "cuda" if str(resolved_device).lower().startswith("cuda") else "cpu"
    dtype = torch.bfloat16 if device_map == "cuda" else torch.float32

    try:
        return pipeline_cls.from_pretrained(
            str(model_dir),
            local_files_only=True,
            device_map=device_map,
            dtype=dtype,
        )
    except TypeError:
        try:
            return pipeline_cls.from_pretrained(
                str(model_dir),
                local_files_only=True,
                device_map=device_map,
                torch_dtype=dtype,
            )
        except TypeError:
            return pipeline_cls.from_pretrained(
                str(model_dir),
                device_map=device_map,
                torch_dtype=dtype,
            )


def _get_chronos_2_model(resolved_device: str):
    model = _MODEL_CACHE.get(resolved_device)
    if model is not None:
        return model

    model = _load_chronos_2_model(
        model_dir=LOCAL_MODEL_DIR,
        resolved_device=resolved_device,
    )
    _MODEL_CACHE[resolved_device] = model
    return model


def _forecast_with_predict(model, history_values, forecast_length: int):
    if not hasattr(model, "predict"):
        raise AttributeError("Loaded Chronos-2 pipeline does not provide predict().")

    inputs = [torch.tensor(history_values, dtype=torch.float32)]

    with torch.no_grad():
        forecast = model.predict(
            inputs=inputs,
            prediction_length=int(forecast_length),
        )
        pred_tensor = torch.as_tensor(forecast[0], dtype=torch.float32)

        if pred_tensor.ndim == 3:
            # Chronos-2 returns (n_variates, n_quantiles, prediction_length).
            if pred_tensor.shape[0] != 1:
                raise RuntimeError(
                    "Only univariate forecasts are supported, but Chronos-2 returned multiple variates."
                )
            output = pred_tensor.mean(dim=1).squeeze(0)
        elif pred_tensor.ndim == 2:
            if pred_tensor.shape[-1] != int(forecast_length):
                raise RuntimeError("Chronos-2 returned an unexpected 2D prediction shape.")
            output = pred_tensor.mean(dim=0)
        elif pred_tensor.ndim == 1:
            output = pred_tensor
        else:
            raise RuntimeError("Chronos-2 returned an unexpected prediction shape.")

        output = output.flatten()[: int(forecast_length)]

    if output.numel() < int(forecast_length):
        raise RuntimeError("Model returned fewer forecast points than requested forecast_length.")
    return output.detach().cpu().numpy()


def chronos_2_forecastor(dataframe, forecast_length, freq=None, device=None):
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if dataframe.shape[1] != 2:
        raise ValueError("Input dataframe must contain exactly two columns.")

    if forecast_length <= 0:
        raise ValueError("forecast_length must be a positive integer.")

    resolved_device = _resolve_device(device)
    print(f"Using device: {resolved_device}")

    input_df = dataframe.iloc[:, [0, 1]].copy().reset_index(drop=True)
    timestamp_col = input_df.columns[0]
    value_col = input_df.columns[1]

    input_df[value_col] = pd.to_numeric(input_df[value_col], errors="coerce")
    if input_df[value_col].isna().any():
        raise ValueError("The value column must be numeric and cannot contain NaN after conversion.")

    model = _get_chronos_2_model(resolved_device=resolved_device)
    forecast_mean = _forecast_with_predict(
        model=model,
        history_values=input_df[value_col].to_numpy(dtype=float),
        forecast_length=int(forecast_length),
    )

    # Keep API compatibility with other forecastors.
    _ = freq
    future_timestamps = _infer_future_timestamps(input_df[timestamp_col], forecast_length)

    return pd.DataFrame(
        {
            timestamp_col: future_timestamps,
            value_col: forecast_mean,
        }
    )
