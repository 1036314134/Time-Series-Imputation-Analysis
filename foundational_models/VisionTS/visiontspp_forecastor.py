from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from transformers import AutoModel
except ImportError as exc:
    raise ImportError("transformers package is required for VisionTSpp.") from exc


LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "VisionTSpp"
MODEL_REPO_ID = "Lefei/VisionTSpp"
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
    if model_dir.exists() and any(model_dir.iterdir()):
        return model_dir
    raise FileNotFoundError(
        f"Local model directory not found or empty: {model_dir}. "
        "Run download_visiontspp.py first."
    )


def _load_visiontspp_model(model_dir: Path, resolved_device: str):
    model_dir = _resolve_local_model_dir(model_dir)
    cache_key = (str(model_dir), resolved_device)
    model = _MODEL_CACHE.get(cache_key)
    if model is not None:
        return model

    signature = inspect.signature(AutoModel.from_pretrained)
    kwargs = {"trust_remote_code": True}
    if "local_files_only" in signature.parameters:
        kwargs["local_files_only"] = True
    if "torch_dtype" in signature.parameters:
        kwargs["torch_dtype"] = torch.bfloat16 if resolved_device.startswith("cuda") else torch.float32

    try:
        model = AutoModel.from_pretrained(str(model_dir), **kwargs)
    except Exception:
        kwargs.pop("local_files_only", None)
        model = AutoModel.from_pretrained(MODEL_REPO_ID, **kwargs)

    if hasattr(model, "to"):
        model = model.to(resolved_device)
    if hasattr(model, "eval"):
        model.eval()

    _MODEL_CACHE[cache_key] = model
    return model


def _normalize_prediction_output(output, forecast_length: int):
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, dict):
        for key in ("mean", "pred", "prediction", "predictions", "forecast", "point_forecast", "yhat", "samples"):
            if key in output:
                return _normalize_prediction_output(output[key], forecast_length)
    if hasattr(output, "predictions"):
        return _normalize_prediction_output(output.predictions, forecast_length)
    if hasattr(output, "mean_predictions"):
        return _normalize_prediction_output(output.mean_predictions, forecast_length)
    if hasattr(output, "logits"):
        return _normalize_prediction_output(output.logits, forecast_length)
    if isinstance(output, pd.DataFrame):
        if output.shape[1] == 0:
            raise RuntimeError("VisionTSpp returned an empty DataFrame.")
        return pd.to_numeric(output.iloc[:, -1], errors="coerce").to_numpy(dtype="float64")
    if isinstance(output, pd.Series):
        return pd.to_numeric(output, errors="coerce").to_numpy(dtype="float64")
    if isinstance(output, torch.Tensor):
        tensor = output.detach().to("cpu")
        if tensor.ndim == 0:
            return np.array([tensor.item()], dtype="float64")
        if tensor.ndim == 1:
            return tensor.numpy().astype("float64")
        if tensor.ndim == 2:
            if tensor.shape[0] == 1:
                return tensor[0].numpy().astype("float64")
            if tensor.shape[1] == 1:
                return tensor[:, 0].numpy().astype("float64")
            return tensor.mean(dim=0).numpy().astype("float64")
        return tensor.reshape(-1).numpy().astype("float64")

    array = np.asarray(output)
    if array.ndim == 0:
        return np.array([float(array)], dtype="float64")
    if array.ndim == 1:
        return array.astype("float64")
    if array.ndim == 2:
        if array.shape[0] == 1:
            return array[0].astype("float64")
        if array.shape[1] == 1:
            return array[:, 0].astype("float64")
        return array.mean(axis=0).astype("float64")
    return array.reshape(-1).astype("float64")


def _predict_with_visiontspp(model, history_values: np.ndarray, forecast_length: int, num_samples: int):
    methods = ["forecast", "predict", "generate", "forward"]
    call_attempts = [
        {"history": history_values, "prediction_length": int(forecast_length), "num_samples": int(num_samples)},
        {"history": history_values, "horizon": int(forecast_length), "num_samples": int(num_samples)},
        {"inputs": [history_values], "prediction_length": int(forecast_length), "num_samples": int(num_samples)},
        {"inputs": [history_values], "horizon": int(forecast_length), "num_samples": int(num_samples)},
        {"input_ids": torch.tensor(history_values, dtype=torch.float32).unsqueeze(0), "max_new_tokens": int(forecast_length)},
        {"x": torch.tensor(history_values, dtype=torch.float32).unsqueeze(0), "prediction_length": int(forecast_length)},
        {"x": torch.tensor(history_values, dtype=torch.float32).unsqueeze(0), "horizon": int(forecast_length)},
        {"prediction_length": int(forecast_length)},
        {"horizon": int(forecast_length)},
        {},
    ]

    errors = []
    for method_name in methods:
        if not hasattr(model, method_name):
            continue
        method = getattr(model, method_name)
        for kwargs in call_attempts:
            try:
                with torch.no_grad():
                    output = method(**kwargs)
                values = _normalize_prediction_output(output, int(forecast_length))
                if len(values) >= int(forecast_length):
                    return values[: int(forecast_length)]
            except TypeError as exc:
                errors.append(f"{method_name}({list(kwargs.keys())}): {exc}")
                continue
            except Exception as exc:
                errors.append(f"{method_name}({list(kwargs.keys())}): {exc}")
                continue

    raise RuntimeError(
        "VisionTSpp inference failed with all known method signatures. "
        f"Error samples: {errors[:3]}"
    )


def visiontspp_forecastor(dataframe, forecast_length, num_samples=100, freq=None, device=None):
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if dataframe.shape[1] != 2:
        raise ValueError("Input dataframe must contain exactly two columns.")

    if forecast_length <= 0:
        raise ValueError("forecast_length must be a positive integer.")

    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")

    resolved_device = _resolve_device(device)
    print(f"Using device: {resolved_device}")

    input_df = dataframe.iloc[:, [0, 1]].copy().reset_index(drop=True)
    timestamp_col = input_df.columns[0]
    value_col = input_df.columns[1]

    input_df[value_col] = pd.to_numeric(input_df[value_col], errors="coerce")
    if input_df[value_col].isna().any():
        raise ValueError("The value column must be numeric and cannot contain NaN after conversion.")

    model = _load_visiontspp_model(
        model_dir=LOCAL_MODEL_DIR,
        resolved_device=resolved_device,
    )

    history_values = input_df[value_col].to_numpy(dtype="float64")
    forecast_mean = _predict_with_visiontspp(
        model=model,
        history_values=history_values,
        forecast_length=int(forecast_length),
        num_samples=int(num_samples),
    )

    if len(forecast_mean) < int(forecast_length):
        raise RuntimeError("Model returned fewer forecast points than requested forecast_length.")

    _ = freq
    future_timestamps = _infer_future_timestamps(input_df[timestamp_col], forecast_length)

    return pd.DataFrame(
        {
            timestamp_col: future_timestamps,
            value_col: pd.Series(forecast_mean[: int(forecast_length)], dtype="float64").to_numpy(),
        }
    )
