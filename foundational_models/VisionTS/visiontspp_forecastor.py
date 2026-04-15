from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from visionts import VisionTSpp, freq_to_seasonality_list
except ImportError as exc:
    raise ImportError("visionts package is required for VisionTSpp.") from exc


LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "VisionTSpp"
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


def _select_local_checkpoint(model_dir: Path) -> Path:
    candidate_names = [
        "visiontspp_model.ckpt",
        "visiontspp_base.ckpt",
        "visiontspp_large.ckpt",
        "visiontspp_base_gifteval_no_leakage.ckpt",
        "visiontspp_large_gifteval_no_leakage.ckpt",
    ]

    for name in candidate_names:
        candidate = model_dir / name
        if candidate.is_file():
            return candidate

    checkpoints = sorted(model_dir.glob("*.ckpt"))
    if checkpoints:
        return checkpoints[0]

    raise FileNotFoundError(
        f"No VisionTSpp checkpoint file was found in {model_dir}."
    )


def _infer_arch_from_checkpoint(ckpt_path: Path) -> str:
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    qkv_weight = state_dict.get("blocks.0.attn.qkv.weight")

    if qkv_weight is None or len(qkv_weight.shape) != 2:
        raise RuntimeError(
            f"Unable to infer VisionTSpp architecture from checkpoint: {ckpt_path}"
        )

    embed_dim = int(qkv_weight.shape[1])
    arch_by_embed_dim = {
        768: "mae_base",
        1024: "mae_large",
        1280: "mae_huge",
    }
    if embed_dim not in arch_by_embed_dim:
        raise RuntimeError(
            f"Unsupported VisionTSpp embed_dim {embed_dim} in checkpoint: {ckpt_path}"
        )
    return arch_by_embed_dim[embed_dim]


def _resolve_periodicity(freq: str | None) -> int:
    if not freq:
        return 1

    try:
        candidates = freq_to_seasonality_list(freq)
    except Exception:
        return 1

    for candidate in candidates:
        candidate_int = int(candidate)
        if candidate_int > 1:
            return candidate_int
    return 1


def _load_visiontspp_model(model_dir: Path, resolved_device: str):
    model_dir = _resolve_local_model_dir(model_dir)
    ckpt_path = _select_local_checkpoint(model_dir)
    arch = _infer_arch_from_checkpoint(ckpt_path)
    cache_key = (str(ckpt_path), arch, resolved_device)
    model = _MODEL_CACHE.get(cache_key)
    if model is not None:
        return model

    model = VisionTSpp(
        arch=arch,
        ckpt_path=str(ckpt_path),
        load_ckpt=True,
        quantile=True,
    )
    model = model.to(resolved_device)
    model.eval()

    _MODEL_CACHE[cache_key] = model
    return model


def _normalize_prediction_output(output, forecast_length: int):
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, list):
        if len(output) == 0:
            raise RuntimeError("VisionTSpp returned an empty list.")
        return _normalize_prediction_output(output[0], forecast_length)
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
    _ = num_samples
    model_device = next(model.parameters()).device
    history_tensor = (
        torch.tensor(history_values, dtype=torch.float32, device=model_device)
        .reshape(1, int(len(history_values)), 1)
    )

    with torch.no_grad():
        output = model(history_tensor)

    values = _normalize_prediction_output(output, int(forecast_length))
    if len(values) < int(forecast_length):
        raise RuntimeError("VisionTSpp returned fewer forecast points than requested forecast_length.")
    return values[: int(forecast_length)]


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
    model.update_config(
        context_len=int(len(input_df)),
        pred_len=int(forecast_length),
        periodicity=_resolve_periodicity(freq),
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
