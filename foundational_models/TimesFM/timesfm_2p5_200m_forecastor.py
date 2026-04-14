from pathlib import Path

import pandas as pd
import torch

try:
    import timesfm
except ImportError as exc:
    raise ImportError(
        "timesfm package is required. Install the package version that supports TimesFM 2.5 PyTorch."
    ) from exc


LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "timesfm_2p5_200m_pytorch"
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
            "Run download_timesfm_2p5_200m.py first."
        )
    return model_dir


def _resolve_safetensors_file(model_dir: Path) -> Path:
    model_file = model_dir / "model.safetensors"
    if model_file.exists() and model_file.is_file():
        return model_file

    candidates = sorted(path for path in model_dir.rglob("*.safetensors") if path.is_file())
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"No .safetensors checkpoint found under: {model_dir}. "
        "Expected model.safetensors from google/timesfm-2.5-200m-pytorch."
    )


def _load_timesfm_2p5_model(model_dir: Path, resolved_device: str, forecast_length: int):
    model_dir = _resolve_local_model_dir(model_dir)
    checkpoint_file = _resolve_safetensors_file(model_dir)

    if not hasattr(timesfm, "TimesFM_2p5_200M_torch"):
        raise AttributeError(
            "Current timesfm package does not provide TimesFM_2p5_200M_torch. "
            "Please upgrade timesfm to a version supporting 2.5 PyTorch."
        )

    # Avoid from_pretrained() to bypass incompatible huggingface_hub kwargs (e.g. proxies).
    model = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)

    if hasattr(model, "model") and hasattr(model.model, "load_checkpoint"):
        model.model.load_checkpoint(str(checkpoint_file), torch_compile=False)
    elif hasattr(model, "load_checkpoint"):
        model.load_checkpoint(str(checkpoint_file), torch_compile=False)
    else:
        raise AttributeError("Loaded TimesFM 2.5 class has no load_checkpoint method.")

    if hasattr(model, "model") and hasattr(model.model, "to"):
        target_device = torch.device(resolved_device)
        model.model.to(target_device)
        model.model.device = target_device
        if target_device.type == "cuda" and torch.cuda.is_available():
            model.model.device_count = max(1, torch.cuda.device_count())
        else:
            model.model.device_count = 1

    if hasattr(model, "eval"):
        model.eval()
    elif hasattr(model, "model") and hasattr(model.model, "eval"):
        model.model.eval()

    if hasattr(timesfm, "ForecastConfig") and hasattr(model, "compile"):
        model.compile(
            timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=int(forecast_length),
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )

    return model


def _get_timesfm_2p5_model(resolved_device: str, forecast_length: int):
    cache_key = (resolved_device, int(forecast_length))
    model = _MODEL_CACHE.get(cache_key)
    if model is not None:
        return model

    model = _load_timesfm_2p5_model(
        model_dir=LOCAL_MODEL_DIR,
        resolved_device=resolved_device,
        forecast_length=int(forecast_length),
    )
    _MODEL_CACHE[cache_key] = model
    return model


def timesfm_2p5_200m_forecastor(dataframe, forecast_length, num_samples=100, freq=None, device=None):
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

    model = _get_timesfm_2p5_model(
        resolved_device=resolved_device,
        forecast_length=int(forecast_length),
    )

    if not hasattr(model, "forecast"):
        raise AttributeError("Loaded model does not provide forecast().")

    # Keep API compatibility with the 2p0 forecastor signature.
    _ = freq
    with torch.no_grad():
        point_forecast, _full_forecast = model.forecast(
            horizon=int(forecast_length),
            inputs=[input_df[value_col].to_numpy(dtype=float)],
        )

    single_forecast = point_forecast[0]
    single_forecast = single_forecast[:forecast_length]

    if len(single_forecast) < forecast_length:
        raise RuntimeError("Model returned fewer forecast points than requested forecast_length.")

    # TimesFM 2p5 point forecast is deterministic; num_samples does not change the output.
    forecast_mean = pd.Series(single_forecast, dtype="float64").to_numpy()
    future_timestamps = _infer_future_timestamps(input_df[timestamp_col], forecast_length)

    return pd.DataFrame(
        {
            timestamp_col: future_timestamps,
            value_col: forecast_mean,
        }
    )
