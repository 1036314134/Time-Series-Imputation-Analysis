import inspect
from pathlib import Path

import pandas as pd
import torch

try:
    import timesfm as _timesfm_pkg
except Exception:
    _timesfm_pkg = None

try:
    import transformers as _transformers_pkg
except Exception:
    _transformers_pkg = None


LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "timesfm_2p0_500m_pytorch"
_LEGACY_MODEL_CACHE = {}
_TRANSFORMERS_MODEL_CACHE = {}


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


def _freq_to_category(freq):
    if freq is None:
        return 0

    freq_upper = str(freq).upper()
    high_freq = ("T", "MIN", "H", "D", "B", "U", "S")
    medium_freq = ("W", "M", "MS", "BM", "BMS")
    low_freq = ("Q", "Y", "A", "AS")

    if freq_upper.startswith(high_freq):
        return 0
    if freq_upper.startswith(medium_freq):
        return 1
    if freq_upper.startswith(low_freq):
        return 2
    return 0


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


def _ensure_local_model_dir(model_dir: Path):
    if not model_dir.exists() or not any(model_dir.iterdir()):
        raise FileNotFoundError(
            f"Local model directory not found or empty: {model_dir}. "
            "Run download_timesfm_2p0_500m.py first."
        )


def _resolve_checkpoint_file(model_dir: Path) -> Path:
    preferred_names = [
        "torch_model.ckpt",
        "pytorch_model.bin",
        "model.ckpt",
        "checkpoint.pt",
        "checkpoint.pth",
    ]
    for name in preferred_names:
        candidate = model_dir / name
        if candidate.exists() and candidate.is_file():
            return candidate

    for pattern in ("*.ckpt", "*.bin", "*.pt", "*.pth"):
        candidates = sorted(path for path in model_dir.rglob(pattern) if path.is_file())
        if candidates:
            return candidates[0]

    raise FileNotFoundError(
        f"No checkpoint file found under: {model_dir}. "
        "Expected one of: *.ckpt, *.bin, *.pt, *.pth"
    )


def _supports_legacy_timesfm():
    return (
        _timesfm_pkg is not None
        and hasattr(_timesfm_pkg, "TimesFm")
        and hasattr(_timesfm_pkg, "TimesFmHparams")
        and hasattr(_timesfm_pkg, "TimesFmCheckpoint")
    )


def _build_legacy_checkpoint(model_dir: Path):
    checkpoint_file = _resolve_checkpoint_file(model_dir)
    signature = inspect.signature(_timesfm_pkg.TimesFmCheckpoint)
    param_names = set(signature.parameters)
    checkpoint_file_str = str(checkpoint_file)
    model_dir_str = str(model_dir)

    if "path" in param_names:
        return _timesfm_pkg.TimesFmCheckpoint(path=checkpoint_file_str)
    if "checkpoint_path" in param_names:
        return _timesfm_pkg.TimesFmCheckpoint(checkpoint_path=checkpoint_file_str)
    if "checkpoint_dir" in param_names:
        return _timesfm_pkg.TimesFmCheckpoint(checkpoint_dir=model_dir_str)
    if "local_dir" in param_names:
        return _timesfm_pkg.TimesFmCheckpoint(local_dir=model_dir_str)

    raise TypeError(
        "Unsupported TimesFmCheckpoint signature for legacy branch. "
        f"Available parameters: {sorted(param_names)}"
    )


def _build_legacy_hparams(backend: str, forecast_length: int):
    desired = {
        "backend": backend,
        "per_core_batch_size": 32,
        "horizon_len": int(forecast_length),
        "input_patch_len": 32,
        "output_patch_len": 128,
        "num_layers": 50,
        "model_dims": 1280,
        "use_positional_embedding": False,
    }

    signature = inspect.signature(_timesfm_pkg.TimesFmHparams)
    filtered = {k: v for k, v in desired.items() if k in signature.parameters}
    return _timesfm_pkg.TimesFmHparams(**filtered)


def _get_legacy_model(backend: str, forecast_length: int):
    cache_key = (backend, int(forecast_length))
    model = _LEGACY_MODEL_CACHE.get(cache_key)
    if model is not None:
        return model

    checkpoint = _build_legacy_checkpoint(LOCAL_MODEL_DIR)
    hparams = _build_legacy_hparams(backend=backend, forecast_length=forecast_length)
    model = _timesfm_pkg.TimesFm(hparams=hparams, checkpoint=checkpoint)
    _LEGACY_MODEL_CACHE[cache_key] = model
    return model


def _forecast_with_legacy_timesfm(model, history_values, freq_category, forecast_length):

    point_forecast, _ = model.forecast(
        inputs=[history_values],
        freq=[freq_category],
    )
    output = point_forecast[0][:forecast_length]
    if len(output) < forecast_length:
        raise RuntimeError("Legacy TimesFM returned fewer points than forecast_length.")
    return output


def _resolve_transformers_timesfm_model_class():
    if _transformers_pkg is None:
        return None

    try:
        from transformers import TimesFmModelForPrediction as cls  # type: ignore

        return cls
    except Exception:
        pass

    try:
        from transformers.models.timesfm.modeling_timesfm import TimesFmModelForPrediction as cls  # type: ignore

        return cls
    except Exception:
        return None


def _load_transformers_model(resolved_device: str):
    model_class = _resolve_transformers_timesfm_model_class()
    if model_class is None:
        raise ImportError(
            "TimesFM model class is unavailable in current environment. "
            f"timesfm package loaded: {_timesfm_pkg is not None}, "
            f"transformers version: {getattr(_transformers_pkg, '__version__', 'missing')}."
        )

    model = model_class.from_pretrained(
        str(LOCAL_MODEL_DIR),
        local_files_only=True,
        trust_remote_code=True,
    )
    model = model.to(resolved_device)
    model.eval()
    return model


def _get_transformers_model(resolved_device: str):
    model = _TRANSFORMERS_MODEL_CACHE.get(resolved_device)
    if model is not None:
        return model

    model = _load_transformers_model(resolved_device)
    _TRANSFORMERS_MODEL_CACHE[resolved_device] = model
    return model


def _forecast_with_transformers(model, history_values, freq_category, forecast_length, resolved_device):
    context = torch.tensor(history_values, dtype=torch.float32, device=resolved_device).unsqueeze(0)
    freq_tensor = torch.tensor([freq_category], dtype=torch.long, device=resolved_device)

    predictions = []
    max_context = 2048

    while len(predictions) < forecast_length:
        input_context = context[:, -max_context:]
        with torch.no_grad():
            outputs = model(past_values=input_context, freq=freq_tensor)

        if not hasattr(outputs, "mean_predictions"):
            raise RuntimeError("Transformers TimesFM output does not contain mean_predictions.")

        chunk = outputs.mean_predictions[0].detach().to("cpu").numpy()
        if len(chunk) == 0:
            raise RuntimeError("Transformers TimesFM returned an empty forecast chunk.")

        remaining = forecast_length - len(predictions)
        chunk_to_use = chunk[:remaining]
        predictions.extend(chunk_to_use.tolist())

        append_chunk = torch.tensor(chunk_to_use, dtype=torch.float32, device=resolved_device).unsqueeze(0)
        context = torch.cat([context, append_chunk], dim=1)

    return predictions


def timesfm_2p0_500m_forecastor(dataframe, forecast_length, num_samples=100, freq=None, device=None):
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if dataframe.shape[1] != 2:
        raise ValueError("Input dataframe must contain exactly two columns.")

    if forecast_length <= 0:
        raise ValueError("forecast_length must be a positive integer.")

    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")

    _ensure_local_model_dir(LOCAL_MODEL_DIR)

    input_df = dataframe.iloc[:, [0, 1]].copy().reset_index(drop=True)
    timestamp_col = input_df.columns[0]
    value_col = input_df.columns[1]

    input_df[value_col] = pd.to_numeric(input_df[value_col], errors="coerce")
    if input_df[value_col].isna().any():
        raise ValueError("The value column must be numeric and cannot contain NaN after conversion.")

    parsed_timestamps = pd.to_datetime(input_df[timestamp_col], errors="coerce")
    if freq is None:
        if parsed_timestamps.notna().all():
            inferred_freq = pd.infer_freq(pd.DatetimeIndex(parsed_timestamps))
            freq = inferred_freq if inferred_freq is not None else "D"
        else:
            freq = "D"

    freq_category = _freq_to_category(freq)
    resolved_device = _resolve_device(device)
    backend = "gpu" if str(resolved_device).lower().startswith("cuda") else "cpu"
    print(f"Using device: {resolved_device}")

    history_values = input_df[value_col].to_numpy(dtype=float)

    if _supports_legacy_timesfm():
        print("TimesFM backend: legacy timesfm API")
        model = _get_legacy_model(backend=backend, forecast_length=int(forecast_length))
        forecast_values = _forecast_with_legacy_timesfm(
            model=model,
            history_values=history_values,
            freq_category=freq_category,
            forecast_length=int(forecast_length),
        )
    else:
        print("TimesFM backend: transformers API")
        model = _get_transformers_model(resolved_device)
        forecast_values = _forecast_with_transformers(
            model=model,
            history_values=history_values,
            freq_category=freq_category,
            forecast_length=int(forecast_length),
            resolved_device=resolved_device,
        )

    forecast_mean = pd.Series(forecast_values, dtype="float64").to_numpy()
    future_timestamps = _infer_future_timestamps(input_df[timestamp_col], forecast_length)

    # Keep API compatibility with other forecastors. TimesFM point forecast is deterministic.
    _ = num_samples

    return pd.DataFrame(
        {
            timestamp_col: future_timestamps,
            value_col: forecast_mean,
        }
    )
