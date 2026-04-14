import inspect
from pathlib import Path

import pandas as pd
import torch

try:
    import timesfm
except ImportError as exc:
    raise ImportError(
        "timesfm package is required. Install an archived 2.0-compatible release, e.g. `pip install timesfm==1.3.0`."
    ) from exc


LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "timesfm_2p0_500m_pytorch"


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


def _resolve_checkpoint_file(model_dir: Path) -> Path:
    if not model_dir.exists() or not any(model_dir.iterdir()):
        raise FileNotFoundError(
            f"Local model directory not found or empty: {model_dir}. "
            "Run download_timesfm_2p0_500m.py first."
        )

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
        candidates = sorted(model_dir.rglob(pattern))
        file_candidates = [path for path in candidates if path.is_file()]
        if file_candidates:
            return file_candidates[0]

    raise FileNotFoundError(
        f"No checkpoint file found under: {model_dir}. "
        "Expected one of: *.ckpt, *.bin, *.pt, *.pth"
    )


def _build_local_checkpoint(model_dir: Path):
    checkpoint_file = _resolve_checkpoint_file(model_dir)

    signature = inspect.signature(timesfm.TimesFmCheckpoint)
    parameter_names = set(signature.parameters)
    checkpoint_file_str = str(checkpoint_file)
    model_dir_str = str(model_dir)

    if "path" in parameter_names:
        return timesfm.TimesFmCheckpoint(path=checkpoint_file_str)
    if "checkpoint_path" in parameter_names:
        return timesfm.TimesFmCheckpoint(checkpoint_path=checkpoint_file_str)
    if "checkpoint_dir" in parameter_names:
        return timesfm.TimesFmCheckpoint(checkpoint_dir=model_dir_str)
    if "local_dir" in parameter_names:
        return timesfm.TimesFmCheckpoint(local_dir=model_dir_str)

    raise TypeError(
        "Unsupported TimesFmCheckpoint signature for local loading. "
        f"Available parameters: {sorted(parameter_names)}"
    )


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


def timesfm_2p0_500m_forecastor(dataframe, forecast_length, num_samples=100, freq=None, device=None):
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if dataframe.shape[1] != 2:
        raise ValueError("Input dataframe must contain exactly two columns.")

    if forecast_length <= 0:
        raise ValueError("forecast_length must be a positive integer.")

    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")

    resolved_device = _resolve_device(device)
    backend = "gpu" if str(resolved_device).lower().startswith("cuda") else "cpu"
    print(f"Using device: {resolved_device}")

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
            if inferred_freq is not None:
                freq = inferred_freq
            else:
                freq = "D"
        else:
            freq = "D"

    freq_category = _freq_to_category(freq)

    checkpoint = _build_local_checkpoint(LOCAL_MODEL_DIR)

    model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend=backend,
            per_core_batch_size=32,
            horizon_len=int(forecast_length),
            input_patch_len=32,
            output_patch_len=128,
            num_layers=50,
            model_dims=1280,
            use_positional_embedding=False,
        ),
        checkpoint=checkpoint,
    )

    point_forecast, _ = model.forecast(
        inputs=[input_df[value_col].to_numpy(dtype=float)],
        freq=[freq_category],
    )

    single_forecast = point_forecast[0]
    single_forecast = single_forecast[:forecast_length]

    if len(single_forecast) < forecast_length:
        raise RuntimeError("Model returned fewer forecast points than requested forecast_length.")

    # Keep API compatibility with the same signature as other forecastors.
    # TimesFM 2p0 point forecast is deterministic; num_samples does not change the output.
    forecast_mean = pd.Series(single_forecast, dtype="float64").to_numpy()
    future_timestamps = _infer_future_timestamps(input_df[timestamp_col], forecast_length)

    return pd.DataFrame(
        {
            timestamp_col: future_timestamps,
            value_col: forecast_mean,
        }
    )
