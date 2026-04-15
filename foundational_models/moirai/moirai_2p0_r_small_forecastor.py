from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
except ImportError:
    MoiraiForecast = None
    MoiraiModule = None

try:
    from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
except ImportError:
    MoiraiMoEForecast = None
    MoiraiMoEModule = None


LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "moirai_2p0_r_small"
MODEL_REPO_ID = "Salesforce/moirai-2.0-R-small"
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
        "Run moirai_2p0_r_small_downloader.py first."
    )


def _build_module_from_hf(module_cls, source: str):
    signature = inspect.signature(module_cls.from_pretrained)
    kwargs = {}
    if "local_files_only" in signature.parameters:
        kwargs["local_files_only"] = True
    return module_cls.from_pretrained(source, **kwargs)


def _load_moirai_model(model_dir: Path, resolved_device: str, forecast_length: int, num_samples: int):
    model_dir = _resolve_local_model_dir(model_dir)
    cache_key = (str(model_dir), resolved_device, int(forecast_length), int(num_samples))
    model = _MODEL_CACHE.get(cache_key)
    if model is not None:
        return model

    if MoiraiModule is None and MoiraiMoEModule is None:
        raise ImportError("uni2ts package is required for moirai models. Please install uni2ts.")

    errors = []

    if MoiraiMoEModule is not None and MoiraiMoEForecast is not None:
        try:
            module = _build_module_from_hf(MoiraiMoEModule, str(model_dir))
        except Exception:
            module = _build_module_from_hf(MoiraiMoEModule, MODEL_REPO_ID)

        try:
            model = MoiraiMoEForecast(
                module=module,
                prediction_length=int(forecast_length),
                context_length=max(128, int(forecast_length) * 2),
                num_samples=int(num_samples),
                patch_size="auto",
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
        except Exception as exc:
            errors.append(f"MoiraiMoEForecast init failed: {exc}")
            model = None

        if model is not None:
            _MODEL_CACHE[cache_key] = model
            return model

    if MoiraiModule is not None and MoiraiForecast is not None:
        try:
            module = _build_module_from_hf(MoiraiModule, str(model_dir))
        except Exception:
            module = _build_module_from_hf(MoiraiModule, MODEL_REPO_ID)

        try:
            model = MoiraiForecast(
                module=module,
                prediction_length=int(forecast_length),
                context_length=max(128, int(forecast_length) * 2),
                num_samples=int(num_samples),
                patch_size="auto",
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
        except Exception as exc:
            errors.append(f"MoiraiForecast init failed: {exc}")
            model = None

        if model is not None:
            _MODEL_CACHE[cache_key] = model
            return model

    raise RuntimeError(f"Unable to initialize moirai model. Errors: {errors}")


def _extract_forecast_values(forecast_obj, forecast_length: int):
    if isinstance(forecast_obj, dict):
        for key in ("mean", "pred", "prediction", "predictions", "forecast", "samples", "target"):
            if key in forecast_obj:
                return _extract_forecast_values(forecast_obj[key], forecast_length)

    if hasattr(forecast_obj, "mean"):
        try:
            return _extract_forecast_values(forecast_obj.mean, forecast_length)
        except Exception:
            pass

    if hasattr(forecast_obj, "samples"):
        samples = np.asarray(forecast_obj.samples)
        if samples.ndim >= 2:
            return samples.reshape(samples.shape[0], -1).mean(axis=0)[:forecast_length]

    array = np.asarray(forecast_obj)
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
    if array.ndim >= 3:
        return array.reshape(array.shape[0], -1).mean(axis=0).astype("float64")
    return array.reshape(-1).astype("float64")


def _predict_with_moirai(model, history_values: np.ndarray, forecast_length: int):
    for method_name in ("forecast", "predict"):
        if not hasattr(model, method_name):
            continue
        method = getattr(model, method_name)

        attempts = [
            {"inputs": [history_values]},
            {"inputs": [history_values.tolist()]},
            {"dataset": [{"target": history_values}]},
            {"dataset": [{"target": history_values.tolist()}]},
            {"target": history_values},
            {"target": history_values.tolist()},
        ]

        for kwargs in attempts:
            try:
                output = method(**kwargs)
                if isinstance(output, (list, tuple)) and len(output) > 0:
                    output = output[0]
                values = _extract_forecast_values(output, forecast_length=int(forecast_length))
                if len(values) >= int(forecast_length):
                    return values[: int(forecast_length)]
            except TypeError:
                continue
            except Exception:
                continue

    raise RuntimeError("Moirai forecast failed with all known calling patterns.")


def moirai_2p0_r_small_forecastor(dataframe, forecast_length, num_samples=100, freq=None, device=None):
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

    model = _load_moirai_model(
        model_dir=LOCAL_MODEL_DIR,
        resolved_device=resolved_device,
        forecast_length=int(forecast_length),
        num_samples=int(num_samples),
    )

    history_values = input_df[value_col].to_numpy(dtype="float64")
    forecast_mean = _predict_with_moirai(
        model=model,
        history_values=history_values,
        forecast_length=int(forecast_length),
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
