from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM

LOCAL_MODEL_DIR = str(Path(__file__).resolve().parent / "sundial_base_128m")
_MODEL_CACHE = {}


def _patch_dynamic_cache_compatibility():
    """Make older Sundial remote code work with newer transformers cache APIs."""
    try:
        from transformers import DynamicCache
    except Exception:
        return

    if not hasattr(DynamicCache, "seen_tokens"):
        def _get_seen_tokens(self):
            if hasattr(self, "_seen_tokens"):
                return self._seen_tokens
            return self.get_seq_length()

        DynamicCache.seen_tokens = property(_get_seen_tokens)

    if not hasattr(DynamicCache, "get_max_length"):
        def _get_max_length(self):
            try:
                max_cache_len = self.max_cache_len
                if max_cache_len is None or max_cache_len <= 0:
                    return None
                return max_cache_len
            except Exception:
                pass

            if hasattr(self, "get_max_cache_shape"):
                try:
                    max_cache_shape = self.get_max_cache_shape()
                    if max_cache_shape is None or max_cache_shape <= 0:
                        return None
                    return max_cache_shape
                except Exception:
                    return None

            return None

        DynamicCache.get_max_length = _get_max_length

    if not hasattr(DynamicCache, "get_usable_length"):
        def _get_usable_length(self, new_seq_length, layer_idx=0):
            previous_seq_length = self.get_seq_length(layer_idx)
            max_length = self.get_max_length()

            if max_length is not None and previous_seq_length + new_seq_length > max_length:
                return max(max_length - new_seq_length, 0)

            return previous_seq_length

        DynamicCache.get_usable_length = _get_usable_length


def _patch_sundial_generation_compatibility(model):
    """Route newer transformers generation back to Sundial's legacy custom loop."""
    model_cls = model.__class__
    decoder_cls = model.get_decoder().__class__

    if not getattr(decoder_cls, "_sundial_forward_compat_patched", False):
        original_decoder_forward = decoder_cls.forward

        def _decoder_forward_with_cache_compat(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
            if use_cache and past_key_values is None:
                try:
                    from transformers import DynamicCache
                    past_key_values = DynamicCache()
                except Exception:
                    pass

            return original_decoder_forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        decoder_cls.forward = _decoder_forward_with_cache_compat
        decoder_cls._sundial_forward_compat_patched = True

    if getattr(model_cls, "_sundial_sample_compat_patched", False):
        return

    if not hasattr(model_cls, "_extract_past_from_model_output"):
        def _extract_past_from_model_output(self, outputs, standardize_cache_format=False):
            if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
                return outputs.past_key_values
            if hasattr(outputs, "mems") and outputs.mems is not None:
                return outputs.mems
            if hasattr(outputs, "past_buckets_states") and outputs.past_buckets_states is not None:
                return outputs.past_buckets_states
            return None

        model_cls._extract_past_from_model_output = _extract_past_from_model_output

    def _legacy_sample(
        self,
        input_ids,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus=False,
        streamer=None,
        **model_kwargs,
    ):
        past_key_values = model_kwargs.get("past_key_values")
        if past_key_values is not None:
            try:
                if past_key_values.get_seq_length() == 0:
                    model_kwargs["past_key_values"] = None
            except Exception:
                pass

        return self._greedy_search(
            input_ids=input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            max_length=None,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_attentions=generation_config.output_attentions,
            output_hidden_states=generation_config.output_hidden_states,
            output_scores=generation_config.output_scores,
            output_logits=getattr(generation_config, "output_logits", False),
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    model_cls._sample = _legacy_sample
    model_cls._sundial_sample_compat_patched = True


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

    _patch_dynamic_cache_compatibility()

    model = _MODEL_CACHE.get(device)
    if model is None:
        print(f"Loading Sundial model once for device={device} from: {LOCAL_MODEL_DIR}")
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_DIR,
            trust_remote_code=True,
        ).to(device)
        _patch_sundial_generation_compatibility(model)
        model.eval()
        _MODEL_CACHE[device] = model
    else:
        _patch_sundial_generation_compatibility(model)

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
