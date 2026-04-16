import torch
from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="thuml/sundial-base-128m",
    local_dir="./sundial-base-128m",
    proxies={
        "http": "http://127.0.0.1:7890",
        "https": "http://127.0.0.1:7890",
    }
)
#
# model = AutoModelForCausalLM.from_pretrained(
#     "./sundial_base_128m",
#     trust_remote_code=True
# )

# load pretrain model
# supports different lookback/forecast lengths
# model = AutoModelForCausalLM.from_pretrained('thuml/sundial-base-128m', trust_remote_code=True)


# prepare input
batch_size, lookback_length = 1, 2880
seqs = torch.randn(batch_size, lookback_length)

# Note that Sundial can generate multiple probable predictions
forecast_length = 96
num_samples = 20

output = model.generate(seqs, max_new_tokens=forecast_length, num_samples=num_samples)

# use raw predictions for mean/quantiles/confidence-interval estimation
print(output.shape)
print(output)