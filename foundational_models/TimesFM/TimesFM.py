import torch
from transformers import Timesfm2P5ModelForPrediction

model = TimesFm2_5ModelForPrediction.from_pretrained("google/timesfm-2.5-200m-transformers")
model = model.to(torch.float32).eval()

past_values = [
    torch.linspace(0, 1, 100),
    torch.sin(torch.linspace(0, 20, 67)),
]

with torch.no_grad():
    outputs = model(past_values=past_values, forecast_context_len=1024)

print(outputs.mean_predictions.shape)
print(outputs.full_predictions.shape)
