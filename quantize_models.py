import torch
from Autoencoder import Autoencoder
from Autoencoder_quant import QuantizedAutoencoder
from config import ENC_IO_SIZE
import torch.nn as nn
import os
from torch.utils.data import DataLoader, TensorDataset

# Path to the model
model_path = "./results-imgnet/ckpt/final/ae_2_64_16.pth"
(enc_layers, img_set_size, latent_dim) = (
    model_path.split("/")[-1].split(".")[0].split("_")[1:4]
)
model_name = model_path.split("/")[-1].split(".")[0]

# Load the model
model = Autoencoder(
    image_size=ENC_IO_SIZE, num_layers=int(enc_layers), latent_dim=int(latent_dim)
)
model_float16 = QuantizedAutoencoder(
    image_size=ENC_IO_SIZE, num_layers=int(enc_layers), latent_dim=int(latent_dim)
)
model.load_state_dict(torch.load(model_path, map_location="cpu")["model_state_dict"])
model_float16.load_state_dict(
    torch.load(model_path, map_location="cpu")["model_state_dict"]
)
model.eval()
model_float16.eval()

dummy_input = torch.randn(1, 16)
torch.onnx.export(
    model.decoder,
    dummy_input,
    "./quantized_models/autoencoder.onnx",
    input_names=["input"],
    output_names=["output"],
)

if not os.path.exists("./quantized_models"):
    os.makedirs("./quantized_models")


# Perform float16 quantization
model_float16 = model.to(torch.float16)

# export the quantized int8 and float16 model and the original model as onnx
torch.onnx.export(
    model_float16.decoder,
    dummy_input.to(torch.float16),
    "./quantized_models/autoencoder_float16.onnx",
    input_names=["input"],
    output_names=["output"],
)
