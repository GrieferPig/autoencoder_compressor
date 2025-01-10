import numpy as np
import torch
from AEDataset import AEDecoderDataset, init_ae_dataset
from Autoencoder import Autoencoder
from Autoencoder_quant import QuantizedAutoencoder
from config import DATASET_REPO, DATASET_SPLIT, ENC_IO_SIZE
import torch.nn as nn
import os
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
import onnxruntime as ort
from onnxruntime import quantization
from DnCNN import DenoisingModel

# Path to the model
model_path = "./results-imgnet/ckpt/final/ae_2_256_32.pth"
model_size = "base"
(enc_layers, img_set_size, latent_dim) = (
    model_path.split("/")[-1].split(".")[0].split("_")[1:4]
)
print(latent_dim)
model_name = model_path.split("/")[-1].split(".")[0]

# Load the model
model = Autoencoder(
    image_size=ENC_IO_SIZE, num_layers=int(enc_layers), latent_dim=int(latent_dim)
)
ckpt = torch.load(model_path, map_location="cpu")
indices = ckpt["indices"]
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

dummy_input = torch.randn(1, int(latent_dim))
normal_model = f"./benchmark/models/{model_size}.onnx"
torch.onnx.export(
    model.decoder,
    dummy_input,
    normal_model,
    input_names=["input"],
    output_names=["output"],
)

if not os.path.exists("./benchmark/models"):
    os.makedirs("./benchmark/models")

base_dataset = load_dataset(DATASET_REPO, split=DATASET_SPLIT)
calib_dataset, calib_loader = init_ae_dataset(base_dataset, indices=indices)

embeddings_dataset = AEDecoderDataset(calib_dataset, int(latent_dim), model, "cpu")


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


ort_provider = ["CPUExecutionProvider"]

ort_sess = ort.InferenceSession(normal_model, providers=ort_provider)

model_prep_path = "resnet18_prep.onnx"

quantization.shape_inference.quant_pre_process(
    normal_model, model_prep_path, skip_symbolic_shape=True
)


class QuntizationDataReader(quantization.CalibrationDataReader):
    def __init__(self, torch_ds, batch_size, input_name):

        self.torch_dl = torch.utils.data.DataLoader(
            torch_ds, batch_size=batch_size, shuffle=False
        )

        self.input_name = input_name
        self.datasize = len(self.torch_dl)

        self.enum_data = iter(self.torch_dl)

    def to_numpy(self, pt_tensor):
        return (
            pt_tensor.detach().cpu().numpy()
            if pt_tensor.requires_grad
            else pt_tensor.cpu().numpy()
        )

    def get_next(self):
        batch = next(self.enum_data, None)
        if batch is not None:
            return {self.input_name: np.expand_dims(self.to_numpy(batch[0]), axis=0)}
        else:
            return None

    def rewind(self):
        self.enum_data = iter(self.torch_dl)


qdr = QuntizationDataReader(
    embeddings_dataset, batch_size=64, input_name=ort_sess.get_inputs()[0].name
)

q_static_opts = {"ActivationSymmetric": True, "WeightSymmetric": True}

model_int8_path = f"./benchmark/models/{model_size}_int8.onnx"
quantized_model = quantization.quantize_static(
    model_input=model_prep_path,
    model_output=model_int8_path,
    calibration_data_reader=qdr,
    extra_options=q_static_opts,
)


# export denoise model
# model_path = "./dncnn_final.pth"

# model = DenoisingModel()

# ckpt = torch.load(model_path, map_location="cpu")
# print(ckpt)
# model.load_state_dict(ckpt)
# model.eval()

# dummy_input = torch.randn(1, 3, 256, 256)
# normal_model = f"./benchmark/models/dncnn.onnx"
# torch.onnx.export(
#     model,
#     dummy_input,
#     normal_model,
#     input_names=["input"],
#     output_names=["output"],
# )
