import torch
import os

# Default save directory
SAVE_DIR = "./results"

# Dataset configuration
DATASET_REPO = "GATE-engine/mini_imagenet"
DATASET_SPLIT = "train"

# Autoencoder settings
# (num_layers, image_set_size, latent_dim)
CUSTOM_TRAINING_CONFIG_SET = [
    (2, 64, 16),
    (2, 64, 32),
    (2, 64, 64),
    (2, 128, 16),
    (2, 128, 32),
    (2, 128, 64),
    (2, 256, 16),
    (2, 256, 32),
    (2, 256, 64),
]
SAVE_DIR_CKPT = os.path.join(SAVE_DIR, "ckpt")
SAVE_DIR_FIGURES = os.path.join(SAVE_DIR, "figures")
ENC_IO_SIZE = 256
BATCH_SIZE = 6000
EPOCHS_BASE_AE = 1500
LR_AE = 1e-4
NUM_SAMPLES = 4
SAVE_PER_EPOCH_AE = 1000

# Denoise settings
SAVE_DIR_DENOISE_BASE = os.path.join(SAVE_DIR, "denoise")
SAVE_DIR_DENOISE_CKPT = os.path.join(SAVE_DIR_DENOISE_BASE, "ckpt")
SAVE_DIR_DENOISE_FIGURES = os.path.join(SAVE_DIR_DENOISE_BASE, "figures")
SAVE_FILENAME_GAUSSIAN = "dncnn_gaussian.pth"
LR_DENOISE = 1e-4
EPOCHS_DENOISE = 100
SAVE_PER_EPOCH_DENOISE = 10
NUM_SAMPLES_DENOISE = 4

# Default device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Compression settings
JPEG_QUALITY_FACTORS = list(range(0, 101, 10))  # 0 to 100 with step 10
PNG_COMPRESSION_LEVELS = list(range(0, 10, 1))  # 0 to 9 with step 1
SAVE_DIR_COMPRESSION = os.path.join(SAVE_DIR, "compression")

# Denoise dataset settings
BATCH_SIZE_DENOISE_DATA_INFERENCE = 32
BATCH_SIZE_DENOISE_DATA = 4

# Autoencoder dataset settings
BATCH_SIZE_AE_DATA = 64
