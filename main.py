# You can run this script to:
# 1) Train an Autoencoder
# 2) Train a Denoising network
# 3) Test JPEG/PNG compression efficiency

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import os
from AEDataset import init_ae_dataset
from DenoisingDataset import init_denoising_dataset
from autoenc import train_ae_model, inference_ae_model, plot_ae_model
from denoise import train_denoise_model, inference_denoise_model, plot_denoise_results
from jpegpng import run_compression_experiment
from config import *


def main():
    # Set environment for Hugging Face Hub
    os.environ["HF_HUB_OFFLINE"] = "1"

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = load_dataset("GATE-engine/mini_imagenet", split="train")
    ae_dataset, ae_dataloader = init_ae_dataset(
        dataset,
        length=64,
    )

    # 2) Train Autoencoder
    autoencoder_model, avg_loss = train_ae_model(
        dataloader=ae_dataloader,
        ENC_LAYERS=2,
        IMG_SET_SIZE=len(ae_dataset),
        LATENT_DIM=16,
        custom_dataset=ae_dataset,
    )

    # 3) Evaluate Autoencoder
    mean_similarity, avg_psnr, avg_ssim = inference_ae_model(
        autoencoder_model, ae_dataloader, num_examples=4
    )
    # Plot the results
    plot_ae_model(
        autoencoder_model,
        ae_dataset,
        ENC_LAYERS=2,
        IMG_SET_SIZE=len(ae_dataset),
        LATENT_DIM=16,
        avg_loss=avg_loss,
        mean_similarity=mean_similarity,
        avg_psnr=avg_psnr,
        avg_ssim=avg_ssim,
    )

    # 4) Prepare DenoisingDataset using the trained autoencoder
    denoise_dataset, denoise_dataloader = init_denoising_dataset(
        dataset=dataset,
        source_ae_model=autoencoder_model,
        indices=ae_dataset.indices,
    )

    # 5) Train DnCNN model
    denoise_model, losses = train_denoise_model(denoise_dataloader)

    # 6) Evaluate DnCNN model
    avg_mse, avg_psnr_denoise = inference_denoise_model(
        denoise_model, denoise_dataloader, num_examples=50
    )
    # Plot the results
    plot_denoise_results(denoise_model, denoise_dataset, num_samples=5)

    # 7) Test JPEG/PNG compression
    run_compression_experiment(ae_dataset, "compression_results.csv")


if __name__ == "__main__":
    main()
