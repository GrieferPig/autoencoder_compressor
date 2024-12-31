# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import os

from DenoisingDataset import DenoisingDataset
from DnCNN import DenoisingModel
from config import (
    DEVICE,
    SAVE_PER_EPOCH_DENOISE,
    LR_DENOISE,
    EPOCHS_DENOISE,
    SAVE_DIR_DENOISE_CKPT,
    SAVE_DIR_DENOISE_FIGURES,
    NUM_SAMPLES_DENOISE,
)
from utils import save_denoise_model


def train_denoise_model(
    dataloader,
    gaussian_noise_model=True,
    enc_layers=None,
    img_set_size=None,
    latent_dim=None,
    save_dir=SAVE_DIR_DENOISE_CKPT,
    epochs=EPOCHS_DENOISE,
):
    """
    Trains the DnCNN denoising model.

    Args:
        dataloader (DataLoader): DataLoader for training data.
        save_dir (str): Directory to save model checkpoints.

    Returns:
        model (DenoisingModel): Trained DnCNN model.
        training_losses (list): List of average losses per epoch.
    """
    if not (gaussian_noise_model or (enc_layers and img_set_size and latent_dim)):
        raise ValueError("Please provide valid model configuration.")

    # Initialize the DnCNN model
    model = DenoisingModel().to(DEVICE)

    # Define Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR_DENOISE)

    training_losses = []

    # Training Loop
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for noisy_imgs, residual, _ in progress_bar:
            noisy_imgs = noisy_imgs.to(DEVICE)
            residual = residual.to(DEVICE)

            optimizer.zero_grad()
            output_residual = model(noisy_imgs)
            loss = criterion(output_residual, residual)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * noisy_imgs.size(0)
            progress_bar.set_postfix({"Loss": loss.item()})

        avg_loss = epoch_loss / len(dataloader.dataset)
        training_losses.append(avg_loss)
        print(f"Epoch [{epoch}/{epochs}], Loss: {avg_loss:.6f}")

        if epoch % SAVE_PER_EPOCH_DENOISE == 0:
            save_denoise_model(
                model,
                optimizer,
                epoch,
                avg_loss,
                save_dir,
                gaussian_noise_model=gaussian_noise_model,
                enc_layers=enc_layers,
                img_set_size=img_set_size,
                latent_dim=latent_dim,
            )

    return model, training_losses


def inference_denoise_model(model, dataloader, num_examples=100):
    """
    Evaluates the trained DnCNN model on the dataset.

    Args:
        model (DenoisingModel): Trained DnCNN model.
        dataloader (DataLoader): DataLoader for evaluation data.
        num_examples (int, optional): Number of examples to evaluate. Defaults to 100.

    Returns:
        avg_mse (float): Average Mean Squared Error over the evaluated examples.
        avg_psnr (float): Average Peak Signal-to-Noise Ratio over the evaluated examples.
    """
    model.eval()
    mse_loss = nn.MSELoss(reduction="mean")
    total_mse = 0.0
    total_psnr = 0.0
    count = 0

    with torch.no_grad():
        for noisy_imgs, _, clean_imgs in dataloader:
            noisy_imgs = noisy_imgs.to(DEVICE)
            clean_imgs = clean_imgs.to(DEVICE)

            denoised_residual = model(noisy_imgs)
            denoised_imgs = noisy_imgs - denoised_residual

            # Compute MSE
            mse = mse_loss(denoised_imgs, clean_imgs)
            total_mse += mse.item()

            # Compute PSNR
            mse_np = mse.item()
            if mse_np == 0:
                psnr = float("inf")
            else:
                psnr = (
                    10 * torch.log10(1 / mse).item()
                )  # Assuming images are normalized between 0 and 1
            total_psnr += psnr

            count += 1
            if count >= num_examples:
                break

    avg_mse = total_mse / count
    avg_psnr = total_psnr / count

    print(
        f"Inference Results - Average MSE: {avg_mse:.6f}, Average PSNR: {avg_psnr:.2f} dB"
    )
    return avg_mse, avg_psnr


def plot_denoise_results(
    model,
    dataset,
    gaussian_noise_model=True,
    enc_layers=None,
    img_set_size=None,
    latent_dim=None,
    num_samples=NUM_SAMPLES_DENOISE,
):
    """
    Plots noisy, denoised, residual, and clean images for visual comparison.

    Args:
        model (DenoisingModel): Trained DnCNN model.
        dataset (DenoisingDataset): The dataset object.
        num_samples (int, optional): Number of samples to plot. Defaults to NUM_SAMPLES_DENOISE.
    """
    if gaussian_noise_model:
        figure_path = os.path.join(
            SAVE_DIR_DENOISE_FIGURES, "denoise_results_gaussian.png"
        )
    elif enc_layers and img_set_size and latent_dim:
        figure_path = os.path.join(
            SAVE_DIR_DENOISE_FIGURES,
            f"denoise_results_{enc_layers}_{img_set_size}_{latent_dim}.png",
        )
    else:
        raise ValueError("Please provide valid model configuration.")

    model.eval()
    examples = random.sample(range(len(dataset)), num_samples)

    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    if num_samples == 1:
        axes = [axes]

    with torch.no_grad():
        for i, idx in enumerate(examples):
            noisy_img, residual_img, clean_img = dataset[idx]
            noisy_img_tensor = noisy_img.unsqueeze(0).to(DEVICE)
            denoised_residual = model(noisy_img_tensor)
            denoised_img = noisy_img_tensor + denoised_residual

            denoised_img = denoised_img.squeeze(0).cpu()
            residual_img = denoised_residual.squeeze(0).cpu()
            # normalize residual image
            residual_img = (residual_img - residual_img.min()) / (
                residual_img.max() - residual_img.min()
            )
            noisy_img = noisy_img.cpu()
            clean_img = clean_img.cpu()

            ax = axes[i][0]
            ax.imshow(noisy_img.permute(1, 2, 0).numpy())
            ax.set_title("Noisy Image")
            ax.axis("off")

            ax = axes[i][1]
            ax.imshow(denoised_img.permute(1, 2, 0).numpy())
            ax.set_title("Denoised Image")
            ax.axis("off")

            ax = axes[i][2]
            ax.imshow(residual_img.permute(1, 2, 0).numpy())
            ax.set_title("Residual")
            ax.axis("off")

            ax = axes[i][3]
            ax.imshow(clean_img.permute(1, 2, 0).numpy())
            ax.set_title("Clean Image")
            ax.axis("off")

    plt.tight_layout()
    if not os.path.exists(SAVE_DIR_DENOISE_FIGURES):
        os.makedirs(SAVE_DIR_DENOISE_FIGURES)

    plt.savefig(figure_path)
    print(f"Results saved at: {figure_path}")
