import os
import torch
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from config import *
from Autoencoder import Autoencoder
from DnCNN import DenoisingModel


def save_autoenc_model(
    epoch,
    loss,
    model,
    optimizer,
    ENC_LAYERS,
    IMG_SET_SIZE,
    LATENT_DIM,
    indices,
    save_dir=SAVE_DIR_CKPT,
    final=False,
    convergence=False,
):
    """
    Save the Autoencoder model checkpoint to a specified directory.

    Parameters:
    - epoch (int): The current epoch number.
    - loss (float): The loss value at the current epoch.
    - model (torch.nn.Module): The model to be saved.
    - ENC_LAYERS (int): The number of encoder layers in the model.
    - IMG_SET_SIZE (int): The size of the image set.
    - LATENT_DIM (int): The dimension of the latent space.
    - indices (list): List of indices used in the model.
    - save_dir (str, optional): The directory where the checkpoint will be saved. Defaults to "./ckpt".
    - final (bool, optional): Whether the model is the final model. Defaults to False

    Returns:
    - None
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(f"{save_dir}/final")

    print(SAVE_DIR_CKPT)

    if final:
        if convergence:
            save_path = f"{save_dir}/final/ae_{ENC_LAYERS}_{IMG_SET_SIZE}_{LATENT_DIM}_convergence.pth"
        save_path = f"{save_dir}/final/ae_{ENC_LAYERS}_{IMG_SET_SIZE}_{LATENT_DIM}.pth"
    else:
        save_path = (
            f"{save_dir}/ae_{epoch}_{ENC_LAYERS}_{IMG_SET_SIZE}_{LATENT_DIM}.pth"
        )
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model,
            "optimizer_state_dict": optimizer,
            "loss": loss,
            "enc_layers": ENC_LAYERS,
            "img_set_size": IMG_SET_SIZE,
            "latent_dim": LATENT_DIM,
            "indices": indices,
        },
        save_path,
    )
    print(f"Autoencoder model saved at {save_path}")


def load_autoenc_model(enc_layers, img_set_size, latent_dim, load_optimizer=False):
    """
    Load the Autoencoder model checkpoint from a specified path.

    Parameters:
    - checkpoint_path (str): The path to the checkpoint file.
    - model (torch.nn.Module): The model to load the checkpoint into.
    - optimizer (torch.optim.Optimizer, optional): The optimizer to load the checkpoint into. Defaults to None.

    Returns:
    - model (torch.nn.Module): The model with the checkpoint loaded.
    - optimizer (torch.optim.Optimizer): The optimizer with the checkpoint loaded.
    - (epoch, loss, indices) (tuple): The epoch, loss, and indices from the checkpoint
    """
    checkpoint_path = (
        f"{SAVE_DIR_CKPT}/final/ae_{enc_layers}_{img_set_size}_{latent_dim}.pth"
    )
    print(f"Loading autoencoder model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model = Autoencoder(
        image_size=ENC_IO_SIZE,
        num_layers=checkpoint["enc_layers"],
        latent_dim=checkpoint["latent_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    if load_optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_AE)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return (
            model,
            optimizer,
            (checkpoint["epoch"], checkpoint["loss"], checkpoint["indices"]),
        )

    return model, (checkpoint["epoch"], checkpoint["loss"], checkpoint["indices"])


def save_denoise_model(
    model,
    optimizer,
    epoch,
    loss,
    save_dir=SAVE_DIR_DENOISE_CKPT,
    gaussian_noise_model=True,
    enc_layers=None,
    img_set_size=None,
    latent_dim=None,
):
    """
    Save the DnCNN model checkpoint to a specified directory.

    Parameters:
    - model (torch.nn.Module): The model to be saved.
    - optimizer (torch.optim.Optimizer): The optimizer to be saved.
    - epoch (int): The current epoch number.
    - loss (float): The loss value at the current epoch.
    - save_dir (str, optional): The directory where the checkpoint will be saved. Defaults to "./ckpt".

    Returns:
    - None
    """
    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")

    if gaussian_noise_model:
        save_path = f"{save_dir}/{SAVE_FILENAME_GAUSSIAN}"
    elif enc_layers and img_set_size and latent_dim:
        save_path = (
            f"{save_dir}/dncnn_final_{enc_layers}_{img_set_size}_{latent_dim}.pth"
        )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "enc_layers": enc_layers,
            "img_set_size": img_set_size,
            "latent_dim": latent_dim,
            "gaussian": gaussian_noise_model,
        },
        save_path,
    )
    print(f"Denoiser model saved at {save_path}")


def load_denoise_model(
    gaussian_noise_model=False,
    enc_layers=None,
    img_set_size=None,
    latent_dim=None,
    load_optimizer=False,
):
    if gaussian_noise_model:
        checkpoint_path = f"{SAVE_DIR_DENOISE_CKPT}/{SAVE_FILENAME_GAUSSIAN}"
    elif enc_layers and img_set_size and latent_dim:
        checkpoint_path = f"{SAVE_DIR_DENOISE_CKPT}/dncnn_final_{enc_layers}_{img_set_size}_{latent_dim}.pth"
    else:
        raise ValueError("Please provide valid model configuration.")

    print(f"Loading denoiser model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model = DenoisingModel().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    if load_optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_DENOISE)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return model, optimizer

    return model


def calculate_psnr_ssim(original, reconstructed):
    """
    Calculates PSNR and SSIM between two batches of images.

    Parameters:
    - original (torch.Tensor): Original images, shape (B, C, H, W)
    - reconstructed (torch.Tensor): Reconstructed images, shape (B, C, H, W)

    Returns:
    - avg_psnr (float): Average PSNR over the batch
    - avg_ssim (float): Average SSIM over the batch
    """
    original = original.cpu().numpy()
    reconstructed = reconstructed.cpu().detach().numpy()

    batch_size = original.shape[0]
    psnr_total = 0.0
    ssim_total = 0.0

    for i in range(batch_size):
        # Convert from (C, H, W) to (H, W, C)
        orig_img = np.transpose(original[i], (1, 2, 0))
        recon_img = np.transpose(reconstructed[i], (1, 2, 0))

        # Ensure the images are in [0, 1] range
        orig_img = np.clip(orig_img, 0, 1)
        recon_img = np.clip(recon_img, 0, 1)

        psnr = compare_psnr(orig_img, recon_img, data_range=1.0)
        ssim = compare_ssim(
            orig_img, recon_img, multichannel=True, data_range=1.0, channel_axis=2
        )

        psnr_total += psnr
        ssim_total += ssim

    avg_psnr = psnr_total / batch_size
    avg_ssim = ssim_total / batch_size

    return avg_psnr, avg_ssim


def get_model_size(model):
    # Specify the checkpoint path
    checkpoint_path = "./.temp.pth"

    # Save only the model's state_dict
    torch.save(model.state_dict(), checkpoint_path)

    # Get the file size in bytes
    file_size_bytes = os.path.getsize(checkpoint_path)

    file_size_mib = file_size_bytes / (1024**2)  # MiB

    # Remove the temporary file
    os.remove(checkpoint_path)

    return file_size_mib


# def run_automated_experiments():
#     results = []
#     parameter_combinations = CUSTOM_TRAINING_CONFIG_SET
#     for _, combo in enumerate(parameter_combinations, 1):
#         ENC_LAYERS = combo[0]
#         IMG_SET_SIZE = combo[1]
#         LATENT_DIM = combo[2]
#         print(
#             f"\nRunning experiment with ENC_LAYERS={ENC_LAYERS}, IMG_SET_SIZE={IMG_SET_SIZE}, LATENT_DIM={LATENT_DIM}"
#         )
#         _, final_loss, mean_similarity, avg_psnr, avg_ssim = train_ae_model(
#             ENC_LAYERS, IMG_SET_SIZE, LATENT_DIM
#         )
#         results.append(
#             {
#                 "ENC_LAYERS": ENC_LAYERS,
#                 "IMG_SET_SIZE": IMG_SET_SIZE,
#                 "LATENT_DIM": LATENT_DIM,
#                 "final_loss": final_loss,
#                 "mean_similarity": mean_similarity,
#                 "avg_psnr": avg_psnr,
#                 "avg_ssim": avg_ssim,
#             }
#         )
