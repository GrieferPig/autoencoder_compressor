import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from config import (
    SAVE_DIR_FIGURES,
    SAVE_DIR_CKPT,
    DEVICE,
    ENC_IO_SIZE,
    DATASET_REPO,
    DATASET_SPLIT,
)
from utils import load_autoenc_model
from datasets import load_dataset
from AEDataset import AEDataset
from torch.utils.data import DataLoader


def main():
    # Load a small subset of the dataset (e.g., 10 samples)
    base_dataset = load_dataset(DATASET_REPO, split=DATASET_SPLIT)
    ae_dataset = AEDataset(base_dataset, shuffle=False)

    checkpoint_dir = os.path.join(SAVE_DIR_CKPT, "final")
    if not os.path.isdir(checkpoint_dir):
        print(f"No checkpoint directory found at {checkpoint_dir}")
        return

    # Find all final checkpoints with enc_layers=2 in the filename: ae_2_IMGSET_LATENT.pth
    checkpoint_files = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith("ae_2_") and f.endswith(".pth")
    ]
    if not checkpoint_files:
        print("No AE models with enc_layers=2 found.")
        return

    # For each such model, load and run inference on the 10 images,
    # producing clean (original), reconstructed ("noisy") and residual images.
    for ckpt_file in checkpoint_files:
        print(f"Using checkpoint: {ckpt_file}")
        # Filename format: ae_2_{IMGSET}_{LATENT}.pth
        parts = ckpt_file.split(".")[0].split("_")  # ["ae","2","imgset","latent"]
        if len(parts) < 4:
            continue
        try:
            enc_layers = int(parts[1])  # should be 2
            img_set_size = int(parts[2])
            latent_dim = int(parts[3])
        except ValueError:
            continue

        # Load the AE model
        model, (_, _, indices) = load_autoenc_model(
            enc_layers, img_set_size, latent_dim
        )
        model.eval()

        # Collect images for display
        clean_imgs = torch.empty(0)
        recon_imgs = torch.empty(0)
        residuals = torch.empty(0)

        with torch.no_grad():
            for indice in indices:
                image = ae_dataset[indice].unsqueeze(0).to(DEVICE)
                recon_img, _ = model(image)
                residual = torch.abs(image - recon_img)

                clean_imgs = torch.cat((clean_imgs, image), dim=0)
                recon_imgs = torch.cat((recon_imgs, recon_img), dim=0)
                residuals = torch.cat((residuals, residual), dim=0)

        length = 10

        # Plot all 10 results in a 3-row layout (clean, recon, residual)
        fig, axes = plt.subplots(nrows=3, ncols=length, figsize=(length * 3, 9))
        fig.suptitle(
            f"AE Model: enc_layers=2, img_set_size={img_set_size}, latent_dim={latent_dim}"
        )

        for i in range(length):
            # Clean
            ax = axes[0, i] if length > 1 else axes[0]
            np_img = np.transpose(clean_imgs[i].numpy(), (1, 2, 0))
            np_img = np.clip(np_img, 0, 1)
            ax.imshow(np_img)
            ax.set_title("Clean")
            ax.axis("off")

            # Recon
            ax = axes[1, i] if length > 1 else axes[1]
            np_img = np.transpose(recon_imgs[i].numpy(), (1, 2, 0))
            np_img = np.clip(np_img, 0, 1)
            ax.imshow(np_img)
            ax.set_title("Reconstructed")
            ax.axis("off")

            # Residual
            ax = axes[2, i] if length > 1 else axes[2]
            np_img = np.transpose(residuals[i].numpy(), (1, 2, 0))
            # Rescale residual for visualization
            res_min, res_max = np_img.min(), np_img.max()
            if abs(res_max - res_min) > 1e-9:
                np_img = (np_img - res_min) / (res_max - res_min)
            ax.imshow(np_img)
            ax.set_title("Residual")
            ax.axis("off")

        # Show and optionally save
        plt.tight_layout()
        plt.show()
        # If desired, one could save each figure:
        # plt.savefig(os.path.join(SAVE_DIR_FIGURES, f"inference_{ckpt_file}.png"))
        plt.close(fig)


if __name__ == "__main__":
    main()
