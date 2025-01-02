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
from AEDataset import AEDataset, init_ae_dataset
from torch.utils.data import DataLoader


def main():
    # Load a small subset of the dataset (e.g., 10 samples)
    base_dataset = load_dataset(DATASET_REPO, split=DATASET_SPLIT)

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

        _, dataloader = init_ae_dataset(
            base_dataset,
            indices=indices,
            shuffle=False,
            process_on_demand=True,
        )

        if not os.path.exists(os.path.join(SAVE_DIR_CKPT, "denoise_dataset")):
            os.mkdir(os.path.join(SAVE_DIR_CKPT, "denoise_dataset"))

        count = 0

        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                count += len(batch)
                print(f"Processing batch {idx} @ {count}")
                # Collect images for display
                clean_imgs = torch.empty(0)
                recon_imgs = torch.empty(0)
                residuals = torch.empty(0)
                image = batch.to(DEVICE)
                recon_img, _ = model(image)
                residual = image - recon_img

                image = image.cpu()
                recon_img = recon_img.cpu()
                residual = residual.cpu()

                clean_imgs = torch.cat((clean_imgs, image), dim=0)
                recon_imgs = torch.cat((recon_imgs, recon_img), dim=0)
                residuals = torch.cat((residuals, residual), dim=0)
                # Save the images
                torch.save(
                    {
                        "clean": clean_imgs,
                        "recon": recon_imgs,
                        "residual": residuals,
                    },
                    os.path.join(
                        SAVE_DIR_CKPT,
                        "denoise_dataset",
                        f"dataset_{ckpt_file}_{idx}.pth",
                    ),
                )


if __name__ == "__main__":
    main()
