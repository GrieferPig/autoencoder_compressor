import argparse
import os
import io
import torch
from PIL import Image
from torchvision import transforms
from datasets import load_dataset
from Autoencoder import Autoencoder
from config import DATASET_REPO, DATASET_SPLIT, DEVICE, ENC_IO_SIZE
from AEDataset import init_ae_dataset
from utils import load_autoenc_model


def main():
    parser = argparse.ArgumentParser(description="Generate a human evaluation set.")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to AE checkpoint."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="human_eval_set",
        help="Directory to save generated images.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Custom dataset path or use config default.",
    )
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality level.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the autoencoder model and get its indices
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    model = Autoencoder(
        image_size=ENC_IO_SIZE,
        num_layers=checkpoint["enc_layers"],
        latent_dim=checkpoint["latent_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)

    indices = checkpoint["indices"]

    model.eval()

    # If --dataset_path not provided, use defaults
    if args.dataset_path:
        base_dataset = args.dataset_path
    else:
        base_dataset = load_dataset(DATASET_REPO, split=DATASET_SPLIT)

    # Create AE dataset and dataloader
    custom_dataset, _ = init_ae_dataset(base_dataset, indices=indices)

    with torch.no_grad():
        for i in range(len(indices)):
            # Pick an index from the stored indices
            idx = indices[i % len(indices)]
            img_tensor = custom_dataset[i % len(indices)]
            img_pil_cpu = transforms.ToPILImage()(img_tensor.cpu())

            # Save clean image
            clean_path = os.path.join(args.output_dir, f"{i}_clean.png")
            transforms.ToPILImage()(img_pil_cpu).save(clean_path)

            # Compress as JPEG
            buffer = io.BytesIO()
            transforms.ToPILImage()(img_pil_cpu).save(buffer, format="JPEG", quality=80)
            buffer.seek(0)
            compressed_jpeg = Image.open(buffer).convert("RGB")
            jpeg_path = os.path.join(args.output_dir, f"{i}_jpeg.jpg")
            compressed_jpeg.save(jpeg_path)

            # Reconstruct with autoencoder
            # Convert compressed image back to tensor
            recon, _ = model(img_tensor.unsqueeze(0).to(DEVICE))
            recon_cpu = recon.squeeze(0).cpu()

            # Save reconstructed image
            recon_path = os.path.join(args.output_dir, f"{i}_reconstructed.png")
            transforms.ToPILImage()(recon_cpu).save(recon_path)

            print(
                f"Saved set {i} -> Clean: {clean_path}, JPEG: {jpeg_path}, Reconstructed: {recon_path}"
            )


if __name__ == "__main__":
    main()
