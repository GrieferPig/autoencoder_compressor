import argparse
import torch
import torch.nn.functional as F
from AEDataset import init_ae_dataset
from utils import calculate_psnr_ssim
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from config import DATASET_REPO, DATASET_SPLIT, DEVICE, ENC_IO_SIZE
from Autoencoder import Autoencoder
from DnCNN import DenoisingModel
from datasets import load_dataset


def load_ae_model(ae_checkpoint_path):
    checkpoint = torch.load(ae_checkpoint_path, map_location=DEVICE)
    enc_layers = checkpoint["enc_layers"]
    latent_dim = checkpoint["latent_dim"]
    print(f"enc_layers: {enc_layers}, latent_dim: {latent_dim}")
    model = Autoencoder(
        image_size=ENC_IO_SIZE,
        num_layers=enc_layers,
        latent_dim=latent_dim,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["indices"]


def load_dncnn(dncnn_path="./dncnn_final.pth"):
    checkpoint = torch.load(dncnn_path, map_location=DEVICE)
    model = DenoisingModel().to(DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def plot_results(original, reconstructed, denoised):
    bs = original.size(0)
    fig, axes = plt.subplots(bs, 3, figsize=(9, 3 * bs))
    for i in range(bs):
        # compute mse/psnr/ssim for "After AE"
        mse_ae = F.mse_loss(reconstructed[i], original[i]).item()
        psnr_ae, ssim_ae = calculate_psnr_ssim(
            original[i].unsqueeze(0), reconstructed[i].unsqueeze(0)
        )

        # compute mse/psnr/ssim for "Denoised"
        mse_dn = F.mse_loss(denoised[i], original[i]).item()
        psnr_dn, ssim_dn = calculate_psnr_ssim(
            original[i].unsqueeze(0), denoised[i].unsqueeze(0)
        )

        axes[i, 0].imshow(original[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(reconstructed[i].permute(1, 2, 0).detach().cpu().numpy())
        axes[i, 1].set_title(
            f"After AE\nMSE:{mse_ae:.4f},PSNR:{psnr_ae:.2f},SSIM:{ssim_ae:.3f}"
        )
        axes[i, 1].axis("off")

        axes[i, 2].imshow(denoised[i].permute(1, 2, 0).detach().cpu().numpy())
        axes[i, 2].set_title(
            f"Denoised\nMSE:{mse_dn:.4f},PSNR:{psnr_dn:.2f},SSIM:{ssim_dn:.3f}"
        )
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ae_checkpoint", type=str, required=True, help="Path to AE checkpoint"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to images folder or single image",
    )
    parser.add_argument(
        "--num_images", type=int, default=4, help="Number of images to test"
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path if args.dataset_path else DATASET_REPO
    ae_model, indices = load_ae_model(args.ae_checkpoint)
    dncnn_model = load_dncnn("./dncnn_final.pth")

    # Load images from dataset
    if dataset_path == DATASET_REPO:
        base_dataset = load_dataset(DATASET_REPO, split=DATASET_SPLIT)
    else:
        base_dataset = dataset_path

    dataset, _ = init_ae_dataset(
        base_dataset,
        indices=indices,
        process_on_demand=True,
    )

    original_imgs = torch.empty(0).to(DEVICE)
    # select 4 random images
    for _ in range(args.num_images):
        idx = torch.randint(0, len(dataset), (1,)).item()
        img = dataset[idx]
        original_imgs = torch.cat((original_imgs, img.unsqueeze(0).to(DEVICE)), dim=0)

    # Pass through AE
    with torch.no_grad():
        reconstructed, _ = ae_model(original_imgs)

    # Pass through denoiser
    with torch.no_grad():
        denoised_residual = dncnn_model(reconstructed)
        denoised = reconstructed + denoised_residual

    # Plot
    plot_results(original_imgs, reconstructed, denoised)


if __name__ == "__main__":
    main()
