# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

from DenoisingDataset import DenoisingDataset
from DnCNN import DenoisingModel
from config import (
    DEVICE,
    SAVE_PER_EPOCH_DENOISE,
    LR_DENOISE,
    EPOCHS_DENOISE,
    SAVE_DIR_DENOISE,
    NUM_SAMPLES_DENOISE,
)


def train_denoise_model(dataloader, save_dir=SAVE_DIR_DENOISE):
    """
    Trains the DnCNN denoising model.

    Args:
        dataloader (DataLoader): DataLoader for training data.
        save_dir (str): Directory to save model checkpoints.

    Returns:
        model (DenoisingModel): Trained DnCNN model.
        training_losses (list): List of average losses per epoch.
    """
    # Initialize the DnCNN model
    model = DenoisingModel().to(DEVICE)

    # Define Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR_DENOISE)

    training_losses = []

    # Training Loop
    model.train()
    for epoch in range(1, EPOCHS_DENOISE + 1):
        epoch_loss = 0.0
        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch}/{EPOCHS_DENOISE}", leave=False
        )
        for noisy_imgs, residual in progress_bar:
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
        print(f"Epoch [{epoch}/{EPOCHS_DENOISE}], Loss: {avg_loss:.6f}")

        # Optionally, save the model checkpoint
        if epoch % SAVE_PER_EPOCH_DENOISE == 0:
            checkpoint_path = f"{save_dir}/dncnn_epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

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
        for noisy_imgs, clean_imgs in dataloader:
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


def plot_denoise_results(model, dataset, num_samples=NUM_SAMPLES_DENOISE):
    """
    Plots noisy, denoised, and clean images for visual comparison.

    Args:
        model (DenoisingModel): Trained DnCNN model.
        dataset (DenoisingDataset): The dataset object.
        num_samples (int, optional): Number of samples to plot. Defaults to NUM_SAMPLES_DENOISE.
    """
    model.eval()
    examples = random.sample(range(len(dataset)), num_samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = [axes]  # Ensure axes is iterable

    with torch.no_grad():
        for i, idx in enumerate(examples):
            noisy_img, clean_img = dataset[idx]
            noisy_img_tensor = noisy_img.unsqueeze(0).to(DEVICE)
            denoised_residual = model(noisy_img_tensor)
            denoised_img = noisy_img_tensor - denoised_residual
            denoised_img = denoised_img.squeeze(0).cpu()

            noisy_img = noisy_img.cpu()
            clean_img = clean_img.cpu()

            # Plot Noisy Image
            ax = axes[i][0] if num_samples > 1 else axes[0]
            ax.imshow(noisy_img.permute(1, 2, 0).numpy())
            ax.set_title("Noisy Image")
            ax.axis("off")

            # Plot Denoised Image
            ax = axes[i][1] if num_samples > 1 else axes[1]
            ax.imshow(denoised_img.permute(1, 2, 0).numpy())
            ax.set_title("Denoised Image")
            ax.axis("off")

            # Plot Clean Image
            ax = axes[i][2] if num_samples > 1 else axes[2]
            ax.imshow(clean_img.permute(1, 2, 0).numpy())
            ax.set_title("Clean Image")
            ax.axis("off")

    plt.tight_layout()
    plt.show()


# Example Usage
# if __name__ == "__main__":
#     # Initialize dataset and dataloader
#     denoising_dataset = DenoisingDataset(...)  # Provide necessary arguments
#     dataloader = DataLoader(
#         denoising_dataset, batch_size=32, shuffle=True, num_workers=4
#     )

#     # Train the model
#     trained_model, losses = train_denoise_model(dataloader)

#     # Perform inference
#     avg_mse, avg_psnr = inference_denoise_model(
#         trained_model, dataloader, num_examples=100
#     )

#     # Plot the results
#     plot_denoise_results(trained_model, denoising_dataset, num_samples=5)
