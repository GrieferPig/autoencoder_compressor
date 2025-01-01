# Imports
from sklearn.linear_model import LinearRegression
from config import *
from utils import calculate_psnr_ssim, save_autoenc_model
import torch
import torch.nn as nn
import random
from sklearn.metrics.pairwise import cosine_similarity
from Autoencoder import Autoencoder
from AEDataset import AEDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pytorch_msssim import ssim, ms_ssim
from tqdm import tqdm
import numpy as np
from torch.amp import GradScaler, autocast


def train_ae_model(
    dataloader,
    ENC_LAYERS,
    IMG_SET_SIZE,
    LATENT_DIM,
    custom_dataset,
    epochs=EPOCHS_BASE_AE,
    till_convergence=False,
    use_ssim=False,
):
    """
    Trains the Autoencoder model.

    Args:
        dataloader (DataLoader): DataLoader for training data.
        ENC_LAYERS (int): Number of encoder layers.
        IMG_SET_SIZE (int): Size of the image dataset.
        LATENT_DIM (int): Dimension of the latent space.
        custom_dataset (AEDataset): The dataset object.

    Returns:
        model (Autoencoder): Trained Autoencoder model.
        avg_loss (float): Average training loss.
    """
    # Instantiate the model
    model = Autoencoder(
        image_size=ENC_IO_SIZE, num_layers=ENC_LAYERS, latent_dim=LATENT_DIM
    ).to(DEVICE)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_AE)
    scaler = GradScaler()
    if use_ssim:

        def ssim_loss(x, y):
            return 1 - ssim(
                x,
                y,
                data_range=1,
                size_average=True,
                nonnegative_ssim=True,
            )

        criterion = ssim_loss
    else:
        criterion = nn.MSELoss()

    if till_convergence:
        # set a large number of epochs to train until convergence
        epochs = 99999
        last_ssim = -float("inf")
        # forcefully train 500 epochs before testing for convergence to avoid noise recon
        patience = 500
        last_100_loss = np.empty(100)

    overview_loss = np.empty(epochs)
    overview_psnr = np.empty(epochs // 100)
    overview_ssim = np.empty(epochs // 100)

    # Training loop
    model.train()
    with tqdm(range(epochs), unit="epoch") as tepochs:
        for epoch in tepochs:
            epoch_loss = 0
            tepochs.set_description(f"Epoch {epoch+1}")
            for batch in dataloader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                with autocast(device_type=DEVICE):  # <-- Start autocast context
                    recon_batch, _ = model(batch)
                    loss = criterion(recon_batch, batch)

                # Scale the loss and call backward
                scaler.scale(loss).backward()  # <-- Scale loss and backward
                scaler.step(optimizer)  # <-- Scaled optimizer step
                scaler.update()  # <-- Update the scaler
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            last_100_loss[epoch % 100] = avg_loss
            overview_loss[epoch] = avg_loss
            tepochs.set_postfix(loss=f"{avg_loss*100:.4f} %")
            if (epoch + 1) % 100 == 0 and till_convergence:
                # do linear regression on last 100 losses
                # test if the slope is less than 0.01
                x = np.arange(len(last_100_loss)).reshape(-1, 1)
                y = np.array(last_100_loss).reshape(-1, 1)
                reg = LinearRegression().fit(x, y)
                slope = abs(reg.coef_[0][0])
                # also test if ssim is within +-0.01 of last ssim
                _, avg_psnr, avg_ssim = inference_ae_model(model, dataloader)
                overview_psnr[epoch // 100] = avg_psnr
                overview_ssim[epoch // 100] = avg_ssim
                if (
                    abs(avg_ssim - last_ssim) < 0.005
                    and slope < 5e-6
                    and epoch > patience
                ):
                    print(
                        f"break at absolute ssim diff: {abs(avg_ssim - last_ssim)}, slope: {slope}"
                    )
                    break

                last_ssim = avg_ssim
                last_100_loss = np.empty(100)

            # Save the model checkpoint every SAVE_PER_EPOCH_DENOISE
            if (epoch + 1) % SAVE_PER_EPOCH_AE == 0:
                save_autoenc_model(
                    epoch + 1,
                    avg_loss,
                    model.state_dict(),
                    optimizer.state_dict(),
                    ENC_LAYERS,
                    IMG_SET_SIZE,
                    LATENT_DIM,
                    custom_dataset.indices,
                )

    # Save the trained model
    save_autoenc_model(
        epochs,
        avg_loss,
        model.state_dict(),
        optimizer.state_dict(),
        ENC_LAYERS,
        IMG_SET_SIZE,
        LATENT_DIM,
        custom_dataset.indices,
        final=True,
        convergence=till_convergence,
    )

    # store trend of loss, psnr and ssim
    if till_convergence:
        torch.save(
            {
                "loss": overview_loss,
                "psnr": overview_psnr,
                "ssim": overview_ssim,
            },
            f"{SAVE_DIR}/trend_{ENC_LAYERS}_{IMG_SET_SIZE}_{LATENT_DIM}.pth",
        )

    return model, avg_loss


def inference_ae_model(model, dataloader, num_examples=100):
    """
    Evaluates the trained model on the dataset using MSE.

    Args:
        model (Autoencoder): Trained Autoencoder model.
        dataloader (DataLoader): DataLoader for evaluation data.
        num_examples (int, optional): Number of examples to evaluate. Defaults to 100.

    Returns:
        mean_mse (float): Mean MSE.
        avg_psnr (float): Average PSNR.
        avg_ssim (float): Average SSIM.
    """
    model.eval()
    mse_total = 0

    with torch.no_grad():
        eval_set = torch.empty(0).to(DEVICE)
        for batch in dataloader:
            batch = batch.to(DEVICE)
            eval_set = torch.cat((eval_set, batch), dim=0)
            if len(eval_set) > num_examples:
                eval_set = eval_set[:num_examples]
                break

        recon_batch, _ = model(eval_set)
        avg_psnr, avg_ssim = calculate_psnr_ssim(eval_set, recon_batch)

        for i in range(len(eval_set)):
            original_flat = eval_set[i].view(-1).cpu().numpy()
            reconstructed_flat = recon_batch[i].view(-1).cpu().numpy()
            mse_val = ((original_flat - reconstructed_flat) ** 2).mean()
            mse_total += mse_val

    mean_mse = mse_total / num_examples
    return mean_mse, avg_psnr, avg_ssim


def plot_ae_model(
    model,
    custom_dataset,
    ENC_LAYERS,
    IMG_SET_SIZE,
    LATENT_DIM,
    avg_loss,
    mean_similarity,
    avg_psnr,
    avg_ssim,
):
    """
    Plots original and reconstructed images from the Autoencoder.

    Args:
        model (Autoencoder): Trained Autoencoder model.
        custom_dataset (AEDataset): The dataset object.
        ENC_LAYERS (int): Number of encoder layers.
        IMG_SET_SIZE (int): Size of the image dataset.
        LATENT_DIM (int): Dimension of the latent space.
        avg_loss (float): Average training loss.
        mean_similarity (float): Mean Cosine Similarity.
        avg_psnr (float): Average PSNR.
        avg_ssim (float): Average SSIM.
    """
    model.eval()
    originals = []
    reconstructeds = []

    for _ in range(NUM_SAMPLES):
        test_idx = random.randint(0, IMG_SET_SIZE - 1)
        test_data = custom_dataset[test_idx]
        with torch.no_grad():
            recon, _ = model(test_data.unsqueeze(0))

        originals.append(test_data.cpu())
        reconstructeds.append(recon.cpu().squeeze())

    plt.figure(figsize=(16, 9))
    plt.subplots_adjust(top=0.85)  # Adjust the top to make space for the text
    plt.suptitle(
        f"ENC_LAYERS: {ENC_LAYERS}, IMG_SET_SIZE: {IMG_SET_SIZE}, LATENT_DIM: {LATENT_DIM}\n"
        f"Final Loss: {avg_loss:.4f}, Mean Cosine Similarity: {mean_similarity:.4f}, Avg PSNR: {avg_psnr:.4f}, Avg SSIM: {avg_ssim:.4f}",
        y=0.98,
        fontsize=12,
    )

    for i in range(NUM_SAMPLES):
        original = originals[i].unsqueeze(0)
        reconstructed = reconstructeds[i].unsqueeze(0)

        psnr, ssim_val = calculate_psnr_ssim(original, reconstructed)

        original = original.squeeze()
        reconstructed = reconstructed.squeeze()

        # Plot Original Image
        ax = plt.subplot(2, NUM_SAMPLES, i + 1)
        img = original.permute(1, 2, 0).numpy()
        plt.imshow(img)
        ax.axis("off")
        ax.set_title(f"Original\nPSNR: {psnr:.4f}, SSIM: {ssim_val:.4f}")

        # Plot Reconstructed Image
        ax = plt.subplot(2, NUM_SAMPLES, i + 1 + NUM_SAMPLES)
        img = reconstructed.permute(1, 2, 0).numpy()
        plt.imshow(img)
        ax.axis("off")
        ax.set_title("Reconstructed")

    if not os.path.exists(SAVE_DIR_FIGURES):
        os.makedirs(SAVE_DIR_FIGURES)

    plt.savefig(
        f"{SAVE_DIR_FIGURES}/recon_{ENC_LAYERS}_{IMG_SET_SIZE}_{LATENT_DIM}.png"
    )
    plt.close()

    # Display the results
    print("\nExperiment Results:")
    print(
        f"ENC_LAYERS: {ENC_LAYERS}, IMG_SET_SIZE: {IMG_SET_SIZE}, LATENT_DIM: {LATENT_DIM}, "
        f"Final Loss: {avg_loss:.4f}, Mean Cosine Similarity: {mean_similarity:.4f}, "
        f"Avg PSNR: {avg_psnr:.4f}, Avg SSIM: {avg_ssim:.4f}"
    )
