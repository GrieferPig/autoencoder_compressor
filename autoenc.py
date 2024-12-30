# Imports
from config import *
from utils import calculate_psnr_ssim, save_autoenc_model
import torch
import random
from sklearn.metrics.pairwise import cosine_similarity
from Autoencoder import Autoencoder
from AEDataset import AEDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pytorch_msssim import ssim


def train_ae_model(dataloader, ENC_LAYERS, IMG_SET_SIZE, LATENT_DIM, custom_dataset):
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

    # Calculate the number of epochs based on dataset size and encoder layers
    real_epoch = int(
        EPOCHS_BASE_AE * max(1, IMG_SET_SIZE / 64) * max(1, ENC_LAYERS - 1)
    )
    print(f"Real Epoch: {real_epoch}")

    # Training loop
    model.train()
    for epoch in range(real_epoch):
        epoch_loss = 0
        for batch in dataloader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon_batch, _ = model(batch)
            loss = 1 - ssim(batch, recon_batch, data_range=255, size_average=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.3f}")

        # Save the model checkpoint every SAVE_PER_EPOCH_DENOISE
        # Skip the final epoch
        if (epoch + 1) % SAVE_PER_EPOCH_AE == 0 and epoch != real_epoch - 1:
            save_autoenc_model(
                epoch,
                avg_loss,
                model.state_dict(),
                ENC_LAYERS,
                IMG_SET_SIZE,
                LATENT_DIM,
                custom_dataset.indices,
                save_dir=SAVE_DIR,
            )

    # Save the trained model
    save_autoenc_model(
        real_epoch,
        avg_loss,
        model.state_dict(),
        ENC_LAYERS,
        IMG_SET_SIZE,
        LATENT_DIM,
        custom_dataset.indices,
        save_dir=SAVE_DIR,
        final=True,
    )

    return model, avg_loss


def inference_ae_model(model, dataloader, num_examples=100):
    """
    Evaluates the trained model on the dataset.

    Args:
        model (Autoencoder): Trained Autoencoder model.
        dataloader (DataLoader): DataLoader for evaluation data.
        num_examples (int, optional): Number of examples to evaluate. Defaults to 100.

    Returns:
        mean_similarity (float): Mean Cosine Similarity.
        avg_psnr (float): Average PSNR.
        avg_ssim (float): Average SSIM.
    """
    model.eval()
    cos_sim_total = 0
    count = 0
    avg_psnr = 0
    avg_ssim = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(DEVICE)
            recon_batch, _ = model(batch)
            batch_psnr, batch_ssim = calculate_psnr_ssim(batch, recon_batch)
            avg_psnr += batch_psnr
            avg_ssim += batch_ssim

            for i in range(len(batch)):
                original_flat = batch[i].view(-1).cpu().numpy().reshape(1, -1)
                reconstructed_flat = (
                    recon_batch[i].view(-1).cpu().numpy().reshape(1, -1)
                )
                similarity = cosine_similarity(original_flat, reconstructed_flat)[0][0]
                cos_sim_total += similarity
                count += 1
                if count >= num_examples:
                    break
            if count >= num_examples:
                break

    mean_similarity = cos_sim_total / num_examples
    avg_psnr /= len(dataloader)
    avg_ssim /= len(dataloader)

    return mean_similarity, avg_psnr, avg_ssim


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
        original = originals[i]
        reconstructed = reconstructeds[i]

        psnr, ssim_val = calculate_psnr_ssim(original, reconstructed)

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


# Example Usage
# if __name__ == "__main__":
#     # Initialize dataset and dataloader
#     custom_dataset = AEDataset(...)
#     dataloader = DataLoader(custom_dataset, batch_size=BATCH_SIZE, shuffle=True)

#     # Train the model
#     model, avg_loss = train_ae_model(
#         dataloader, ENC_LAYERS, IMG_SET_SIZE, LATENT_DIM, custom_dataset
#     )

#     # Perform inference
#     mean_similarity, avg_psnr, avg_ssim = inference_ae_model(
#         model, dataloader, num_examples=100
#     )

#     # Plot the results
#     plot_ae_model(
#         model,
#         custom_dataset,
#         ENC_LAYERS,
#         IMG_SET_SIZE,
#         LATENT_DIM,
#         avg_loss,
#         mean_similarity,
#         avg_psnr,
#         avg_ssim,
#     )
