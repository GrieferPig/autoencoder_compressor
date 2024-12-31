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
from pytorch_msssim import ssim, ms_ssim
from tqdm import tqdm


def train_ae_model(
    dataloader,
    ENC_LAYERS,
    IMG_SET_SIZE,
    LATENT_DIM,
    custom_dataset,
    epochs=EPOCHS_BASE_AE,
    till_convergence=False,
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

    if till_convergence:
        # set a large number of epochs to train until convergence
        epochs = 99999
        # set to 1 to avoid division by zero
        last_ssim = 1
        ssim_not_improved_count = 0
        # forcefully train 500 epochs before testing for convergence to avoid noise recon
        patience = 500

    # Training loop
    model.train()
    with tqdm(range(epochs), unit="epoch") as tepochs:
        for epoch in tepochs:
            epoch_loss = 0
            tepochs.set_description(f"Epoch {epoch+1}")
            for batch in dataloader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                recon_batch, _ = model(batch)
                loss = 1 - ssim(
                    batch,
                    recon_batch,
                    data_range=1,
                    size_average=True,
                    nonnegative_ssim=True,
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            tepochs.set_postfix(loss=f"{avg_loss*100:.4f} %")
            if (epoch + 1) % 10 == 0 and till_convergence and epoch > patience:
                # test the relative difference between the last ssim and the current one
                if abs(avg_loss - last_ssim) < 1e-5:
                    print(f"Converged at epoch {epoch+1}")
                    ssim_not_improved_count += 1
                else:
                    ssim_not_improved_count -= 1
                    if ssim_not_improved_count < 0:
                        ssim_not_improved_count = 0
                last_ssim = avg_loss
                if ssim_not_improved_count > 10:
                    break
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
    avg_psnr = 0
    avg_ssim = 0

    with torch.no_grad():
        eval_set = torch.zeros((num_examples, 3, ENC_IO_SIZE, ENC_IO_SIZE)).to(DEVICE)
        for batch in dataloader:
            batch = batch.to(DEVICE)
            eval_set = torch.cat((eval_set, batch), dim=0)
            # fill the eval_set until it reaches num_examples
            if len(eval_set) > num_examples:
                eval_set = eval_set[:num_examples]
                break

        recon_batch, _ = model(eval_set)
        batch_psnr, batch_ssim = calculate_psnr_ssim(eval_set, recon_batch)
        avg_psnr += batch_psnr
        avg_ssim += batch_ssim

        for i in range(len(batch)):
            original_flat = eval_set[i].view(-1).cpu().numpy()
            reconstructed_flat = recon_batch[i].view(-1).cpu().numpy()
            mse_val = ((original_flat - reconstructed_flat) ** 2).mean()
            mse_total += mse_val

    mean_mse = mse_total / num_examples
    avg_psnr /= num_examples
    avg_ssim /= num_examples

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
