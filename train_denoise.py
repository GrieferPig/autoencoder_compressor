import os
import glob  # Added for checkpoint file handling
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from DnCNN import DenoisingModel
from config import SAVE_DIR_CKPT, DEVICE
import math
import random
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm  # Import tqdm


class DenoiseDataset(Dataset):
    def __init__(self, clean_dir, recon_dir, transform=None):
        """
        Initializes the dataset by listing all image files in clean and recon directories.

        Args:
            clean_dir (str): Path to the directory containing clean images.
            recon_dir (str): Path to the directory containing recon images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.clean_dir = clean_dir
        self.recon_dir = recon_dir
        self.transform = transform

        # List all PNG files in the clean directory
        self.clean_files = sorted(
            [f for f in os.listdir(clean_dir) if f.lower().endswith(".png")]
        )
        # List all PNG files in the recon directory
        self.recon_files = sorted(
            [f for f in os.listdir(recon_dir) if f.lower().endswith(".png")]
        )

        if len(self.clean_files) != len(self.recon_files):
            raise ValueError("The number of clean and recon images must be the same.")

        # Ensure that filenames match
        for clean_file, recon_file in zip(self.clean_files, self.recon_files):
            # separate name by dot
            clean_file = clean_file.split(".")[0]
            recon_file = recon_file.split(".")[0]
            if clean_file != recon_file:
                raise ValueError(f"Mismatched files: {clean_file} and {recon_file}")

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        """
        Retrieves the recon and clean images for the given index and computes the residual.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (recon_image, clean_image, residual)
        """
        clean_path = os.path.join(self.clean_dir, self.clean_files[idx])
        recon_path = os.path.join(self.recon_dir, self.recon_files[idx])

        # Load images
        clean_image = Image.open(clean_path).convert("RGB")
        recon_image = Image.open(recon_path).convert("RGB")

        transform = transforms.ToTensor()
        clean_image = transform(clean_image)
        recon_image = transform(recon_image)

        return recon_image, clean_image


def test_model(model, test_loader, criterion, epoch, device=DEVICE):
    """
    Evaluates the model on the test set and saves sample denoised images.

    Args:
        model (nn.Module): The denoising model.
        test_loader (DataLoader): DataLoader for the test set.
        criterion (nn.Module): Loss function.
        epoch (int): Current epoch number.
        device (torch.device): Device to perform computations on.
    """
    samples = []
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for recon, clean in tqdm(
            test_loader, desc=f"Epoch {epoch} - Testing", leave=False
        ):
            recon, clean = recon.to(device), clean.to(device)
            outputs = model(recon)
            loss = criterion(outputs, clean)
            if len(samples) < 5:
                samples.append(
                    (
                        recon.squeeze().cpu(),
                        outputs.squeeze().cpu(),
                        clean.squeeze().cpu(),
                    )
                )
            test_loss += loss.item() * recon.size(0)
    test_loss /= len(test_loader.dataset)

    fig, axes = plt.subplots(4, 3, figsize=(12, 12))
    for i, (recon_img, denoised_img, clean_img) in enumerate(samples):
        axes[i, 0].imshow(recon_img.squeeze(0).permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Noisy")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(denoised_img.squeeze(0).permute(1, 2, 0).numpy())
        axes[i, 1].set_title("Denoised")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(clean_img.squeeze(0).permute(1, 2, 0).numpy())
        axes[i, 2].set_title("Clean")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(f"test_results_epoch_{epoch}.png")
    plt.close()


def main():
    data_dir = "C:\\Users\\evanl\\Downloads\\subset"
    clean_dir = os.path.join(data_dir, "clean")
    recon_dir = os.path.join(data_dir, "recon")

    if not os.path.isdir(clean_dir) or not os.path.isdir(recon_dir):
        raise FileNotFoundError(
            f"One or both data directories {clean_dir}, {recon_dir} do not exist."
        )

    # Define transformations (if any)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # Add other transformations here if needed
        ]
    )

    # Instantiate the dataset
    full_dataset = DenoiseDataset(clean_dir, recon_dir, transform=transform)
    num_samples = len(full_dataset)

    if num_samples == 0:
        raise ValueError("No samples found in the dataset.")

    print(f"Found {num_samples} samples in the dataset.")

    # Create a list of global indices
    indices = list(range(num_samples))

    # Shuffle indices before splitting
    random.shuffle(indices)

    # Split the indices into train, val, and test sets
    train_size = math.ceil(0.8 * num_samples)
    val_size = math.ceil(0.1 * num_samples)
    test_size = num_samples - train_size - val_size
    print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")
    train_indices, val_indices, test_indices = random_split(
        indices,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),  # For reproducibility
    )

    # Convert Subset objects to lists
    train_indices = list(train_indices)
    val_indices = list(val_indices)
    test_indices = list(test_indices)

    # Instantiate subsets
    train_set = torch.utils.data.Subset(full_dataset, train_indices)
    val_set = torch.utils.data.Subset(full_dataset, val_indices)
    test_set = torch.utils.data.Subset(full_dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    device = DEVICE
    print(f"Using device: {device}")

    model = DenoisingModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Ensure the checkpoints directory exists
    os.makedirs("checkpoints", exist_ok=True)

    # Initialize training parameters
    num_epochs = 101  # Adjust as needed
    start_epoch = 1  # Default starting epoch

    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join("checkpoints", "dncnn_epoch_*.pth"))
    if checkpoint_files:
        # Extract epoch numbers from filenames
        epochs = []
        for file in checkpoint_files:
            basename = os.path.basename(file)
            try:
                epoch_num = int(basename.split("_")[2].split(".")[0])
                epochs.append(epoch_num)
            except (IndexError, ValueError):
                continue
        if epochs:
            latest_epoch = max(epochs)
            latest_checkpoint = os.path.join(
                "checkpoints", f"dncnn_epoch_{latest_epoch}.pth"
            )
            print(f"Loading checkpoint from epoch {latest_epoch}...")
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resuming training from epoch {start_epoch}...")
        else:
            print("No valid checkpoint files found. Starting training from scratch.")
    else:
        print("No checkpoint files found. Starting training from scratch.")

    # Adjust the range of epochs based on the starting epoch
    if start_epoch > num_epochs:
        print(
            f"Checkpoint epoch ({start_epoch}) is greater than num_epochs ({num_epochs}). Exiting."
        )
        return

    # Wrap the epoch loop with tqdm
    for epoch in tqdm(range(start_epoch, num_epochs + 1), desc="Training Epochs"):
        model.train()
        running_loss = 0.0

        # Wrap the training batch loop with tqdm
        for recon, clean in tqdm(
            train_loader, desc=f"Epoch {epoch}/{num_epochs} - Training", leave=False
        ):
            recon, clean = recon.to(device), clean.to(device)
            optimizer.zero_grad()
            outputs = model(recon)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * recon.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0

        # Wrap the validation batch loop with tqdm
        for recon, clean in tqdm(
            val_loader, desc=f"Epoch {epoch}/{num_epochs} - Validation", leave=False
        ):
            recon, clean = recon.to(device), clean.to(device)
            outputs = model(recon)
            loss = criterion(outputs, clean)
            val_loss += loss.item() * recon.size(0)
        val_loss /= len(val_loader.dataset)

        tqdm.write(
            f"Epoch [{epoch}/{num_epochs}] Train Loss: {epoch_loss:.6f} Val Loss: {val_loss:.6f}"
        )

        # Save checkpoints at specified intervals
        if epoch % 10 == 0 or epoch == num_epochs:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": epoch_loss,
                "val_loss": val_loss,
            }
            checkpoint_path = os.path.join("checkpoints", f"dncnn_epoch_{epoch}.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
            test_model(model, test_loader, criterion, epoch)

    # Save the final model
    final_model_path = "dncnn_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")


if __name__ == "__main__":
    main()
