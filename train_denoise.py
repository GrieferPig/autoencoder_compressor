import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from DnCNN import DenoisingModel
import math

from config import SAVE_DIR_CKPT, DEVICE
import random
import matplotlib.pyplot as plt

# Import AMP modules
from torch.amp import GradScaler, autocast


class DenoiseDataset(Dataset):

    def __init__(self, chunk_files, chunk_indices_range, indices):
        """
        Initializes the dataset by mapping global indices to specific chunks and sample indices.
        """
        self.chunk_files = chunk_files
        self.chunk_indices_range = chunk_indices_range
        self.indices = indices
        self.current_chunk = None
        self.current_data = None
        self.current_indice_range = None
        self.load_chunk(0)

    def load_chunk(self, chunk_idx):
        """
        Loads the specified chunk into memory.
        """
        if self.current_chunk != chunk_idx:
            if self.current_data is not None:
                del self.current_data
            print(f"Loading chunk {chunk_idx}")
            checkpoint = torch.load(
                self.chunk_files[chunk_idx], map_location=DEVICE, weights_only=True
            )
            self.current_data = (
                checkpoint["clean"],
                checkpoint["recon"],
                checkpoint["residual"],
            )
            self.current_chunk = chunk_idx
            self.current_indice_range = self.chunk_indices_range[chunk_idx]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Retrieves the recon and clean images for the given index.
        """
        global_idx = self.indices[idx]
        current_min, current_max = self.current_indice_range
        if current_min <= global_idx < current_max:
            return (
                self.current_data[1][global_idx - current_min],
                self.current_data[0][global_idx - current_min],
                self.current_data[2][global_idx - current_min],
            )

        # find the min chunk index where the global index is less than the chunk index range
        chunk_idx = next(
            i
            for i, (min_idx, max_idx) in enumerate(self.chunk_indices_range)
            if global_idx < max_idx
        )
        self.load_chunk(chunk_idx)
        current_min, current_max = self.current_indice_range
        return (
            self.current_data[1][global_idx - current_min],
            self.current_data[0][global_idx - current_min],
            self.current_data[2][global_idx - current_min],
        )


def main():
    data_dir = os.path.join(SAVE_DIR_CKPT, "denoise_dataset")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    # Calculate the total number of samples
    num_samples = 0
    chunk_files = []
    chunk_indices_range = []
    for f in os.listdir(data_dir):
        if f.startswith("dataset_") and f.endswith(".pth"):
            filename = os.path.join(data_dir, f)
            print(f"Loading {filename}")
            checkpoint = torch.load(filename, map_location="cpu", weights_only=True)
            chunk_files.append(filename)
            chunk_indices_range.append(
                (num_samples, num_samples + len(checkpoint["clean"]))
            )
            num_samples += len(checkpoint["clean"])
            del checkpoint

    if num_samples == 0:
        raise ValueError("No samples found in the dataset.")

    print(f"Found {num_samples} samples in the dataset.")

    # Create a list of global indices
    indices = list(range(num_samples))

    # Split the indices into train, val, and test sets
    train_size = math.ceil(0.8 * num_samples)
    val_size = math.ceil(0.1 * num_samples)
    test_size = num_samples - train_size - val_size
    train_subset, val_subset, test_subset = random_split(
        indices, [train_size, val_size, test_size]
    )

    # Convert Subset objects to lists
    train_indices = list(train_subset)
    val_indices = list(val_subset)
    test_indices = list(test_subset)

    # sort the indices to decrease the number of chunk loads
    train_indices = sorted(train_indices)
    val_indices = sorted(val_indices)
    test_indices = sorted(test_indices)

    # Instantiate datasets
    train_set = DenoiseDataset(chunk_files, chunk_indices_range, train_indices)
    val_set = DenoiseDataset(chunk_files, chunk_indices_range, val_indices)
    test_set = DenoiseDataset(chunk_files, chunk_indices_range, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_set, batch_size=2, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=2, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    device = DEVICE
    print(f"Using device: {device}")

    model = DenoisingModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Initialize GradScaler for AMP
    scaler = GradScaler(device=device, enabled=True)

    os.makedirs("checkpoints", exist_ok=True)

    num_epochs = 500  # Updated to match the print statement

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for recon, _, residual in train_loader:
            recon, residual = recon.to(device, non_blocking=True), residual.to(
                device, non_blocking=True
            )
            optimizer.zero_grad()

            # Enable autocast for mixed precision
            with autocast():
                outputs = model(recon)
                loss = criterion(outputs, residual)

            # Scale the loss and perform backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * recon.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (
                recon,
                _,
                residual,
            ) in val_loader:  # Changed from train_loader to val_loader
                recon, residual = recon.to(device, non_blocking=True), residual.to(
                    device, non_blocking=True
                )
                with autocast(device_type=device, enabled=True, dtype=torch.bfloat16):
                    outputs = model(recon)
                    loss = criterion(outputs, residual)
                val_loss += loss.item() * recon.size(0)
        val_loss /= len(val_loader.dataset)

        print(
            f"Epoch [{epoch}/{num_epochs}] Train Loss: {epoch_loss:.6f} Val Loss: {val_loss:.6f}"
        )

        if epoch % 20 == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),  # Save scaler state
                "train_loss": epoch_loss,
                "val_loss": val_loss,
            }
            checkpoint_path = os.path.join("checkpoints", f"dncnn_epoch_{epoch}.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    # Test the model
    # Select 4 images from the test set and plot them
    samples = []
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for recon, clean, residual in test_loader:
            recon, residual = recon.to(device, non_blocking=True), residual.to(
                device, non_blocking=True
            )
            with autocast(device_type=device, enabled=True, dtype=torch.bfloat16):
                outputs = model(recon)
                loss = criterion(outputs, residual)
            if len(samples) < 5:
                samples.append(
                    (
                        recon.squeeze().cpu(),
                        (outputs + recon).squeeze().cpu(),
                        clean.squeeze().cpu(),
                    )
                )
            test_loss += loss.item() * recon.size(0)
    test_loss /= len(test_loader.dataset)

    fig, axes = plt.subplots(4, 3, figsize=(12, 12))
    for i, (recon_img, output_img, clean_img) in enumerate(samples):
        if i >= 4:
            break  # Ensure only 4 samples are plotted
        axes[i, 0].imshow(recon_img.permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Noisy")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(output_img.permute(1, 2, 0).numpy())
        axes[i, 1].set_title("Denoised")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(clean_img.permute(1, 2, 0).numpy())
        axes[i, 2].set_title("Clean")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()
    print(f"Test Loss: {test_loss:.6f}")

    # Save the final model
    torch.save(model.state_dict(), "dncnn_final.pth")


if __name__ == "__main__":
    main()
