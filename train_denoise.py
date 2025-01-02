import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from DnCNN import DenoisingModel
import math

from config import SAVE_DIR_CKPT
import random
from config import DEVICE
import matplotlib.pyplot as plt


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
                checkpoint.get("residual", None),
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
        )


def main():
    data_dir = os.path.join(SAVE_DIR_CKPT, "denoise_dataset")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    # Calculate the total number of samples
    num_samples = 0
    chunk_files = []
    chunk_indices_range = []
    count = 0
    for f in os.listdir(data_dir):
        if f.startswith("dataset_") and f.endswith(".pth"):
            filename = os.path.join(data_dir, f)
            print(f"Loading {filename}")
            checkpoint = torch.load(filename, map_location="cpu")
            chunk_files.append(filename)
            chunk_indices_range.append(
                (num_samples, num_samples + len(checkpoint["clean"]))
            )
            num_samples += len(checkpoint["clean"])
            del checkpoint
            count += 1
            if count == 2:
                break

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

    # Plot a sample data from train dataset
    def plot_sample(set):
        recon, clean = set[0]
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(recon[0].permute(1, 2, 0).cpu().numpy().squeeze(), cmap="gray")
        axes[0].set_title("Reconstructed Image")
        axes[1].imshow(clean[0].permute(1, 2, 0).cpu().numpy().squeeze(), cmap="gray")
        axes[1].set_title("Clean Image")
        plt.show()

    plot_sample(train_set)

    print(train_set[0])

    # Save the final model
    torch.save(model.state_dict(), "dncnn_final.pth")


if __name__ == "__main__":
    main()
