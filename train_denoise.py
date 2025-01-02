import os
import glob
import torch
import random
from torch.utils.data import Dataset, DataLoader, random_split
from denoise import train_denoise_model
from config import DEVICE, BATCH_SIZE_DENOISE_DATA


class ChunkedDenoiseDataset(Dataset):
    """
    Loads denoise data generated from `generate_denoise_dataset` on demand from several .pth chunks.
    Each .pth file contains a dict with "clean", "recon", and "residual".
    """

    def __init__(self, chunk_dir="results/ckpt/denoise_dataset", transform=None):
        super().__init__()
        self.transform = transform
        self.chunk_files = glob.glob(os.path.join(chunk_dir, "*.pth"))
        random.shuffle(self.chunk_files)
        # Build an index of (file_index, sample_index_in_file)
        self.index_map = []
        self.file_content_lens = {}
        for fidx, path in enumerate(self.chunk_files):
            chunk_data = torch.load(path, map_location="cpu")
            length = len(chunk_data["clean"])
            self.file_content_lens[fidx] = length
            # Create an index for each sample of this chunk
            for idx in range(length):
                self.index_map.append((fidx, idx))
        random.shuffle(self.index_map)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        fidx, sample_idx = self.index_map[idx]
        file_path = self.chunk_files[fidx]
        chunk_data = torch.load(file_path, map_location="cpu")
        noisy_img = chunk_data["recon"][sample_idx]
        residual = chunk_data["residual"][sample_idx]
        clean_img = chunk_data["clean"][sample_idx]
        return noisy_img, residual, clean_img


if __name__ == "__main__":
    # Prepare dataset
    dataset = ChunkedDenoiseDataset()

    # Split into train, val, test
    total_len = len(dataset)
    val_len = int(0.15 * total_len)
    test_len = int(0.15 * total_len)
    train_len = total_len - val_len - test_len
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_len, val_len, test_len]
    )

    # Prepare dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE_DENOISE_DATA, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE_DENOISE_DATA, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE_DENOISE_DATA, shuffle=False
    )

    # Train for 500 epochs, save checkpoint every 50
    EPOCHS = 1
    SAVE_PER_EPOCH = 50
    model, train_losses = train_denoise_model(
        dataloader=train_loader,
        gaussian_noise_model=False,  # Because we use AE residual chunks
        epochs=EPOCHS,
    )

    # save final model
    torch.save(
        {"model_state_dict": model.state_dict(), "train_losses": train_losses},
        "final_denoise_model.pth",
    )

    # Manual checkpoint saves every 50 epochs if needed
    # (the train_denoise_model call can also be modified to pass in a custom save frequency).
    # For demonstration, assume we don't alter train_denoise_model, so do a manual save below.
    # This snippet is conceptual if user wants additional saves beyond the built-in process.
    # for epoch_i in range(EPOCHS):
    #     ...
    #     if (epoch_i + 1) % SAVE_PER_EPOCH == 0:
    #         # save checkpoint manually
    #         pass
