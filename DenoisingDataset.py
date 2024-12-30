import torch
import numpy as np
from config import (
    DEVICE,
    ENC_IO_SIZE,
    BATCH_SIZE_DENOISE_DATA_INFERENCE,
    BATCH_SIZE_DENOISE_DATA,
)
from torchvision import transforms


class DenoisingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        source_ae_model,
        indices,
        device=DEVICE,
    ):
        self.indices = indices
        self.dataset = torch.utils.data.Subset(dataset, indices)

        self.transform = transforms.Compose(
            [
                transforms.Resize((ENC_IO_SIZE, ENC_IO_SIZE)),
                transforms.ToTensor(),
            ]
        )

        self._preprocess_all_images(device, source_ae_model)

    def _preprocess_all_images(self, device, source_ae_model):
        preprocessed_images = []

        for idx in range(len(self.dataset)):
            example = self.dataset[idx]
            image = example["image"]

            # Ensure the image is in RGB format
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Apply the transformations
            if self.transform:
                image = self.transform(image)

        # inference on source ae model to get residual
        source_ae_model.eval()
        with torch.no_grad():
            # split the preprocessed images into batches of 32
            residual = []
            noisy_images = []
            batch_size = BATCH_SIZE_DENOISE_DATA_INFERENCE
            for i in range(0, len(preprocessed_images), batch_size):
                batch = preprocessed_images[i : i + batch_size]
                _, recon_batch = source_ae_model(batch)
                residual.append(batch - recon_batch)
                noisy_images.append(recon_batch)

            residual = torch.cat(residual, dim=0)
            noisy_images = torch.cat(noisy_images, dim=0)
            self.noisy_images = noisy_images
            self.residual = residual

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        return (self.noisy_images[idx], self.residual[idx])


def init_denoising_dataset(dataset, source_ae_model, indices):
    denoising_dataset = DenoisingDataset(
        dataset=dataset,
        source_ae_model=source_ae_model,
        indices=indices,
    )

    dataloader = torch.utils.data.DataLoader(
        denoising_dataset, batch_size=BATCH_SIZE_DENOISE_DATA, shuffle=True
    )

    return denoising_dataset, dataloader
