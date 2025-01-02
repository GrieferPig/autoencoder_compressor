import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from config import DEVICE, ENC_IO_SIZE, BATCH_SIZE_AE_DATA


class AEDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        device=DEVICE,
        length=None,
        transform_img_size=ENC_IO_SIZE,
        indices=None,
        shuffle=True,
    ):
        if indices is None:
            indices = np.arange(len(dataset))
            if shuffle:
                np.random.shuffle(indices)

        if not isinstance(indices, list):
            indices = indices.tolist()

        if length is not None:
            indices = indices[:length]

        self.indices = indices
        self.dataset = torch.utils.data.Subset(dataset, indices)

        self.transform = transforms.Compose(
            [
                transforms.Resize((transform_img_size, transform_img_size)),
                transforms.ToTensor(),
            ]
        )

        self._preprocess_all_images(device)

    def _preprocess_all_images(self, device):
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

            preprocessed_images.append(image)
            if idx % 100 == 0:
                print(f"Preprocessed {idx} / {len(self.dataset)} images")

        self.preprocessed_data = torch.stack(preprocessed_images).to(device)

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        return self.preprocessed_data[idx]


def init_ae_dataset(dataset, length=None, indices=None, shuffle=True):
    custom_dataset = AEDataset(dataset, length=length, indices=indices, shuffle=shuffle)
    dataloader = torch.utils.data.DataLoader(
        custom_dataset, batch_size=BATCH_SIZE_AE_DATA, shuffle=False
    )
    return custom_dataset, dataloader
