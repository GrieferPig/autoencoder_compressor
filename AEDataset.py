import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from config import DEVICE, ENC_IO_SIZE, BATCH_SIZE_AE_DATA
from glob import glob
import os


class AEDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        device=DEVICE,
        length=None,
        transform_img_size=ENC_IO_SIZE,
        indices=None,
        shuffle=True,
        process_on_demand=False,
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
        self.process_on_demand = process_on_demand

        self.len = len(indices)

        self.transform = transforms.Compose(
            [
                transforms.Resize((transform_img_size, transform_img_size)),
                transforms.ToTensor(),
            ]
        )

        if not process_on_demand:
            self._preprocess_all_images(device)

    def _preprocess_all_images(self, device):
        self.preprocessed_data = []
        for idx in range(len(self.dataset)):
            self.preprocessed_data.append(self._preprocess_one_image(idx))
            if idx % 100 == 0:
                print(f"Preprocessed {idx} / {len(self.dataset)} images")

        self.preprocessed_data = torch.stack(self.preprocessed_data).to(device)

    def _preprocess_one_image(self, idx):
        example = self.dataset[idx]
        image = example["image"]

        # Ensure the image is in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply the transformations
        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.process_on_demand:
            return self._preprocess_one_image(idx)

        return self.preprocessed_data[idx]


def init_ae_dataset(
    dataset, length=None, indices=None, shuffle=True, process_on_demand=False
):
    if isinstance(indices, str):
        # split the string into a list of integers
        indices = list(map(int, indices.split(",")))
    if isinstance(dataset, str):
        # If the dataset is a string, it is a path to a folder containing images
        print("Using custom dataset")
        custom_dataset = AECustomDataset(
            dataset,
            length=length,
            indices=indices,
            shuffle=shuffle,
            process_on_demand=process_on_demand,
        )
    else:
        custom_dataset = AEDataset(
            dataset,
            length=length,
            indices=indices,
            shuffle=shuffle,
            process_on_demand=process_on_demand,
        )
    dataloader = torch.utils.data.DataLoader(
        custom_dataset, batch_size=BATCH_SIZE_AE_DATA, shuffle=False
    )
    return custom_dataset, dataloader


class AECustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_folder_path,
        device=DEVICE,
        length=None,
        transform_img_size=ENC_IO_SIZE,
        indices=None,
        shuffle=True,
        process_on_demand=False,
    ):

        # filter only jpg, jpeg and png
        all_files = glob(os.path.join(image_folder_path, "*"))
        all_files = [
            f
            for f in all_files
            if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")
        ]
        all_files = sorted(all_files)

        if indices is None:
            indices = np.arange(len(all_files))
            if shuffle:
                np.random.shuffle(indices)

        if not isinstance(indices, list):
            indices = indices.tolist()

        if length is not None:
            indices = indices[:length]
        self.indices = indices

        self.files = [all_files[i] for i in indices]
        self.process_on_demand = process_on_demand
        self.len = len(self.files)

        self.transform = transforms.Compose(
            [
                transforms.Resize((transform_img_size, transform_img_size)),
                transforms.ToTensor(),
            ]
        )

        if not process_on_demand:
            self._preprocess_all_images(device)

    def _preprocess_all_images(self, device):
        self.preprocessed_data = []
        for idx in range(self.len):
            self.preprocessed_data.append(self._preprocess_one_image(idx))
            if idx % 100 == 0:
                print(f"Preprocessed {idx} / {self.len} images")

        self.preprocessed_data = torch.stack(self.preprocessed_data).to(device)

    def _preprocess_one_image(self, idx):
        image_path = self.files[idx]
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.process_on_demand:
            return self._preprocess_one_image(idx)
        return self.preprocessed_data[idx]
