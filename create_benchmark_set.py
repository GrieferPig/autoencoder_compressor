import os
import random
from PIL import Image
from datasets import load_dataset
from config import ENC_IO_SIZE, DATASET_REPO, DATASET_SPLIT
from AEDataset import AEDataset


def main():
    length = 256
    base_dataset = load_dataset(DATASET_REPO, split=DATASET_SPLIT)
    ae_dataset = AEDataset(base_dataset, length=length, process_on_demand=True)
    output_folder = "test_decompress_set"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Collect all images
    for i in range(length):
        img = ae_dataset[i]
        img = img.numpy().transpose(1, 2, 0)
        # unnormalize and convert to uint8
        img = (img * 255).astype("uint8")
        img = Image.fromarray(img)
        img.save(os.path.join(output_folder, f"image_{i}.png"))
        img.save(os.path.join(output_folder, f"image_{i}.jpg"))


if __name__ == "__main__":
    main()
