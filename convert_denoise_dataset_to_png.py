import os
import torch
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage
from PIL import Image
import argparse


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def tensor_to_pil(tensor):
    """
    Converts a torch tensor to a PIL Image.
    Assumes the tensor is in CxHxW format and normalized between 0 and 1.
    """
    tensor = tensor.clamp(0, 1)  # Ensure the tensor is in [0,1]
    pil_image = ToPILImage()(tensor)
    return pil_image


def main():
    parser = argparse.ArgumentParser(
        description="Convert .pth data chunks to PNG images."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the .pth files (e.g., SAVE_DIR_CKPT/denoise_dataset).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the PNG images will be saved.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["clean", "recon", "residual"],
        help="Categories to convert. Default: clean recon residual",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    categories = args.categories

    # Create output directories for each category
    for category in categories:
        create_dir(os.path.join(output_dir, category))

    # List all .pth files in the data directory
    pth_files = [f for f in os.listdir(data_dir) if f.endswith(".pth")]
    if not pth_files:
        print(f"No .pth files found in {data_dir}.")
        return

    for pth_file in pth_files:
        pth_path = os.path.join(data_dir, pth_file)
        print(f"Processing file: {pth_file}")

        # Load the data
        try:
            data = torch.load(pth_path, map_location="cpu")
        except Exception as e:
            print(f"Failed to load {pth_file}: {e}")
            continue

        # Check if all required categories are present
        missing_categories = [cat for cat in categories if cat not in data]
        if missing_categories:
            print(f"Missing categories {missing_categories} in {pth_file}. Skipping.")
            continue

        # Iterate over each category
        for category in categories:
            images = data[category]  # Tensor of shape (N, C, H, W)
            for idx in range(images.size(0)):
                img_tensor = images[idx]
                pil_image = tensor_to_pil(img_tensor)

                # Define the filename
                base_filename = os.path.splitext(pth_file)[0]
                image_filename = f"{base_filename}_{category}_{idx}.png"
                image_path = os.path.join(output_dir, category, image_filename)

                # Save the image
                pil_image.save(image_path)

        print(f"Finished processing {pth_file}.")

    print("All files have been processed successfully.")


if __name__ == "__main__":
    main()
