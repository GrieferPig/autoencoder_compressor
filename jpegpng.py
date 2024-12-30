from config import JPEG_QUALITY_FACTORS, PNG_COMPRESSION_LEVELS, SAVE_DIR_COMPRESSION
from PIL import Image
from torchvision import transforms
import io
import numpy as np
from utils import calculate_psnr_ssim
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Compression cannot be done on GPU
DEVICE = "cpu"


def compress_image(idx, image_pil):
    """
    Compresses an image using both JPEG and PNG, calculates metrics, and returns the results.

    Parameters:
    - idx (int): Index of the image in the dataset.
    - image_pil (PIL.Image): PIL Image object to be compressed.

    Returns:
    - results (list of dict): Metrics for JPEG and PNG compression.
    """
    results = []
    theoretical_size = image_pil.size[0] * image_pil.size[1] * 3  # RGB888

    # JPEG Compression
    for quality in JPEG_QUALITY_FACTORS:
        buffer = io.BytesIO()
        image_pil.save(buffer, format="JPEG", quality=quality, optimize=True)
        compressed_size = buffer.tell()

        # Decompress JPEG
        buffer.seek(0)
        decompressed_image = Image.open(buffer).convert("RGB")
        decompressed_tensor = transforms.ToTensor()(decompressed_image).to(DEVICE)

        # Original tensor for metrics
        original_tensor = transforms.ToTensor()(image_pil).to(DEVICE)

        # Compression Ratio
        compression_ratio = (
            theoretical_size / compressed_size if compressed_size != 0 else 0
        )

        # Calculate metrics
        similarity, psnr, ssim = calculate_psnr_ssim(
            original_tensor, decompressed_tensor
        )

        results.append(
            {
                "Image_Index": idx,
                "Algorithm": "JPEG",
                "Parameter": quality,
                "Compression_Ratio": compression_ratio,
                "Cosine_Similarity": similarity,
                "PSNR": psnr,
                "SSIM": ssim,
            }
        )

    # PNG Compression
    for compression_level in PNG_COMPRESSION_LEVELS:
        buffer = io.BytesIO()
        image_pil.save(
            buffer, format="PNG", compress_level=compression_level, optimize=True
        )
        compressed_size = buffer.tell()

        # Decompress PNG
        buffer.seek(0)
        decompressed_image = Image.open(buffer).convert("RGB")
        decompressed_tensor = transforms.ToTensor()(decompressed_image).to(DEVICE)

        # Compression Ratio
        compression_ratio = (
            theoretical_size / compressed_size if compressed_size != 0 else 0
        )

        # For PNG, set Cosine Similarity to 1, PSNR and SSIM to 1.0 since it's lossless
        similarity = 1.0
        psnr = float("inf")  # Perfect reconstruction
        ssim = 1.0

        results.append(
            {
                "Image_Index": idx,
                "Algorithm": "PNG",
                "Parameter": compression_level,
                "Compression_Ratio": compression_ratio,
                "Cosine_Similarity": similarity,
                "PSNR": psnr,
                "SSIM": ssim,
            }
        )

    return results


def plot_metrics(average_metrics, figures_dir):
    """
    Generates and saves plots for Compression Ratio, Cosine Similarity, PSNR, and SSIM.

    Parameters:
    - average_metrics (pd.DataFrame): DataFrame containing average metrics.
    - figures_dir (str): Directory to save the plots.
    """
    metrics = ["Compression_Ratio", "Cosine_Similarity", "PSNR", "SSIM"]
    y_labels = {
        "Compression_Ratio": "Average Compression Ratio",
        "Cosine_Similarity": "Average Cosine Similarity",
        "PSNR": "Average PSNR",
        "SSIM": "Average SSIM",
    }

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for algorithm in ["JPEG", "PNG"]:
            subset = average_metrics[average_metrics["Algorithm"] == algorithm]
            plt.plot(subset["Parameter"], subset[metric], marker="o", label=algorithm)
        plt.title(f'{metric.replace("_", " ")} vs Compression Parameter')
        plt.xlabel("Compression Parameter (Quality for JPEG, Level for PNG)")
        plt.ylabel(y_labels[metric])
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figures_dir, f"{metric.lower()}.png"))
        plt.close()


def generate_example_reconstructions(
    preprocessed_dataset, compression_settings, figures_dir, num_examples=4
):
    """
    Generates and saves example reconstructions for both JPEG and PNG compression.

    Parameters:
    - preprocessed_dataset (CustomDataset): The dataset containing preprocessed images.
    - compression_settings (dict): Dictionary containing JPEG and PNG parameters.
    - figures_dir (str): Directory to save the example plots.
    - num_examples (int): Number of example images to generate.
    """
    example_indices = np.random.choice(
        len(preprocessed_dataset), size=num_examples, replace=False
    )

    for algo in ["JPEG", "PNG"]:
        for param in compression_settings[algo]:
            plt.figure(figsize=(12, 6))
            for i, idx in enumerate(example_indices):
                image_pil = preprocessed_dataset[idx]
                if algo == "JPEG":
                    buffer = io.BytesIO()
                    image_pil.save(buffer, format="JPEG", quality=param, optimize=True)
                else:
                    buffer = io.BytesIO()
                    image_pil.save(
                        buffer, format="PNG", compress_level=param, optimize=True
                    )
                buffer.seek(0)
                decompressed_image = Image.open(buffer).convert("RGB")
                decompressed_tensor = transforms.ToTensor()(decompressed_image)

                # Original Image
                ax = plt.subplot(2, num_examples, i + 1)
                plt.imshow(image_pil)
                ax.axis("off")
                if i == 0:
                    ax.set_title("Original")

                # Reconstructed Image
                ax = plt.subplot(2, num_examples, i + 1 + num_examples)
                plt.imshow(decompressed_image)
                ax.axis("off")
                if i == 0:
                    ax.set_title(f"Reconstructed ({algo}={param})")

            plt.suptitle(f"Example Reconstructions using {algo} (Parameter={param})")
            plot_filename = os.path.join(
                figures_dir, f"reconstructions_{algo}_{param}.png"
            )
            plt.savefig(plot_filename)
            plt.close()


def run_compression_experiment(custom_dataset, CSV_FILE):
    print("Preparing arguments for compression...")
    compression_args = list(enumerate(custom_dataset))

    # Start sequential compression
    print("Starting compression...")
    results = []
    for args in tqdm(compression_args, desc="Compressing Images"):
        result = compress_image(args)
        results.extend(result)

    # Save results to CSV
    print(f"Saving results to {CSV_FILE}...")
    df = pd.DataFrame(results)
    df.to_csv(CSV_FILE, index=False)

    # Calculate average metrics for each algorithm and parameter
    print("Calculating average metrics...")
    average_metrics = (
        df.groupby(["Algorithm", "Parameter"])
        .agg(
            {
                "Compression_Ratio": "mean",
                "Cosine_Similarity": "mean",
                "PSNR": "mean",
                "SSIM": "mean",
            }
        )
        .reset_index()
    )

    # Save average metrics to CSV
    average_csv = os.path.join(SAVE_DIR_COMPRESSION, "average_metrics.csv")
    average_metrics.to_csv(average_csv, index=False)

    figures_dir = os.path.join(SAVE_DIR_COMPRESSION, "figures")

    # Plotting
    print("Generating plots...")
    plot_metrics(average_metrics, figures_dir)

    # Generate Example Reconstructions
    print("Generating example reconstructions...")
    compression_settings = {"JPEG": JPEG_QUALITY_FACTORS, "PNG": PNG_COMPRESSION_LEVELS}
    generate_example_reconstructions(custom_dataset, compression_settings, figures_dir)

    print("Compression experiment completed successfully!")
