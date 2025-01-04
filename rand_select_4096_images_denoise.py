import os
import random
import shutil

# ------------------- Configuration -------------------

# Paths to the converted PNG image directories
CLEAN_DIR = "C:\\denoise_images\\clean"
RECON_DIR = "C:\\denoise_images\\recon"

# Path to the subset output directory
SUBSET_DIR = "C:\\denoise_images\\subset"

# Number of image sets to select
NUM_PAIRS = 4096

# -----------------------------------------------------


def create_dir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def main():
    # Step 1: Verify input directories exist
    for directory in [CLEAN_DIR, RECON_DIR]:
        if not os.path.isdir(directory):
            print(f"Error: Directory does not exist - {directory}")
            return

    # Step 2: Create subset directories
    subset_clean = os.path.join(SUBSET_DIR, "clean")
    subset_recon = os.path.join(SUBSET_DIR, "recon")

    for subset_dir in [subset_clean, subset_recon]:
        create_dir(subset_dir)

    # Step 3: Get list of clean images
    clean_images = os.listdir(CLEAN_DIR)
    total_available = len(clean_images)

    if total_available < NUM_PAIRS:
        print(
            f"Warning: Requested {NUM_PAIRS} pairs, but only {total_available} available."
        )
        num_to_select = total_available
    else:
        num_to_select = NUM_PAIRS

    print(f"Total available image sets: {total_available}")
    print(f"Selecting {num_to_select} image sets randomly.")

    # Step 4: Randomly select image sets
    selected_ids = random.sample(clean_images, num_to_select)

    # Step 5: Copy selected image sets to the subset directory
    for image_id in selected_ids:
        clean_src = os.path.join(CLEAN_DIR, image_id)
        recon_src = os.path.join(RECON_DIR, image_id.replace("clean", "recon"))

        if not os.path.exists(recon_src):
            raise FileNotFoundError(f"Reconstructed image not found: {recon_src}")

        clean_dst = os.path.join(subset_clean, image_id)
        recon_dst = os.path.join(subset_recon, image_id)

        shutil.copy(clean_src, clean_dst)
        shutil.copy(recon_src, recon_dst)

    print(
        f"Subset creation complete. {num_to_select} image sets copied to {SUBSET_DIR}."
    )


if __name__ == "__main__":
    main()
