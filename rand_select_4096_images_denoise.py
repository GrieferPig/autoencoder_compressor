import os
import random
import shutil

# ------------------- Configuration -------------------

# Paths to the converted PNG image directories
CLEAN_DIR = "/path/to/SAVE_DIR_CKPT/denoise_images/clean"
RECON_DIR = "/path/to/SAVE_DIR_CKPT/denoise_images/recon"

# Path to the subset output directory
SUBSET_DIR = "/path/to/SAVE_DIR_CKPT/denoise_subset"

# Number of image sets to select
NUM_PAIRS = 4096

# -----------------------------------------------------


def create_dir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def get_unique_identifiers(clean_dir):
    """
    Extract unique identifiers from clean image filenames.

    Assumes filenames are in the format: {base_pth_filename}_clean_{index}.png
    """
    identifiers = []
    for filename in os.listdir(clean_dir):
        if filename.endswith(".png") and "_clean_" in filename:
            # Remove the '_clean_{index}.png' suffix to get the unique identifier
            base = filename.split("_clean_")[0]
            identifiers.append(base)
    return identifiers


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

    # Step 3: Get list of unique identifiers from clean images
    unique_ids = get_unique_identifiers(CLEAN_DIR)
    total_available = len(unique_ids)

    if total_available < NUM_PAIRS:
        print(
            f"Warning: Requested {NUM_PAIRS} pairs, but only {total_available} available."
        )
        num_to_select = total_available
    else:
        num_to_select = NUM_PAIRS

    print(f"Total available image sets: {total_available}")
    print(f"Selecting {num_to_select} image sets randomly.")

    # Step 4: Randomly select identifiers
    selected_ids = random.sample(unique_ids, num_to_select)

    # Step 5: Copy selected images to subset directories
    for idx, base_id in enumerate(selected_ids, 1):
        # Construct filenames for each category
        clean_filename = f"{base_id}_clean_{idx}.png"
        recon_filename = f"{base_id}_recon_{idx}.png"

        # Source file paths
        clean_src = os.path.join(CLEAN_DIR, f"{base_id}_clean_{idx}.png")
        recon_src = os.path.join(RECON_DIR, f"{base_id}_recon_{idx}.png")

        # Destination file paths
        clean_dst = os.path.join(subset_clean, f"{base_id}_clean_{idx}.png")
        recon_dst = os.path.join(subset_recon, f"{base_id}_recon_{idx}.png")

        # Check if all source files exist
        if not (os.path.isfile(clean_src) and os.path.isfile(recon_src)):
            raise FileNotFoundError("One or more source files not found.")
            continue

        # Copy files
        shutil.copy2(clean_src, clean_dst)
        shutil.copy2(recon_src, recon_dst)

        if idx % 500 == 0 or idx == num_to_select:
            print(f"Copied {idx}/{num_to_select} image sets.")

    print(
        f"Subset creation complete. {num_to_select} image sets copied to {SUBSET_DIR}."
    )


if __name__ == "__main__":
    main()
