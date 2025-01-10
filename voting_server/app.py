from flask import (
    Flask,
    render_template,
    request,
    redirect,
    send_from_directory,
    url_for,
)
import os
import glob
import math
import random

app = Flask(__name__)

# Configuration
OUTPUT_DIR = "human_dataset"  # Directory containing the category subdirectories
LOG_FILE = "results.log"  # File to log results every 10 votes
VOTES_PER_LOG = 5  # Number of votes after which to log results

# Categories
CATEGORIES = ["small", "base", "large", "xlarge"]

# Initialize variables
image_sets = (
    []
)  # List of dictionaries: {'category': ..., 'base_name': ..., 'clean_path': ..., 'jpeg_path': ..., 'recon_path': ...}
current_index = 0  # Not needed for random selection but kept if you want to track
jpeg_wins = {category: 0 for category in CATEGORIES}
algorithm_wins = {category: 0 for category in CATEGORIES}
total_votes = {category: 0 for category in CATEGORIES}


# Load all image sets on startup
def load_image_sets():
    global image_sets
    for category in CATEGORIES:
        category_dir = os.path.join(OUTPUT_DIR, category)
        if not os.path.isdir(category_dir):
            raise ValueError(
                f"Category directory '{category}' does not exist in the output directory."
            )

        clean_images = sorted(glob.glob(os.path.join(category_dir, "*_clean.png")))
        for clean_path in clean_images:
            base_name = os.path.basename(clean_path).replace("_clean.png", "")
            jpeg_path = os.path.join(category_dir, f"{base_name}_jpeg.jpg")
            recon_path = os.path.join(category_dir, f"{base_name}_reconstructed.png")
            if os.path.exists(jpeg_path) and os.path.exists(recon_path):
                # convet the paths to use forward slashes for Windows compatibility
                clean_path = clean_path.replace("\\", "/")
                jpeg_path = jpeg_path.replace("\\", "/")
                recon_path = recon_path.replace("\\", "/")
                image_sets.append(
                    {
                        "category": category,
                        "base_name": base_name,
                        "clean_path": clean_path,
                        "jpeg_path": jpeg_path,
                        "recon_path": recon_path,
                    }
                )
    if not image_sets:
        raise ValueError("No valid image sets found in the output directory.")


# Load image sets at startup
load_image_sets()


# Route for the main page
@app.route("/")
def index():
    if not image_sets:
        return render_template(
            "index.html",
            finished=True,
            jpeg_wins=jpeg_wins,
            algorithm_wins=algorithm_wins,
            total_votes=total_votes,
        )

    # Randomly select an image set
    image_set = random.choice(image_sets)

    # Shuffle the order of the two images
    options = [
        {
            "displayName": "Compressed",
            "choice": "jpeg",
            "path": image_set["jpeg_path"],
        },
        {
            "displayName": "Compressed",
            "choice": "algorithm",
            "path": image_set["recon_path"],
        },
    ]
    random.shuffle(options)

    return render_template(
        "index.html",
        image_set=image_set,
        options=options,
        jpeg_wins=jpeg_wins,
        algorithm_wins=algorithm_wins,
        total_votes=total_votes,
        finished=False,
    )


# Route to handle voting
@app.route("/vote", methods=["POST"])
def vote():
    global image_sets, total_votes, jpeg_wins, algorithm_wins
    choice = request.form.get("choice")
    category = request.form.get("category")  # Ensure the category is sent from the form

    if category not in CATEGORIES:
        # Invalid category; redirect back without incrementing
        return redirect(url_for("index"))

    if choice == "jpeg":
        jpeg_wins[category] += 1
    elif choice == "algorithm":
        algorithm_wins[category] += 1
    else:
        # Invalid choice; redirect back without incrementing
        return redirect(url_for("index"))

    total_votes[category] += 1

    # Remove the voted image set to avoid repetition
    image_sets = [
        img
        for img in image_sets
        if not (
            img["category"] == category
            and img["base_name"] == request.form.get("base_name")
        )
    ]

    # Log results every VOTES_PER_LOG votes per category
    if total_votes[category] % VOTES_PER_LOG == 0:
        log_results(category)

    return redirect(url_for("index"))


# Function to log results
def log_results(category=None):
    with open(LOG_FILE, "a") as f:  # Changed to append mode to keep previous logs
        if category:
            f.write(f"--- {category.capitalize()} Category ---\n")
            f.write(f"Total Votes: {total_votes[category]}\n")
            f.write(f"JPEG Wins: {jpeg_wins[category]}\n")
            f.write(f"Algorithm Wins: {algorithm_wins[category]}\n")
            if total_votes[category] > 0:
                jpeg_rate = (jpeg_wins[category] / total_votes[category]) * 100
                algorithm_rate = (
                    algorithm_wins[category] / total_votes[category]
                ) * 100
                f.write(f"JPEG Win Rate: {jpeg_rate:.2f}%\n")
                f.write(f"Algorithm Win Rate: {algorithm_rate:.2f}%\n")
            else:
                f.write("No votes yet.\n")
            f.write("\n")  # Add a newline for readability
        else:
            # Log all categories if no specific category is provided
            for cat in CATEGORIES:
                f.write(f"--- {cat.capitalize()} Category ---\n")
                f.write(f"Total Votes: {total_votes[cat]}\n")
                f.write(f"JPEG Wins: {jpeg_wins[cat]}\n")
                f.write(f"Algorithm Wins: {algorithm_wins[cat]}\n")
                if total_votes[cat] > 0:
                    jpeg_rate = (jpeg_wins[cat] / total_votes[cat]) * 100
                    algorithm_rate = (algorithm_wins[cat] / total_votes[cat]) * 100
                    f.write(f"JPEG Win Rate: {jpeg_rate:.2f}%\n")
                    f.write(f"Algorithm Win Rate: {algorithm_rate:.2f}%\n")
                else:
                    f.write("No votes yet.\n")
                f.write("\n")  # Add a newline for readability


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
