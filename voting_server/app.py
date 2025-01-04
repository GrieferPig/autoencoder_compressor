from flask import Flask, render_template, request, redirect, url_for
import os
import glob
import math

app = Flask(__name__)

# Configuration
OUTPUT_DIR = "human_eval_set"  # Directory containing the images
LOG_FILE = "results.log"  # File to log results every 10 votes
VOTES_PER_LOG = 10  # Number of votes after which to log results

# Initialize variables
image_sets = []  # List of tuples: (index, clean_path, jpeg_path, recon_path)
current_index = 0
jpeg_wins = 0
algorithm_wins = 0
total_votes = 0


# Load all image sets on startup
def load_image_sets():
    global image_sets
    clean_images = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_clean.png")))
    for clean_path in clean_images:
        base_name = os.path.basename(clean_path).replace("_clean.png", "")
        jpeg_path = os.path.join(OUTPUT_DIR, f"{base_name}_jpeg.jpg")
        recon_path = os.path.join(OUTPUT_DIR, f"{base_name}_reconstructed.png")
        if os.path.exists(jpeg_path) and os.path.exists(recon_path):
            image_sets.append((base_name, clean_path, jpeg_path, recon_path))
    if not image_sets:
        raise ValueError("No valid image sets found in the output directory.")


# Load image sets at startup
load_image_sets()


# Route for the main page
@app.route("/")
def index():
    global current_index, total_votes, jpeg_wins, algorithm_wins
    if current_index >= len(image_sets):
        return render_template(
            "index.html",
            finished=True,
            jpeg_wins=jpeg_wins,
            algorithm_wins=algorithm_wins,
            total_votes=total_votes,
        )

    image_set = image_sets[current_index]
    return render_template(
        "index.html",
        image_set=image_set,
        jpeg_wins=jpeg_wins,
        algorithm_wins=algorithm_wins,
        total_votes=total_votes,
        finished=False,
    )


# Route to handle voting
@app.route("/vote", methods=["POST"])
def vote():
    global current_index, total_votes, jpeg_wins, algorithm_wins
    choice = request.form.get("choice")

    if choice == "jpeg":
        jpeg_wins += 1
    elif choice == "algorithm":
        algorithm_wins += 1
    else:
        # Invalid choice; redirect back without incrementing
        return redirect(url_for("index"))

    total_votes += 1
    current_index += 1

    # Log results every VOTES_PER_LOG votes
    if total_votes % VOTES_PER_LOG == 0:
        log_results()

    return redirect(url_for("index"))


# Function to log results
def log_results():
    with open(LOG_FILE, "w") as f:
        f.write(f"Total Votes: {total_votes}\n")
        f.write(f"JPEG Wins: {jpeg_wins}\n")
        f.write(f"Algorithm Wins: {algorithm_wins}\n")
        if total_votes > 0:
            jpeg_rate = (jpeg_wins / total_votes) * 100
            algorithm_rate = (algorithm_wins / total_votes) * 100
            f.write(f"JPEG Win Rate: {jpeg_rate:.2f}%\n")
            f.write(f"Algorithm Win Rate: {algorithm_rate:.2f}%\n")
        else:
            f.write("No votes yet.\n")


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
