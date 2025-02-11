import os
import shutil

# Define the base directory where the experiment folders are located
root_dir = "/Users/yujie4/Documents/Code/PycharmProjects/Bipartite-nonclasscality/Data"

# Define subdirectories for different file types within the same Data folder
counts_dir = os.path.join(root_dir, "Counts")
prob_dir = os.path.join(root_dir, "Probabilities")
tomo_dir = os.path.join(root_dir, "Tomography")

# Create the subdirectories if they do not exist
os.makedirs(counts_dir, exist_ok=True)
os.makedirs(prob_dir, exist_ok=True)
os.makedirs(tomo_dir, exist_ok=True)

# Loop through each folder in the root directory
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)

    # Ensure it's a directory
    if os.path.isdir(folder_path):
        # Extract the last few digits from the folder name
        folder_suffix = folder.split("_")[-1]  # Get the last part after the last underscore

        # Define file paths in the source directory
        counts_file = os.path.join(folder_path, "counts.npy")
        prob_file = os.path.join(folder_path, "probabilities.npy")
        rho_file = os.path.join(folder_path, "rho_36_states.npy")

        # Check if each file exists before renaming and copying
        if os.path.exists(counts_file):
            shutil.copy(counts_file, os.path.join(counts_dir, f"counts_{folder_suffix}.npy"))
        if os.path.exists(prob_file):
            shutil.copy(prob_file, os.path.join(prob_dir, f"prob_{folder_suffix}.npy"))
        if os.path.exists(rho_file):
            shutil.copy(rho_file, os.path.join(tomo_dir, f"tomo_{folder_suffix}.npy"))

print("Sorting Completed")