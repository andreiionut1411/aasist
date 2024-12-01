import os
import shutil

# Paths
txt_file = "LA_mini/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"  # Replace with the path to your .txt file
flac_dir = "LA_mini/ASVspoof2019_LA_train/flac"  # Replace with the directory containing .flac files
output_dir = "flac"  # Replace with a directory where you want to keep the valid files

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Step 1: Read the .txt file and extract valid filenames
with open(txt_file, "r") as f:
    valid_files = {line.split()[1] + ".flac" for line in f.readlines()}  # Extract the second column and add ".flac"

# Step 2: Iterate through the directory and keep only valid files
for file_name in os.listdir(flac_dir):
    if file_name in valid_files:
        # Move valid files to the output directory
        shutil.move(os.path.join(flac_dir, file_name), os.path.join(output_dir, file_name))
    else:
        # Optionally delete invalid files
        os.remove(os.path.join(flac_dir, file_name))

print(f"Filtered files are moved to {output_dir}")
