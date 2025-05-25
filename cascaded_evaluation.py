from tqdm import tqdm

spoofing_file = "rsc_test/wav2vec_results.txt"   # Contains: 0/1 score
verification_file = "rsc_test/titanet.txt"  # Contains: 0/1 score
ground_truth_file = "/home/andrei/facultate/licenta/rsc_combined_test_file.txt"  # Contains: 0/1 file1 file2

# Read all files line by line
with open(spoofing_file, "r") as f1, open(verification_file, "r") as f2, open(ground_truth_file, "r") as f3:
    spoofing_lines = f1.readlines()
    verification_lines = f2.readlines()
    ground_truth_lines = f3.readlines()

# Ensure all files have the same number of lines
assert len(spoofing_lines) == len(verification_lines) == len(ground_truth_lines), "Files must have the same number of lines"

correct = 0
total = len(ground_truth_lines)

# Process each line
for i in tqdm(range(total), desc="Processing lines"):
    # Extract the first number (0 or 1) from each file
    spoofing_result = int(spoofing_lines[i].strip().split()[0])
    verification_result = int(verification_lines[i].strip().split()[0])
    ground_truth = int(ground_truth_lines[i].strip().split()[0])

    # Compute combined result
    combined_result = 1 if (spoofing_result == 1 and verification_result == 1) else 0

    # Check against ground truth
    if combined_result == ground_truth:
        correct += 1

# Compute accuracy
accuracy = correct / total

print(f"Accuracy of the combined biometric system: {accuracy:.4f}")
