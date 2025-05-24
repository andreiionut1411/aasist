def calculate_accuracy(file_path, threshold):
    """
    Calculate the percentage of correctly classified positive and negative files.

    Args:
        file_path (str): Path to the file containing filenames and scores.
        threshold (float): The threshold for score classification.

    Returns:
        tuple: (positive_accuracy, negative_accuracy)
    """
    positive_correct = 0
    positive_total = 0
    negative_correct = 0
    negative_total = 0

    with open(file_path, 'r') as f:
        for line in f:
            filename, score = line.strip().split()
            score = float(score)

            # Extract speaker ID from filename (assuming format: user-contributed-<spkr_id>-id.wav)
            id = filename.split('.')[0].split('-')[3]

            if len(id) == 32:
                positive_total += 1
                if score >= threshold:
                    positive_correct += 1
            elif len(id) in [1, 2]:
                negative_total += 1
                if score < threshold:
                    negative_correct += 1

    positive_accuracy = (positive_correct / positive_total) * 100 if positive_total > 0 else 0
    negative_accuracy = (negative_correct / negative_total) * 100 if negative_total > 0 else 0

    return positive_accuracy, negative_accuracy


file_path = 'echo_test_noiseless/multihead_aasist_init_results.txt'
thresholds = {"wav2vec": 1.465143, "aasist_pretrained": 1.49328, "aasist": 1.282139, "RAWGAT-ST": 0.608895, "RAWNET": -0.006465, "multihead": 3.4528308, "vanilla_wav2vec": 3.0124712}

positive_accuracy, negative_accuracy = calculate_accuracy(file_path, thresholds["multihead"])
print(f"Positive Class Accuracy: {positive_accuracy:.2f}%")
print(f"Negative Class Accuracy: {negative_accuracy:.2f}%")