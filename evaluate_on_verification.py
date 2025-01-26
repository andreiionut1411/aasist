def process_files(file1_path, file2_path, output_path, threshold):
    """
    Process the input files to generate the desired output file with normalized scores.

    Args:
        file1_path (str): Path to the verification file.
        file2_path (str): Path to the file that contains the scores.
        output_path (str): Path to the output file.
        threshold (float): Threshold for determining 1 or 0 based on the score.
    """
    score_dict = {}
    scores = []
    with open(file2_path, "r") as file2:
        for line in file2:
            filename, score = line.strip().split()
            score = float(score)
            score_dict[filename] = score
            scores.append(score)

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        raise ValueError("All scores are the same; normalization is not possible.")

    normalized_threshold = (threshold - min_score) / (max_score - min_score)

    # Normalize scores
    for key in score_dict:
        score_dict[key] = (score_dict[key] - min_score) / (max_score - min_score)

    # Process file1_path and write output with normalized scores
    with open(file1_path, "r") as file1, open(output_path, "w") as output_file:
        for line in file1:
            label, file1_path, file2_path = line.strip().split()
            file2_name = file2_path.split("/")[-1]

            score = score_dict.get(file2_name, None)
            if score is None:
                raise ValueError(f"Score for {file2_name} not found in {file2_path}")

            output_label = 1 if score > normalized_threshold else 0
            output_file.write(f"{output_label} {score:.4f}\n")


thresholds = {"wav2vec": 1.465143, "aasist_pretrained": 1.49328, "aasist": 1.282139, "RAWGAT-ST": 0.608895, "RAWNET": 0.006465, "multihead": 3.4528308, "vanilla_wav2vec": 3.0124712}
file1_path = "/home/andrei/facultate/licenta/combined_test_file.txt"
file2_path = "echo_test/aasist_init_results.txt"
output_path = "echo_test/aasist_results.txt"
threshold = thresholds['aasist']

process_files(file1_path, file2_path, output_path, threshold)