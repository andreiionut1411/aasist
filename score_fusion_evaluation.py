from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
import numpy as np

def evaluate_performance(ground_truth_file, prediction_file):
    """
    Evaluate speaker verification performance based on ground truth and prediction files.

    Args:
        ground_truth_file (str): Path to the ground truth file (1/0 file1 file2).
        prediction_file (str): Path to the prediction file (1/0 score).
        threshold (float): Threshold for classification.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1-score.
    """
    ground_truths = []
    predictions = []

    with open(ground_truth_file, "r") as gt_file:
        for line in gt_file:
            label, _, _ = line.strip().split()
            ground_truths.append(int(label))

    with open(prediction_file, "r") as pred_file:
        for line in pred_file:
            predicted_label, _ = line.strip().split()
            predictions.append(int(predicted_label))

    accuracy = accuracy_score(ground_truths, predictions)
    precision = precision_score(ground_truths, predictions)
    recall = recall_score(ground_truths, predictions)
    f1 = f1_score(ground_truths, predictions)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }


def compute_eer(fused_scores, labels):
    """
    Compute Equal Error Rate (EER) and corresponding threshold.

    Args:
        fused_scores (list): List of fused prediction scores.
        labels (list): List of ground truth labels (0/1).

    Returns:
        eer (float): Equal Error Rate.
        threshold (float): Threshold at which EER occurs.
    """
    fpr, tpr, thresholds = roc_curve(labels, fused_scores, pos_label=1)

    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    threshold = thresholds[eer_idx]

    return eer, threshold


def fuse_and_evaluate(gt_file, pred_file1, pred_file2, output_path):
    """
    Fuse prediction scores from two files, compute new threshold using EER, and save results.

    Args:
        gt_file (str): Path to the ground truth file.
        pred_file1 (str): Path to the first prediction file.
        pred_file2 (str): Path to the second prediction file.
        output_path (str): Path to save the fused results.
    """
    labels = []
    with open(gt_file, "r") as gt:
        for line in gt:
            label, _, _ = line.strip().split()
            labels.append(int(label))

    scores1 = []
    scores2 = []
    with open(pred_file1, "r") as f1, open(pred_file2, "r") as f2:
        for line1, line2 in zip(f1, f2):
            _, score1 = line1.strip().split()
            _, score2 = line2.strip().split()
            scores1.append(float(score1))
            scores2.append(float(score2))

    fused_scores = [(s1 + s2) for s1, s2 in zip(scores1, scores2)]
    eer, threshold = compute_eer(fused_scores, labels)

    with open(output_path, "w") as out:
        for score, label in zip(fused_scores, labels):
            predicted_label = 1 if score > threshold else 0
            out.write(f"{label} {score:.4f} {predicted_label}\n")

    print(f"Fused results saved to {output_path}")
    print(f"Equal Error Rate (EER): {eer:.4f}")
    print(f"Threshold at EER: {threshold:.4f}")


ground_truth_file = "/home/andrei/facultate/licenta/rsc_combined_test_file.txt"
prediction_file = "rsc_test/titanet.txt"
spoof_file = "rsc_test/wav2vec_results.txt"
# results = evaluate_performance(ground_truth_file, prediction_file)

# print("Evaluation Results:")
# for metric, value in results.items():
#     print(f"{metric}: {value:.4f}")

fuse_and_evaluate(ground_truth_file, prediction_file, spoof_file, 'ceva.txt')