import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from model import BiLSTMAttentionSignClassifier

try:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def load_model(model_path="unified_sign_model.pt", classes_path="classes.json"):
    with open(classes_path, "r") as f:
        classes = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMAttentionSignClassifier(
        input_dim=138,
        hidden_dim=128,
        num_layers=2,
        num_classes=len(classes)
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, classes, device


def evaluate_npy_dataset(dataset_dir, model, classes, device, output_cm="confusion_matrix.png"):
    """
    Evaluates model accuracy on a directory containing .npy files.
    Expected structure:
      dataset_dir/
         ClassA/
             seq_001.npy (shape: 60, 138)
             ...
         ClassB/
             ...
    """
    y_true = []
    y_pred = []
    y_probs = []

    class_to_idx = {c: i for i, c in enumerate(classes)}

    print(f"[*] Scanning dataset directory: {dataset_dir}")
    total_files = 0

    for class_name in os.listdir(dataset_dir):
        class_folder = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_folder):
            continue

        if class_name not in class_to_idx:
            print(f"[!] Warning: Folder '{class_name}' is not in classes.json — skipping.")
            continue

        true_label_idx = class_to_idx[class_name]
        files = [f for f in os.listdir(class_folder) if f.endswith(".npy")]

        for f in files:
            file_path = os.path.join(class_folder, f)
            try:
                seq = np.load(file_path)  # Shape: (60, 138)
                if seq.shape != (60, 138):
                    # Pad or resample if needed
                    if len(seq.shape) == 2 and seq.shape[1] == 138:
                        indices = np.linspace(0, len(seq) - 1, 60, dtype=int)
                        seq = seq[indices]
                    else:
                        continue

                tensor_input = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(tensor_input)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    pred_label_idx = np.argmax(probs)

                y_true.append(true_label_idx)
                y_pred.append(pred_label_idx)
                y_probs.append(probs)
                total_files += 1
            except Exception as e:
                print(f"[!] Error reading {file_path}: {e}")

    if total_files == 0:
        print("[!] No valid .npy sequences found to evaluate.")
        return

    print_evaluation_results(y_true, y_pred, y_probs, classes, output_cm)


def print_evaluation_results(y_true, y_pred, y_probs, classes, output_cm="confusion_matrix.png"):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    acc = np.mean(y_true == y_pred) * 100.0

    # Top-3 Accuracy
    top3_correct = 0
    for true_idx, probs in zip(y_true, y_probs):
        top3_preds = np.argsort(probs)[-3:]
        if true_idx in top3_preds:
            top3_correct += 1
    top3_acc = (top3_correct / len(y_true)) * 100.0

    print("\n" + "=" * 65)
    print(" 🎯 MODEL ACCURACY EVALUATION REPORT")
    print("=" * 65)
    print(f"Total Evaluated Samples : {len(y_true)}")
    print(f"Top-1 Accuracy          : {acc:.2f}%")
    print(f"Top-3 Accuracy          : {top3_acc:.2f}%")
    print("=" * 65)

    if SKLEARN_AVAILABLE:
        present_indices = sorted(list(set(y_true) | set(y_pred)))
        present_names = [classes[i] for i in present_indices]
        print("\n📊 Detailed Classification Report (Precision / Recall / F1):")
        print(classification_report(y_true, y_pred, labels=present_indices, target_names=present_names, zero_division=0))

        # Save Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=present_indices)
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                    xticklabels=present_names, yticklabels=present_names)
        plt.title(f"Confusion Matrix (Top-1 Accuracy: {acc:.2f}%)")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_cm, dpi=200)
        print(f"\n[✓] Confusion matrix plot saved to: {output_cm}")
    else:
        print("\n[!] Tip: Install scikit-learn, matplotlib, and seaborn for confusion matrix visual plotting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Accuracy of Unified Sign Language BiLSTM Model")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Path to extracted landmark .npy dataset")
    parser.add_argument("--model", type=str, default="unified_sign_model.pt", help="Path to .pt model weights")
    parser.add_argument("--classes", type=str, default="classes.json", help="Path to classes.json")
    parser.add_argument("--output_cm", type=str, default="confusion_matrix.png", help="Output path for confusion matrix")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[!] Error: Model file '{args.model}' not found.")
        sys.exit(1)

    if not os.path.exists(args.classes):
        print(f"[!] Error: Classes file '{args.classes}' not found.")
        sys.exit(1)

    model, classes, device = load_model(args.model, args.classes)
    print(f"[✓] Loaded model with {len(classes)} classes on {device}.")

    if os.path.exists(args.dataset_dir):
        evaluate_npy_dataset(args.dataset_dir, model, classes, device, args.output_cm)
    else:
        print(f"[!] Dataset directory '{args.dataset_dir}' not found.")
        print("    You can run accuracy test on extracted .npy features, or use 'test_video.py' to test video files.")
