import os
import sys
import json
import argparse
import cv2
import numpy as np
import torch
import mediapipe as mp

from model import BiLSTMAttentionSignClassifier
from feature_utils import (
    resolve_active_model, get_feature_extractor, infer_input_dim,
)


def resample_sequence(raw_arr: np.ndarray, target_frames: int) -> np.ndarray:
    """Uniformly resample (N, F) -> (target_frames, F)."""
    total_frames = raw_arr.shape[0]
    if total_frames <= 0:
        raise ValueError("Empty sequence")
    indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
    return raw_arr[indices]


def predict_video(video_path, model, classes, device, target_frames=60, extract=None):
    if not os.path.exists(video_path):
        print(f"[!] Error: Video file not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Error: Could not open video: {video_path}")
        return None

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )

    raw_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = holistic.process(frame_rgb)
        features = extract(results)
        raw_frames.append(features)

    cap.release()
    holistic.close()

    if not raw_frames:
        print("[!] No frames extracted from video.")
        return None

    raw_arr = np.array(raw_frames, dtype=np.float32)
    seq = resample_sequence(raw_arr, target_frames)

    seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(seq_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_indices = np.argsort(probs)[::-1][:5]

    print("\n" + "=" * 50)
    print(f" 📹 VIDEO PREDICTION RESULTS: {os.path.basename(video_path)}")
    print("=" * 50)
    print(f"Total Video Frames Extracted: {raw_arr.shape[0]} (Resampled to {target_frames})")
    print("\n🏆 Top-5 Predictions:")
    for rank, idx in enumerate(top_indices, start=1):
        label = classes[idx]
        conf = probs[idx] * 100.0
        bar = "█" * int(conf / 5)
        print(f"  {rank}. {label:<16} : {conf:6.2f}%  |{bar}")
    print("=" * 50)

    top_label = classes[top_indices[0]]
    top_conf = probs[top_indices[0]]
    return top_label, top_conf


def main():
    parser = argparse.ArgumentParser(description="Test Sign Language AI Model on a Single Video File")
    parser.add_argument("video_path", type=str, help="Path to input .mp4 video file")
    parser.add_argument("--model", type=str, default=None, help="Path to .pt model weights (default: auto-detect v2/legacy)")
    parser.add_argument("--classes", type=str, default=None, help="Path to classes.json (default: from active model config)")

    args = parser.parse_args()

    # Auto-detect active model (v2 config first, legacy fallback)
    model_path, cfg = resolve_active_model()
    if args.model:
        model_path = args.model
    classes = cfg["classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(model_path, map_location=device)
    model = BiLSTMAttentionSignClassifier(
        input_dim=infer_input_dim(state),
        hidden_dim=128,
        num_layers=2,
        num_classes=len(classes)
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    extract = get_feature_extractor(cfg["version"])
    print(f"[*] Using {os.path.basename(model_path)} (v{cfg['version']}, input_dim={cfg['input_dim']})")

    predict_video(args.video_path, model, classes, device, target_frames=cfg.get("seq_len", 60), extract=extract)


if __name__ == "__main__":
    main()
