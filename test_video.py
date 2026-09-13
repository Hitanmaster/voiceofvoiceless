import os
import sys
import json
import argparse
import cv2
import numpy as np
import torch
import mediapipe as mp
from model import BiLSTMAttentionSignClassifier


def extract_landmarks(results):
    # Left Hand (63)
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3, dtype=np.float32)

    # Right Hand (63)
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3, dtype=np.float32)

    # Upper Pose (12)
    if results.pose_landmarks:
        pose_lms = results.pose_landmarks.landmark
        upper_pose = []
        for idx in [11, 12, 13, 14]:
            if idx < len(pose_lms):
                lm = pose_lms[idx]
                upper_pose.extend([lm.x, lm.y, lm.z])
            else:
                upper_pose.extend([0.0, 0.0, 0.0])
        pose = np.array(upper_pose, dtype=np.float32)
    else:
        pose = np.zeros(4 * 3, dtype=np.float32)

    return np.concatenate([lh, rh, pose])


def predict_video(video_path, model, classes, device, target_frames=60):
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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = holistic.process(frame_rgb)
        features = extract_landmarks(results)
        raw_frames.append(features)

    cap.release()
    holistic.close()

    if not raw_frames:
        print("[!] No frames extracted from video.")
        return None

    raw_arr = np.array(raw_frames, dtype=np.float32)
    total_frames = raw_arr.shape[0]

    # Uniform resampling to target_frames (60)
    indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
    seq = raw_arr[indices]

    seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(seq_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_indices = np.argsort(probs)[::-1][:5]

    print("\n" + "=" * 50)
    print(f" 📹 VIDEO PREDICTION RESULTS: {os.path.basename(video_path)}")
    print("=" * 50)
    print(f"Total Video Frames Extracted: {total_frames} (Resampled to {target_frames})")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Sign Language AI Model on a Single Video File")
    parser.add_argument("video_path", type=str, help="Path to input .mp4 video file")
    parser.add_argument("--model", type=str, default="unified_sign_model.pt", help="Path to .pt model weights")
    parser.add_argument("--classes", type=str, default="classes.json", help="Path to classes.json")

    args = parser.parse_args()

    with open(args.classes, "r") as f:
        classes = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMAttentionSignClassifier(
        input_dim=138,
        hidden_dim=128,
        num_layers=2,
        num_classes=len(classes)
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    predict_video(args.video_path, model, classes, device)
