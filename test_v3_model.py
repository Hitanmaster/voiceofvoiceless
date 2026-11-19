"""
test_v3_model.py — Diagnostic & Testing Suite for unified_sign_model_v3.pt (Vāksetu 301 Classes)
Voice of the Voiceless Project
"""

import os
import sys
import json
import torch
import numpy as np

from model import BiLSTMAttentionSignClassifier

def run_diagnostics():
    print("=" * 70)
    print(" 🔬 V3 MODEL (VĀKSETU 301-CLASS) IN-DEPTH DIAGNOSTIC & TEST SUITE")
    print("=" * 70)

    # 1. Load Config
    cfg_file = "model_config.json"
    if not os.path.exists(cfg_file):
        print(f"[!] Error: {cfg_file} missing.")
        return

    with open(cfg_file) as f:
        cfg = json.load(f)

    classes = cfg["classes"]
    num_classes = len(classes)
    model_file = cfg.get("model_file", "unified_sign_model_v3.pt")

    print(f"\n[1] METADATA & ARCHITECTURE SPECS:")
    print(f"    - Model Checkpoint: {model_file} ({os.path.getsize(model_file)/1024/1024:.2f} MB)")
    print(f"    - Model Version: v{cfg.get('version')} ({cfg.get('feature')})")
    print(f"    - Input Features: {cfg.get('input_dim')} dimensions (253 base + 253 velocity)")
    print(f"    - Sequence Length: {cfg.get('seq_len')} frames (sampled from {cfg.get('window_frames', 45)}-frame buffer)")
    print(f"    - Classes Count: {num_classes}")
    print(f"    - Stated Test Accuracy (Vāksetu dataset): {cfg.get('test_accuracy')}%")
    print(f"    - Stated Other-Signer Accuracy (INCLUDE dataset): {cfg.get('include_other_signer_accuracy')}%")

    # 2. Checkpoint & Weight Integrity
    print(f"\n[2] CHECKPOINT & WEIGHT INTEGRITY:")
    state = torch.load(model_file, map_location="cpu")
    has_nan = False
    has_inf = False
    max_weight = -1e9
    min_weight = 1e9

    for k, v in state.items():
        if torch.isnan(v).any():
            has_nan = True
        if torch.isinf(v).any():
            has_inf = True
        max_weight = max(max_weight, float(v.max()))
        min_weight = min(min_weight, float(v.min()))

    print(f"    - Total State Tensors: {len(state)}")
    print(f"    - NaN weights detected: {has_nan}")
    print(f"    - Inf weights detected: {has_inf}")
    print(f"    - Weight Value Range: [{min_weight:.4f}, {max_weight:.4f}] -> Normal range")

    # Load Model
    model = BiLSTMAttentionSignClassifier(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg.get("hidden_dim", 128),
        num_layers=2,
        num_classes=num_classes
    )
    model.load_state_dict(state)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"    - Total Parameter Count: {total_params:,}")

    # 3. Behavioral Tests on Input Scenarios
    print(f"\n[3] INFERENCE BEHAVIOR UNDER REAL-WORLD CAMERA SCENARIOS:")

    scenarios = {}

    # Scenario A: Empty frame (no hands, no face detected, proximity=1.0)
    x_empty = torch.zeros(1, 20, 506)
    x_empty[:, :, 252] = 1.0  # proximity default
    scenarios["A. Empty / No Person (Camera seeing blank wall)"] = x_empty

    # Scenario B: Pure random noise (camera artifact or noisy background)
    scenarios["B. High-Frequency Video Noise"] = torch.randn(1, 20, 506) * 0.1

    # Scenario C: Hands resting in lap / no movement (static pose)
    x_rest = torch.zeros(1, 20, 506)
    x_rest[:, :, 63:126] = 0.15 # small hand values
    x_rest[:, :, 252] = 0.8
    # velocities all 0
    scenarios["C. Hands Resting Still (Velocities = 0)"] = x_rest

    # Scenario D: Active Right Hand Motion
    x_right = torch.zeros(1, 20, 506)
    motion = torch.sin(torch.linspace(0, 3.14, 20)).unsqueeze(1)
    x_right[:, :, 63:126] = motion.repeat(1, 63) * 0.4
    x_right[:, :, 189:252] = 0.3
    x_right[:, :, 252] = 0.25 # close to face
    # Compute velocity
    x_right[:, 1:, 253:] = x_right[:, 1:, :253] - x_right[:, :-1, :253]
    scenarios["D. Smooth Right-Hand Gesture toward Face"] = x_right

    # Scenario E: Same gesture, but on Left Hand (e.g. left-handed signer or inverted mirror)
    x_left = torch.zeros(1, 20, 506)
    x_left[:, :, 0:63] = motion.repeat(1, 63) * 0.4
    x_left[:, :, 126:189] = 0.3
    x_left[:, :, 252] = 0.25
    x_left[:, 1:, 253:] = x_left[:, 1:, :253] - x_left[:, :-1, :253]
    scenarios["E. Same Gesture on Left Hand (Mirror / Dominance test)"] = x_left

    for title, inp in scenarios.items():
        with torch.no_grad():
            logits = model(inp)
            probs = torch.softmax(logits, dim=1)[0]
            top3 = torch.topk(probs, 3)

        print(f"\n  👉 {title}:")
        for rank, (idx, prob) in enumerate(zip(top3.indices, top3.values), start=1):
            lbl = classes[idx]
            p = prob.item() * 100
            print(f"      #{rank}: {lbl:<16} ({p:5.2f}%)")

    # 4. Top Confusions from Training Logs
    print(f"\n[4] DATASET CONFUSION ANALYSIS (FROM TRAINING):")
    if os.path.exists("confusion_top_errors.txt"):
        with open("confusion_top_errors.txt") as f:
            lines = [line.strip() for line in f if line.strip()][:12]
        print("    Most frequent class confusions in test set:")
        for l in lines:
            print(f"      • {l}")

    print("\n" + "=" * 70)
    print(" 🏁 DIAGNOSTIC COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    run_diagnostics()

