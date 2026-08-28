"""
collect_dataset.py — Unified Holistic Dataset Collector (v2.0)

Collects 30-frame sequences with 138 features per frame:
  - Left hand  : 21 landmarks × 3 (x,y,z) = 63 features
  - Right hand : 21 landmarks × 3 (x,y,z) = 63 features
  - Upper body : 4 joints (shoulders+elbows) × 3 = 12 features
  Total: 138 features per frame, (30, 138) per sequence → saved as .npy

Usage:
    python collect_dataset.py
    → Shows menu of all signs
    → Records 30 sequences with countdown for each sign
    → Saves to dataset/<sign_name>/seq_XXXX.npy
"""

import cv2
import os
import time
import numpy as np
import mediapipe as mp

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR       = "dataset"
SEQUENCE_LEN   = 30    # frames per sequence (~1 second @ 30fps)
NUM_SEQUENCES  = 30    # sequences to collect per sign per session
COUNTDOWN_SECS = 2     # seconds of countdown before each sequence

mp_holistic  = mp.solutions.holistic
mp_drawing   = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

# ── Complete ISL-inspired vocabulary ─────────────────────────────────────────
# Priority 1: Basic Needs (most important for real deaf users)
BASIC_NEEDS = [
    "help", "water", "food", "bathroom", "doctor", "pain",
    "medicine", "emergency",
]
# Priority 2: Common Communication
COMMON = [
    "yes", "no", "hello", "goodbye", "thank_you", "please",
    "sorry", "good", "bad", "more", "stop",
]
# Priority 3: Family
FAMILY = ["mother", "father", "friend"]
# Priority 4: Alphabet (static poses via sequence model — no mode switch)
ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
# Priority 5: Idle / Background
SPECIAL = ["idle"]

ALL_SIGNS = BASIC_NEEDS + COMMON + FAMILY + ALPHABET + SPECIAL


# ── Feature Extraction ───────────────────────────────────────────────────────
def extract_holistic_features(results) -> np.ndarray:
    """
    Extract 138-dimensional feature vector from MediaPipe Holistic results.

    Layout:
        [  0 – 62 ] Left hand  (zeros if not detected)
        [ 63 –125 ] Right hand (zeros if not detected)
        [126 –137 ] Upper body: landmarks 11,12 (shoulders) + 13,14 (elbows)
    """
    # Left hand — 63 features or zeros
    if results.left_hand_landmarks:
        lh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        lh = np.zeros(63, dtype=np.float32)

    # Right hand — 63 features or zeros
    if results.right_hand_landmarks:
        rh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        rh = np.zeros(63, dtype=np.float32)

    # Upper body pose — landmarks 11, 12 (shoulders) + 13, 14 (elbows)
    pose_indices = [11, 12, 13, 14]
    if results.pose_landmarks:
        pose = np.array(
            [
                [results.pose_landmarks.landmark[i].x,
                 results.pose_landmarks.landmark[i].y,
                 results.pose_landmarks.landmark[i].z]
                for i in pose_indices
            ],
            dtype=np.float32,
        ).flatten()
    else:
        pose = np.zeros(12, dtype=np.float32)

    return np.concatenate([lh, rh, pose])   # shape: (138,)


# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_landmarks(frame, results):
    """Draw hand + pose skeleton on frame."""
    # Left hand
    mp_drawing.draw_landmarks(
        frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )
    # Right hand
    mp_drawing.draw_landmarks(
        frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )
    # Pose (upper body)
    mp_drawing.draw_landmarks(
        frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(80, 110, 10), thickness=1, circle_radius=1),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(80, 256, 121), thickness=1, circle_radius=1),
    )


# ── Sequence Collection ───────────────────────────────────────────────────────
def collect_sign(sign_name: str, num_sequences: int = NUM_SEQUENCES):
    """
    Collect `num_sequences` of 30-frame sequences for a given sign.

    Each sequence saved as:  dataset/<sign_name>/seq_XXXX.npy
    Shape of each .npy file: (30, 138)
    """
    sign_dir = os.path.join(DATA_DIR, sign_name)
    os.makedirs(sign_dir, exist_ok=True)

    # Count existing sequences to avoid overwriting
    existing_files = sorted([f for f in os.listdir(sign_dir) if f.endswith(".npy")])
    start_idx      = len(existing_files)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera cannot open. Is another app using it?")
        return

    print(f"\n{'='*55}")
    print(f"  Collecting: '{sign_name}'  |  Target: {num_sequences} sequences")
    print(f"  Already collected: {start_idx}  |  Will add: {num_sequences}")
    print(f"{'='*55}")
    print("  Press 'q' at any time to stop early.\n")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        for seq_idx in range(start_idx, start_idx + num_sequences):
            # ── Countdown before each sequence ────────────────────────────
            for count in range(COUNTDOWN_SECS, 0, -1):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                # Dark overlay for text readability
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                              (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

                cv2.putText(frame, f"Sign: {sign_name}", (15, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.putText(frame, f"Sequence {seq_idx + 1}/{start_idx + num_sequences}",
                            (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Get ready... {count}",
                            (frame.shape[1]//2 - 120, frame.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

                cv2.imshow("Collect Dataset — Voiceless to Voice", frame)
                if cv2.waitKey(1000) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    print(f"\n[Stopped early] Saved {seq_idx - start_idx} sequences.")
                    return

            # ── Record SEQUENCE_LEN frames ─────────────────────────────────
            sequence = []
            hand_detected_count = 0

            for frame_idx in range(SEQUENCE_LEN):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                # Run holistic detection
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_rgb.flags.writeable = False
                results = holistic.process(img_rgb)
                img_rgb.flags.writeable = True

                # Draw skeleton
                draw_landmarks(frame, results)

                # Extract + store features
                features = extract_holistic_features(results)
                sequence.append(features)

                # Count frames where at least one hand detected
                if results.left_hand_landmarks or results.right_hand_landmarks:
                    hand_detected_count += 1

                # ── HUD ───────────────────────────────────────────────────
                progress = (frame_idx + 1) / SEQUENCE_LEN
                bar_w    = frame.shape[1] - 40
                cv2.rectangle(frame, (20, frame.shape[0] - 30),
                              (20 + bar_w, frame.shape[0] - 10), (50, 50, 50), -1)
                cv2.rectangle(frame, (20, frame.shape[0] - 30),
                              (20 + int(bar_w * progress), frame.shape[0] - 10),
                              (0, 220, 100), -1)

                hand_status = "✓ Hand detected" if (results.left_hand_landmarks or
                              results.right_hand_landmarks) else "✗ NO HAND!"
                status_color = (0, 255, 0) if (results.left_hand_landmarks or
                               results.right_hand_landmarks) else (0, 0, 255)

                cv2.putText(frame, f"REC  {sign_name}  [{frame_idx+1}/{SEQUENCE_LEN}]",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, hand_status, (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                # Red recording dot
                cv2.circle(frame, (frame.shape[1] - 25, 22), 10, (0, 0, 255), -1)

                cv2.imshow("Collect Dataset — Voiceless to Voice", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # ── Save sequence ──────────────────────────────────────────────
            seq_array  = np.array(sequence, dtype=np.float32)  # (30, 138)
            save_path  = os.path.join(sign_dir, f"seq_{seq_idx:04d}.npy")
            np.save(save_path, seq_array)

            hand_pct = hand_detected_count / SEQUENCE_LEN * 100
            quality  = "✓ Good" if hand_pct >= 70 else "⚠ Low hand visibility"
            print(f"  [{seq_idx + 1 - start_idx:2d}/{num_sequences}] Saved seq_{seq_idx:04d}.npy "
                  f"| Hand visible {hand_pct:.0f}%  {quality}")

    cap.release()
    cv2.destroyAllWindows()
    total = len([f for f in os.listdir(sign_dir) if f.endswith(".npy")])
    print(f"\n✓ Done! '{sign_name}' now has {total} sequences in {sign_dir}/")


# ── Dataset Summary ───────────────────────────────────────────────────────────
def show_dataset_summary():
    """Print a summary of collected sequences per sign."""
    if not os.path.exists(DATA_DIR):
        print("[INFO] No dataset folder found yet.")
        return
    print(f"\n{'─'*50}")
    print(f"  DATASET SUMMARY  ({DATA_DIR}/)")
    print(f"{'─'*50}")
    total_seqs = 0
    for sign in sorted(os.listdir(DATA_DIR)):
        sign_dir = os.path.join(DATA_DIR, sign)
        if not os.path.isdir(sign_dir):
            continue
        count = len([f for f in os.listdir(sign_dir) if f.endswith(".npy")])
        bar   = "█" * min(count, 30) + "░" * max(0, 30 - count)
        print(f"  {sign:<15s}  {bar}  {count:3d} seqs")
        total_seqs += count
    print(f"{'─'*50}")
    print(f"  Total sequences: {total_seqs} | Total classes: "
          f"{len([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])}")


# ── Main Menu ─────────────────────────────────────────────────────────────────
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    while True:
        print(f"\n{'='*55}")
        print("  SIGN LANGUAGE DATASET COLLECTOR  (v2.0)")
        print(f"{'='*55}")
        print("  1. Collect a specific sign")
        print("  2. Collect all BASIC NEEDS signs  (recommended first)")
        print("  3. Collect all COMMON signs")
        print("  4. Collect IDLE (background) class")
        print("  5. Collect EVERYTHING (all 35+ signs)")
        print("  6. Show dataset summary")
        print("  0. Exit")
        print(f"{'='*55}")

        choice = input("\nChoice (0-6): ").strip()

        if choice == "1":
            print("\nAvailable signs:")
            for i, s in enumerate(ALL_SIGNS):
                print(f"  {s}", end="\t")
                if (i + 1) % 8 == 0:
                    print()
            print()
            sign = input("Enter sign name: ").strip()
            if sign:
                collect_sign(sign)
        elif choice == "2":
            for sign in BASIC_NEEDS:
                print(f"\n>>> Next: '{sign}' <<<")
                input("Press Enter when ready...")
                collect_sign(sign)
        elif choice == "3":
            for sign in COMMON:
                print(f"\n>>> Next: '{sign}' <<<")
                input("Press Enter when ready...")
                collect_sign(sign)
        elif choice == "4":
            print("\n[IDLE class] Wave hands randomly, rest them, look away — anything NOT a sign.")
            input("Press Enter when ready...")
            collect_sign("idle")
        elif choice == "5":
            for sign in ALL_SIGNS:
                print(f"\n>>> Next: '{sign}' <<<")
                input("Press Enter when ready (or Ctrl+C to skip)...")
                try:
                    collect_sign(sign)
                except KeyboardInterrupt:
                    print(f"  Skipped '{sign}'")
        elif choice == "6":
            show_dataset_summary()
        elif choice == "0":
            print("\nExiting collector. Good luck with training!")
            break
        else:
            print("Invalid choice. Enter 0–6.")


if __name__ == "__main__":
    main()
