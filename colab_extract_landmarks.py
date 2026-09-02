# ==============================================================================
#  Indian Sign Language (ISL) Video-to-Landmark Feature Extractor for Google Colab
#  Converts downloaded .mp4 videos into uniform (60, 138) MediaPipe Holistic sequences.
# ==============================================================================

# ── CELL 1: Install Dependencies ──────────────────────────────────────────────
# In Colab, uncomment and run:
# !pip install -q mediapipe opencv-python-headless numpy tqdm matplotlib

# ── CELL 2: Mount Google Drive ────────────────────────────────────────────────
# In Colab, uncomment and run:
# from google.colab import drive
# drive.mount('/content/drive')

import os
import sys
import glob
import re
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# ── CELL 3: Configuration & Target Classes ────────────────────────────────────

# Path where your downloaded ISL .mp4 videos are stored in Google Drive
INPUT_VIDEOS_DIR = "/content/drive/MyDrive/ISL_Videos"

# Output directory for extracted .npy sequences
OUTPUT_DATASET_DIR = "/content/drive/MyDrive/SignLanguageAI/dataset"

# Sequence parameters (matches 1-Month Plan: 60 frames, 138 features)
TARGET_FRAMES = 60
TARGET_FEATURES = 138  # Left Hand (63) + Right Hand (63) + Upper Pose (12)

# Number of augmented variations to generate per video (recommended: 10 to 15)
# Because dictionary videos typically have 1-2 videos per word, this produces
# 10-15 distinct training sequences per class with temporal & spatial jitter.
AUGMENTATIONS_PER_VIDEO = 12

# Processing Mode:
# "TOP_35": Extracts 35 common everyday vocabulary signs (Recommended for fast, high-accuracy demo)
# "ALL":    Extracts all valid vocabulary videos found in the input folder (excluding story videos)
EXTRACTION_MODE = "TOP_35"

# Carefully selected 35 high-utility everyday signs verified in your dataset:
TOP_35_CLASSES = [
    "Thank You", "Help", "Sorry", "Welcome", "Good-1",
    "Water", "Food", "Eat", "Drink", "Father",
    "Mother-1", "Brother", "Sister", "Family", "Friend",
    "School", "Doctor", "Hospital", "Teacher", "Book",
    "Money", "House", "Home", "Work", "Time",
    "Day", "Sun", "Moon", "Car", "Bus",
    "Baby", "Boy", "Girl", "Dog", "Cat-1"
]

print(f"[*] Extraction Mode: {EXTRACTION_MODE}")
if EXTRACTION_MODE == "TOP_35":
    print(f"[*] Target Classes ({len(TOP_35_CLASSES)}): {TOP_35_CLASSES}")


# ── CELL 4: MediaPipe Holistic Feature Extraction Function ────────────────────

mp_holistic = mp.solutions.holistic

def extract_holistic_landmarks(results) -> np.ndarray:
    """
    Extracts 138 features from MediaPipe Holistic results:
    - Left Hand: 21 landmarks * 3 (x, y, z) = 63 features (zeros if absent)
    - Right Hand: 21 landmarks * 3 (x, y, z) = 63 features (zeros if absent)
    - Upper Pose: 4 landmarks (11: left shoulder, 12: right shoulder,
                              13: left elbow, 14: right elbow) * 3 = 12 features
    Total = 63 + 63 + 12 = 138 features.
    """
    # Left Hand (21 * 3 = 63)
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3, dtype=np.float32)

    # Right Hand (21 * 3 = 63)
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3, dtype=np.float32)

    # Upper Body Pose (4 * 3 = 12)
    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark
        # Indices 11, 12, 13, 14
        upper_pose = []
        for idx in [11, 12, 13, 14]:
            if idx < len(pose_landmarks):
                lm = pose_landmarks[idx]
                upper_pose.extend([lm.x, lm.y, lm.z])
            else:
                upper_pose.extend([0.0, 0.0, 0.0])
        pose = np.array(upper_pose, dtype=np.float32)
    else:
        pose = np.zeros(4 * 3, dtype=np.float32)

    return np.concatenate([lh, rh, pose])  # Shape: (138,)


def sample_frame_indices(total_frames: int, target_frames: int = 60) -> np.ndarray:
    """Uniformly samples or interpolates frame indices to reach target_frames."""
    if total_frames <= 0:
        return np.zeros(target_frames, dtype=int)
    if total_frames >= target_frames:
        return np.linspace(0, total_frames - 1, target_frames, dtype=int)
    else:
        # If video is shorter than 60 frames, repeat frames evenly
        return np.round(np.linspace(0, total_frames - 1, target_frames)).astype(int)


def process_video_to_raw_sequence(video_path: str, holistic) -> np.ndarray:
    """Reads a video file and extracts raw landmark features for all frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    raw_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        frame_features = extract_holistic_landmarks(results)
        raw_frames.append(frame_features)

    cap.release()
    if not raw_frames:
        return None

    raw_arr = np.array(raw_frames, dtype=np.float32)  # Shape: (N, 138)
    total_frames = raw_arr.shape[0]

    # Resample to exactly TARGET_FRAMES (60)
    indices = sample_frame_indices(total_frames, TARGET_FRAMES)
    resampled_sequence = raw_arr[indices]  # Shape: (60, 138)
    return resampled_sequence


# ── CELL 5: Sequence Data Augmentations ────────────────────────────────────────

def augment_sequence(seq: np.ndarray, aug_idx: int) -> np.ndarray:
    """
    Applies spatial jitter, scaling, and temporal speed variation to create
    realistic alternative samples for training data diversity.
    """
    augmented = seq.copy()

    # 1. Non-zero landmark mask (so we don't augment zero-padded missing hands)
    nonzero_mask = augmented != 0.0

    # 2. Gaussian coordinate jitter (slight tremor / position difference)
    noise_scale = 0.008 * (1.0 + (aug_idx % 3) * 0.5)
    noise = np.random.normal(0.0, noise_scale, size=augmented.shape).astype(np.float32)
    augmented[nonzero_mask] += noise[nonzero_mask]

    # 3. Spatial scale factor (slight variation in distance from camera: 0.92x to 1.08x)
    scale = np.random.uniform(0.92, 1.08)
    augmented[nonzero_mask] *= scale

    # 4. Temporal speed warping (speed up or slow down slightly)
    # Resample with a slight random curve
    if np.random.rand() > 0.3:
        curve = np.linspace(0, TARGET_FRAMES - 1, TARGET_FRAMES)
        warp = np.sin(np.linspace(0, np.pi, TARGET_FRAMES)) * np.random.uniform(-3, 3)
        warped_idx = np.clip(np.round(curve + warp), 0, TARGET_FRAMES - 1).astype(int)
        augmented = augmented[warped_idx]

    return augmented


# ── CELL 6: Main Batch Processing Pipeline ────────────────────────────────────

def clean_class_name(filename: str) -> str:
    """Extracts a normalized class name from video file name."""
    base = os.path.splitext(os.path.basename(filename))[0]
    # Remove trailing variant numbers like '_2', '_3'
    base = re.sub(r'_\d+$', '', base)
    return base.strip()


def run_pipeline():
    os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)

    # 1. Find all MP4 files
    all_videos = glob.glob(os.path.join(INPUT_VIDEOS_DIR, "*.mp4"))
    if not all_videos:
        print(f"[!] No MP4 files found in {INPUT_VIDEOS_DIR}")
        print("    Please ensure your Google Drive is mounted and the path is correct.")
        return

    print(f"[*] Found {len(all_videos)} total videos in Drive directory.")

    # 2. Filter videos according to selection mode
    target_video_map = {}  # class_name -> list of video file paths

    for vpath in all_videos:
        file_size = os.path.getsize(vpath)
        # Skip story videos (> 3.0 MB) or empty files (< 50 KB)
        if file_size > 3.0 * 1024 * 1024 or file_size < 50 * 1024:
            continue

        raw_name = os.path.splitext(os.path.basename(vpath))[0]
        class_name = clean_class_name(raw_name)

        if EXTRACTION_MODE == "TOP_35":
            # Match against target list (case-insensitive)
            matched = None
            for tc in TOP_35_CLASSES:
                if tc.lower() == class_name.lower() or tc.lower() == raw_name.lower():
                    matched = tc
                    break
            if matched:
                target_video_map.setdefault(matched, []).append(vpath)
        else:
            target_video_map.setdefault(class_name, []).append(vpath)

    print(f"[*] Selected {len(target_video_map)} classes for extraction.")
    for cls, vlist in list(target_video_map.items())[:10]:
        print(f"    - {cls}: {len(vlist)} video file(s)")
    if len(target_video_map) > 10:
        print(f"    ... and {len(target_video_map) - 10} more.")

    # 3. Initialize MediaPipe Holistic
    print("\n[*] Initializing MediaPipe Holistic...")
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )

    total_saved = 0
    class_counts = {}

    # 4. Extract landmarks & save .npy files
    for cls_name, video_paths in tqdm(target_video_map.items(), desc="Processing Signs"):
        safe_dir_name = re.sub(r'[\\/*?:"<>|]', '_', cls_name)
        cls_dir = os.path.join(OUTPUT_DATASET_DIR, safe_dir_name)
        os.makedirs(cls_dir, exist_ok=True)

        class_counts[cls_name] = 0

        for v_idx, vpath in enumerate(video_paths):
            base_seq = process_video_to_raw_sequence(vpath, holistic)
            if base_seq is None or base_seq.shape != (TARGET_FRAMES, TARGET_FEATURES):
                continue

            # Save base sequence
            base_file = os.path.join(cls_dir, f"{safe_dir_name}_v{v_idx}_orig.npy")
            np.save(base_file, base_seq)
            class_counts[cls_name] += 1
            total_saved += 1

            # Generate augmented versions
            for aug_i in range(AUGMENTATIONS_PER_VIDEO):
                aug_seq = augment_sequence(base_seq, aug_i)
                aug_file = os.path.join(cls_dir, f"{safe_dir_name}_v{v_idx}_aug{aug_i}.npy")
                np.save(aug_file, aug_seq)
                class_counts[cls_name] += 1
                total_saved += 1

    holistic.close()

    # ── CELL 7: Summary Report ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  EXTRACTION COMPLETE — DATASET SUMMARY")
    print("=" * 65)
    print(f"  Total Classes Processed : {len(class_counts)}")
    print(f"  Total Sequences Saved   : {total_saved}")
    print(f"  Sequence Tensor Shape   : ({TARGET_FRAMES}, {TARGET_FEATURES})")
    print(f"  Dataset Saved To        : {OUTPUT_DATASET_DIR}")
    print("=" * 65)

    print("\nPer-class sequence counts:")
    for cls, count in sorted(class_counts.items()):
        status = "[OK]" if count >= 10 else "[LOW]"
        print(f"  {status} {cls:<25s} : {count:3d} sequences")

    print("\n[NEXT STEP]: Add the 'idle' class, then train unified_sign_model.pt!")

if __name__ == "__main__":
    run_pipeline()
