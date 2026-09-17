# ==============================================================================
#  ISL v2 Landmark Extractor — Google Colab (GPU not required for this step)
#  Converts Drive .mp4 videos into (60, 150) NORMALIZED landmark sequences.
#
#  Usage: paste each CELL into a separate Colab cell and run in order.
#  Output: /content/drive/MyDrive/SignLanguageAI/dataset_v2/<Class>/*.npy
# ==============================================================================


# ═══════════════════ CELL 1: Install dependencies (auto-detects environment) ═════════
# NOTE: Colab defaults to NumPy 2.x, but legacy MediaPipe requires NumPy 1.x and protobuf<4.26.
# This cell installs compatible versions and tests the Holistic API.
import sys, subprocess, os

def _pip(*pkgs):
    return subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", *pkgs],
                          capture_output=True, text=True)

print(f"[*] Python {sys.version.split()[0]}")

# Ensure NumPy 1.x is installed (MediaPipe C-bindings fail under NumPy 2.x)
import numpy as _np
_need_restart = False
if int(_np.__version__.split(".")[0]) >= 2:
    print("[*] Downgrading numpy from 2.x to <2.0.0 for MediaPipe compatibility...")
    _pip("numpy<2.0.0")
    _need_restart = True

CANDIDATES = [
    ["mediapipe==0.10.14", "numpy<2.0.0", "protobuf<4.26.0", "opencv-python-headless", "tqdm"],
    ["mediapipe==0.10.21", "numpy<2.0.0", "protobuf<4.26.0", "opencv-python-headless", "tqdm"],
    ["mediapipe==0.10.18", "numpy<2.0.0", "protobuf<4.26.0", "opencv-python-headless", "tqdm"],
    ["mediapipe", "numpy<2.0.0", "opencv-python-headless", "tqdm"],
]

_ok = False
for _pkgs in CANDIDATES:
    _cand_name = _pkgs[0]
    _r = _pip(*_pkgs)
    if _r.returncode != 0:
        continue
    try:
        sys.modules.pop("mediapipe", None)
        sys.modules.pop("mediapipe.python", None)
        sys.modules.pop("mediapipe.python.solutions", None)
        try:
            import mediapipe.python.solutions.holistic as _h_mod
            _ok = True
            print(f"  [OK] {_cand_name} (Holistic solutions API verified)")
            break
        except Exception:
            import mediapipe as _mp
            if hasattr(_mp, "solutions") and hasattr(_mp.solutions, "holistic"):
                _ok = True
                print(f"  [OK] {_cand_name} (mp.solutions.holistic verified)")
                break
    except Exception as _e:
        print(f"  [skip] {_cand_name}: {_e}")

if _need_restart:
    print("\n[!] IMPORTANT: NumPy was downgraded. You MUST restart the runtime session.")
    print("    Click: Runtime -> Restart session (or Runtime -> Restart runtime), then re-run from CELL 2.")
else:
    print("[OK] Dependencies verified. Continue to CELL 2.")



# ═══════════════════════════ CELL 2: Mount Google Drive ═══════════════════════
from google.colab import drive
drive.mount('/content/drive')


# ═══════════════════════════ CELL 3: Configuration ════════════════════════════
import os
import re
import glob
import cv2
import numpy as np
import mediapipe as mp
from tqdm.auto import tqdm

# Where your downloaded ISL .mp4 videos live in Drive
INPUT_VIDEOS_DIR = "/content/drive/MyDrive/ISL_Videos"

# Output dataset directory
OUTPUT_DATASET_DIR = "/content/drive/MyDrive/SignLanguageAI/dataset_v2"

TARGET_FRAMES = 60      # sequence length
TARGET_FEATURES = 150   # v2 feature dim (60+3+60+3+24)

# Skip story videos (> 3 MB) and broken/tiny files (< 50 KB)
MAX_VIDEO_BYTES = 3.0 * 1024 * 1024
MIN_VIDEO_BYTES = 50 * 1024

# 35 everyday vocabulary signs (matches classes.json minus 'idle')
TOP_35_CLASSES = [
    "Thank You", "Help", "Sorry", "Welcome", "Good-1",
    "Water", "Food", "Eat", "Drink", "Father",
    "Mother-1", "Brother", "Sister", "Family", "Friend",
    "School", "Doctor", "Hospital", "Teacher", "Book",
    "Money", "House", "Home", "Work", "Time",
    "Day", "Sun", "Moon", "Car", "Bus",
    "Baby", "Boy", "Girl", "Dog", "Cat-1"
]

# Variant aliasing: "Mother-2"/"Mother_3" videos are the SAME sign as "Mother-1"
# -> they get merged into one class, giving more REAL samples per class.
VARIANT_ALIASING = True

# Process every vocabulary video found (overrides TOP_35 list). Keep False for
# the focused 35-class dataset; True only if you trained on all classes.
EXTRACT_ALL = False


def base_sign_name(name: str) -> str:
    """Normalize 'Mother-2' / 'Mother_3' -> 'Mother' (variant aliasing)."""
    base = re.sub(r'[_\-]\d+$', '', name).strip()
    return base


def canonical_class_name(name: str, known: set) -> str:
    """Map a file stem to a canonical class label (e.g. 'Mother' -> 'Mother-1')."""
    if not VARIANT_ALIASING:
        return name
    base = base_sign_name(name)
    # exact match first, then base match (case-insensitive)
    for k in known:
        if k.lower() == base.lower():
            return k
    for k in known:
        if base_sign_name(k).lower() == base.lower():
            return k
    return name  # no alias found; use raw name


KNOWN_CLASSES = set(TOP_35_CLASSES)
print(f"[*] Extraction mode: {'ALL vocabulary' if EXTRACT_ALL else 'TOP_35'}")
print(f"[*] Input : {INPUT_VIDEOS_DIR}")
print(f"[*] Output: {OUTPUT_DATASET_DIR}")


# ═══════════════════ CELL 4: v2 feature extraction (normalized) ═══════════════
POSE_IDS = (11, 12, 13, 14, 15, 16, 23, 24)  # shoulders, elbows, wrists, hips
DEFAULT_SHOULDER_SCALE = 0.2


def extract_features_v2(results) -> np.ndarray:
    """(150,) scale- & position-invariant features. Must match local feature_utils.py."""
    if getattr(results, "pose_landmarks", None):
        plms = results.pose_landmarks.landmark

        def P(i):
            p = plms[i] if i < len(plms) else None
            return np.array([p.x, p.y, p.z], dtype=np.float32) if p else np.zeros(3, dtype=np.float32)
    else:
        def P(i):
            return np.zeros(3, dtype=np.float32)

    l_sh, r_sh = P(11), P(12)
    mid_sh = (l_sh + r_sh) * 0.5
    scale = float(np.linalg.norm((l_sh - r_sh)[:2]))
    if scale < 1e-3:
        scale = DEFAULT_SHOULDER_SCALE

    def hand_block(landmarks):
        if not landmarks:
            return np.zeros(63, dtype=np.float32)
        pts = np.array([[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32)
        if pts.shape[0] != 21:
            fixed = np.zeros((21, 3), dtype=np.float32)
            fixed[:min(len(pts), 21)] = pts[:21]
            pts = fixed
        wrist = pts[0]
        shape = ((pts[1:] - wrist) / scale).flatten()   # (60,)
        wpos = (wrist - mid_sh) / scale                 # (3,)
        return np.concatenate([shape, wpos]).astype(np.float32)

    lh = hand_block(getattr(results, "left_hand_landmarks", None))
    rh = hand_block(getattr(results, "right_hand_landmarks", None))
    pose = np.concatenate([(P(i) - mid_sh) / scale for i in POSE_IDS]).astype(np.float32)
    return np.concatenate([lh, rh, pose])               # (150,)


def process_video(video_path: str, holistic) -> np.ndarray | None:
    """Video -> (60, 150) resampled normalized sequence (or None)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    raw = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        try:
            results = holistic.process(rgb)
        except Exception:
            continue
        raw.append(extract_features_v2(results))
    cap.release()
    if not raw:
        return None
    arr = np.array(raw, dtype=np.float32)
    idx = np.linspace(0, arr.shape[0] - 1, TARGET_FRAMES, dtype=int)
    return arr[idx]


try:
    import mediapipe.python.solutions.holistic as mp_holistic
except (AttributeError, ImportError):
    try:
        import mediapipe as mp
        mp_holistic = mp.solutions.holistic
    except Exception as e:
        raise ImportError(
            "MediaPipe Holistic could not be loaded. Please ensure you ran CELL 1, "
            "then restarted the runtime session via 'Runtime -> Restart session', "
            "and re-ran from CELL 2."
        ) from e

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)
print("[OK] MediaPipe Holistic initialized successfully!")


# ═══════════════════════ CELL 5: Run batch extraction ═════════════════════════
os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)
all_videos = glob.glob(os.path.join(INPUT_VIDEOS_DIR, "**", "*.mp4"), recursive=True)
print(f"[*] Found {len(all_videos)} videos in {INPUT_VIDEOS_DIR}")

target_map = {}  # canonical class -> [video paths]
for vpath in all_videos:
    size = os.path.getsize(vpath)
    if size > MAX_VIDEO_BYTES or size < MIN_VIDEO_BYTES:
        continue
    stem = os.path.splitext(os.path.basename(vpath))[0]
    cls = canonical_class_name(stem, KNOWN_CLASSES)
    if not EXTRACT_ALL and cls not in KNOWN_CLASSES:
        continue
    target_map.setdefault(cls, []).append(vpath)

print(f"[*] Selected {len(target_map)} classes, "
      f"{sum(len(v) for v in target_map.values())} videos total")

meta = {"classes": {}, "videos": {}, "feature_version": 2,
        "target_frames": TARGET_FRAMES, "target_features": TARGET_FEATURES}
total_saved = 0

for cls_name, vpaths in tqdm(sorted(target_map.items()), desc="Classes"):
    safe = re.sub(r'[\\/*?:"<>|]', '_', cls_name)
    cls_dir = os.path.join(OUTPUT_DATASET_DIR, safe)
    os.makedirs(cls_dir, exist_ok=True)
    meta["classes"].setdefault(cls_name, 0)

    for vpath in tqdm(vpaths, desc=f"  {cls_name}", leave=False):
        seq = process_video(vpath, holistic)
        if seq is None or seq.shape != (TARGET_FRAMES, TARGET_FEATURES):
            print(f"  [!] SKIP (no landmarks / wrong shape): {os.path.basename(vpath)}")
            continue
        stem = re.sub(r'[\\/*?:"<>|]', '_', os.path.splitext(os.path.basename(vpath))[0])
        out_path = os.path.join(cls_dir, f"{stem}.npy")
        np.save(out_path, seq)
        meta["classes"][cls_name] += 1
        meta["videos"][os.path.basename(vpath)] = cls_name
        total_saved += 1

import json
with open(os.path.join(OUTPUT_DATASET_DIR, "dataset_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\n" + "=" * 65)
print("  EXTRACTION COMPLETE")
print("=" * 65)
print(f"  Classes : {len(meta['classes'])}")
print(f"  Sequences (real videos): {total_saved}")
print(f"  Saved to: {OUTPUT_DATASET_DIR}")
print("=" * 65)
for cls, count in sorted(meta["classes"].items()):
    status = "[OK ]" if count >= 3 else ("[LOW]" if count >= 1 else "[ZERO]")
    print(f"  {status} {cls:<20s} : {count} video sequence(s)")

print("\n[NEXT] Run colab_train_v2.py cells to train the v2 model on GPU.")


# ═══════════════ CELL 6 (optional): close resources ═══════════════════════════
# holistic.close()
