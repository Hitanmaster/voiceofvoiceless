# ==============================================================================
#  ISL v2 Landmark Extractor — Google Colab (100% AUTOMATED SINGLE CELL)
#  ✅ Paste this ENTIRE script into ONE Colab cell and hit Run.
#  ✅ Handles Python 3.13 automatically — no manual runtime change needed.
#  ✅ Mounts Drive and extracts (60, 150) normalized landmarks end-to-end.
#
#  Output: /content/drive/MyDrive/SignLanguageAI/dataset_v2/<Class>/*.npy
# ==============================================================================

import sys, os, subprocess

print("=" * 65)
print(f"  ISL v2 Extractor — Host Python {sys.version.split()[0]}")
print("=" * 65)

# ── STEP 1: Mount Google Drive ─────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

# ── STEP 2: Get a working Python + MediaPipe setup ─────────────────────────────
#
# Strategy:
#   Python 3.8-3.12 → install mediapipe directly into current Python
#   Python 3.13+    → use `uv` to download Python 3.11 standalone binary,
#                     install packages into /tmp/mp_deps, run worker with
#                     PYTHONPATH pointing there. No venv, no apt-get needed.

MP_DEPS = [
    "numpy<2.0.0",
    "protobuf>=3.20.0,<4.26.0",
    "mediapipe==0.10.14",
    "opencv-python-headless",
    "tqdm",
]

def _run(*cmd, **kw):
    return subprocess.run(list(cmd), check=True, **kw)

def _run_capture(*cmd):
    r = subprocess.run(list(cmd), capture_output=True, text=True, check=True)
    return r.stdout.strip()

if sys.version_info >= (3, 13):
    print("[*] Python 3.13 — MediaPipe has no wheels. Provisioning Python 3.11 via uv...")

    # 1. Install uv (ultra-fast Python package/version manager)
    _run(sys.executable, "-m", "pip", "install", "-q", "uv")
    print("[OK] uv installed")

    # 2. Download the standalone Python 3.11 binary (cached after first run)
    _run("uv", "python", "install", "3.11")
    print("[OK] Python 3.11 downloaded")

    # 3. Locate the binary path
    PY311 = _run_capture("uv", "python", "find", "3.11")
    print(f"[OK] Python 3.11 binary: {PY311}")

    # 4. Install packages directly into /tmp/mp_deps (no venv needed)
    DEPS_DIR = "/tmp/mp_deps"
    os.makedirs(DEPS_DIR, exist_ok=True)
    _run("uv", "pip", "install",
         "--python", PY311,
         "--target", DEPS_DIR,
         *MP_DEPS)
    print(f"[OK] MediaPipe dependencies installed to {DEPS_DIR}")

    WORKER_PYTHON = PY311
    WORKER_ENV = {**os.environ, "PYTHONPATH": DEPS_DIR}

else:
    print(f"[*] Python {sys.version.split()[0]} — installing MediaPipe natively...")

    # Downgrade NumPy if needed (Colab sometimes ships NumPy 2.x)
    import numpy as _np
    if int(_np.__version__.split(".")[0]) >= 2:
        print(f"[*] NumPy {_np.__version__} too new — downgrading to <2.0.0 ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy<2.0.0"], check=True)
        print("[!] NumPy downgraded — RESTARTING runtime to load it...")
        os.kill(os.getpid(), 9)   # Colab auto-restarts and re-runs the cell

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *MP_DEPS], check=True)
    print("[OK] MediaPipe dependencies installed")

    WORKER_PYTHON = sys.executable
    WORKER_ENV    = None   # inherit current env

# ── STEP 3: Write the extraction worker script ─────────────────────────────────
WORKER_PATH = "/tmp/isl_worker.py"

with open(WORKER_PATH, "w") as _f:
    _f.write(r'''
import os, sys, re, glob, json
import cv2
import numpy as np
from tqdm.auto import tqdm

# ── Robust MediaPipe import ────────────────────────────────────────────────────
try:
    import mediapipe.python.solutions.holistic as _mp_holistic
except Exception:
    import mediapipe as _mp
    _mp_holistic = _mp.solutions.holistic

# ── CONFIG — edit these if your Drive paths differ ────────────────────────────
INPUT_VIDEOS_DIR   = "/content/drive/MyDrive/ISL_Videos"
OUTPUT_DATASET_DIR = "/content/drive/MyDrive/SignLanguageAI/dataset_v2"

TARGET_FRAMES   = 60
TARGET_FEATURES = 150

MAX_VIDEO_BYTES = 3.0 * 1024 * 1024   # skip > 3 MB (story clips)
MIN_VIDEO_BYTES = 50  * 1024          # skip < 50 KB (broken files)

TOP_35_CLASSES = [
    "Thank You", "Help",     "Sorry",    "Welcome",  "Good-1",
    "Water",     "Food",     "Eat",      "Drink",    "Father",
    "Mother-1",  "Brother",  "Sister",   "Family",   "Friend",
    "School",    "Doctor",   "Hospital", "Teacher",  "Book",
    "Money",     "House",    "Home",     "Work",     "Time",
    "Day",       "Sun",      "Moon",     "Car",      "Bus",
    "Baby",      "Boy",      "Girl",     "Dog",      "Cat-1",
]

VARIANT_ALIASING = True
EXTRACT_ALL      = False
KNOWN_CLASSES    = set(TOP_35_CLASSES)

# ── Helpers ────────────────────────────────────────────────────────────────────
def base_sign_name(name):
    return re.sub(r'[_\-]\d+$', '', name).strip()

def canonical_class_name(name, known):
    if not VARIANT_ALIASING:
        return name
    base = base_sign_name(name)
    for k in known:
        if k.lower() == base.lower():
            return k
    for k in known:
        if base_sign_name(k).lower() == base.lower():
            return k
    return name

POSE_IDS = (11, 12, 13, 14, 15, 16, 23, 24)
DEFAULT_SHOULDER_SCALE = 0.2

def extract_features_v2(results):
    if getattr(results, "pose_landmarks", None):
        plms = results.pose_landmarks.landmark
        def P(i):
            p = plms[i] if i < len(plms) else None
            return np.array([p.x, p.y, p.z], dtype=np.float32) if p else np.zeros(3, dtype=np.float32)
    else:
        def P(i): return np.zeros(3, dtype=np.float32)

    l_sh, r_sh = P(11), P(12)
    mid_sh = (l_sh + r_sh) * 0.5
    scale  = float(np.linalg.norm((l_sh - r_sh)[:2]))
    if scale < 1e-3:
        scale = DEFAULT_SHOULDER_SCALE

    def hand_block(lm):
        if not lm:
            return np.zeros(63, dtype=np.float32)
        pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
        if pts.shape[0] != 21:
            fixed = np.zeros((21, 3), dtype=np.float32)
            fixed[:min(len(pts), 21)] = pts[:21]
            pts = fixed
        wrist = pts[0]
        shape = ((pts[1:] - wrist) / scale).flatten()
        wpos  = (wrist - mid_sh) / scale
        return np.concatenate([shape, wpos]).astype(np.float32)

    lh   = hand_block(getattr(results, "left_hand_landmarks",  None))
    rh   = hand_block(getattr(results, "right_hand_landmarks", None))
    pose = np.concatenate([(P(i) - mid_sh) / scale for i in POSE_IDS]).astype(np.float32)
    return np.concatenate([lh, rh, pose])   # (150,)

def process_video(video_path, holistic):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    raw = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

# ── Main ───────────────────────────────────────────────────────────────────────
print(f"[*] Worker Python: {sys.version.split()[0]}")
print("[*] Initializing MediaPipe Holistic...")
holistic = _mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
)
print("[OK] MediaPipe Holistic ready")
print(f"[*] Input  : {INPUT_VIDEOS_DIR}")
print(f"[*] Output : {OUTPUT_DATASET_DIR}")

os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)
all_videos = glob.glob(os.path.join(INPUT_VIDEOS_DIR, "**", "*.mp4"), recursive=True)
print(f"[*] Found {len(all_videos)} total .mp4 files")

target_map = {}
for vpath in all_videos:
    size = os.path.getsize(vpath)
    if size > MAX_VIDEO_BYTES or size < MIN_VIDEO_BYTES:
        continue
    stem = os.path.splitext(os.path.basename(vpath))[0]
    cls  = canonical_class_name(stem, KNOWN_CLASSES)
    if not EXTRACT_ALL and cls not in KNOWN_CLASSES:
        continue
    target_map.setdefault(cls, []).append(vpath)

print(f"[*] {len(target_map)} classes, "
      f"{sum(len(v) for v in target_map.values())} videos after size filter")

meta = {
    "classes": {}, "videos": {},
    "feature_version": 2,
    "target_frames": TARGET_FRAMES,
    "target_features": TARGET_FEATURES,
}
total_saved = 0

for cls_name, vpaths in tqdm(sorted(target_map.items()), desc="Classes"):
    safe    = re.sub(r'[\\/*?"<>|]', '_', cls_name)
    cls_dir = os.path.join(OUTPUT_DATASET_DIR, safe)
    os.makedirs(cls_dir, exist_ok=True)
    meta["classes"].setdefault(cls_name, 0)

    for vpath in tqdm(vpaths, desc=f"  {cls_name}", leave=False):
        seq = process_video(vpath, holistic)
        if seq is None or seq.shape != (TARGET_FRAMES, TARGET_FEATURES):
            print(f"  [!] SKIP: {os.path.basename(vpath)}")
            continue
        stem     = re.sub(r'[\\/*?"<>|]', '_', os.path.splitext(os.path.basename(vpath))[0])
        out_path = os.path.join(cls_dir, f"{stem}.npy")
        np.save(out_path, seq)
        meta["classes"][cls_name] += 1
        meta["videos"][os.path.basename(vpath)] = cls_name
        total_saved += 1

holistic.close()

with open(os.path.join(OUTPUT_DATASET_DIR, "dataset_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\n" + "=" * 65)
print("  EXTRACTION COMPLETE")
print("=" * 65)
print(f"  Classes   : {len(meta['classes'])}")
print(f"  Sequences : {total_saved}")
print(f"  Saved to  : {OUTPUT_DATASET_DIR}")
print("=" * 65)
for cls, count in sorted(meta["classes"].items()):
    status = "[OK ]" if count >= 3 else ("[LOW]" if count >= 1 else "[ZERO]")
    print(f"  {status} {cls:<22s} : {count} sequence(s)")

print("\n[NEXT] Run colab_train_v2.py cells on GPU to train the model.")
''')

# ── STEP 4: Run the extraction worker ─────────────────────────────────────────
print("\n[*] Starting extraction worker...")
subprocess.run(
    [WORKER_PYTHON, WORKER_PATH],
    env=WORKER_ENV,
    check=True,
)
