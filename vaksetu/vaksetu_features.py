"""
vaksetu_features.py — Live feature extraction that matches the
Vāksetu ISL Feature Dataset v1.0 (Zenodo, DOI 10.5281/zenodo.23004815).

The dataset stores every sample as a (20, 506) float32 array:

    per frame (253 "base" features):
      [  0: 63]  left hand,  wrist-centred, divided by max wrist distance
      [ 63:126]  right hand, wrist-centred, divided by max wrist distance
      [126:189]  left hand  relative to the face  ((p - nose) / eye distance)
      [189:252]  right hand relative to the face
      [252]      hand-to-face proximity (min norm of the face-relative hands,
                 1.0 when no face or no hand)
    + 253 velocity features: frame[t] - frame[t-1]  (frame 0 velocity = 0)

The maths below is ported from the dataset author's own extractor
(JosephJonathanFernandes/sign_to_text_module, src/shared/feature_extractor.py
and src/preprocessing/preprocess.py) so our live features line up with the
training data.

Holistic vs. HandLandmarker handedness
──────────────────────────────────────
Vāksetu was recorded on a mirrored (selfie) webcam frame and used the
MediaPipe Tasks HandLandmarker, whose "Left"/"Right" labels are correct on a
mirrored image. Our detector uses Holistic, which labels hands assuming an
UN-mirrored image. So on a mirrored frame Holistic's left/right are swapped
relative to Vāksetu. SWAP_HOLISTIC_HANDS handles that. If live predictions
look poor, flip it and compare (see README_VAKSETU.md, "Hand check").
"""

from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21
NUM_COORDS = 3
LANDMARK_DIM = NUM_LANDMARKS * NUM_COORDS        # 63
BASE_DIM = 4 * LANDMARK_DIM + 1                  # 253
FULL_DIM = BASE_DIM * 2                          # 506
SEQ_LEN = 20                                     # frames per sample in Vāksetu

FACE_NOSE = 1
FACE_LEFT_EYE = 33
FACE_RIGHT_EYE = 263

# Detector mirrors the webcam by default (cv2.flip(frame, 1)), same as Vāksetu.
# On a mirrored frame Holistic's hand labels are the opposite of HandLandmarker's.
SWAP_HOLISTIC_HANDS = True


# ── per-hand helpers (ported 1:1) ─────────────────────────────────────────────
def normalize_hand(hand_raw: np.ndarray) -> np.ndarray:
    """(63,) raw -> wrist-centred, scaled by max distance from wrist."""
    if hand_raw is None or hand_raw.size != LANDMARK_DIM or not np.any(hand_raw):
        return np.zeros(LANDMARK_DIM, dtype=np.float32)
    h = hand_raw.reshape(NUM_LANDMARKS, NUM_COORDS).astype(np.float32).copy()
    h -= h[0].copy()
    m = float(np.linalg.norm(h, axis=1).max())
    if m > 1e-6:
        h /= m
    return h.flatten().astype(np.float32)


def face_anchor(face_pts: np.ndarray | None):
    """face_pts: (N, 3) with N > 263 -> (nose centre, eye distance) or (None, 1)."""
    if face_pts is None or len(face_pts) <= FACE_RIGHT_EYE:
        return None, 1.0
    center = face_pts[FACE_NOSE]
    scale = float(np.linalg.norm(face_pts[FACE_LEFT_EYE] - face_pts[FACE_RIGHT_EYE]))
    if scale < 1e-6:
        scale = 1.0
    return center, scale


def face_relative(hand_raw: np.ndarray, center, scale) -> np.ndarray:
    if center is None or hand_raw is None or not np.any(hand_raw):
        return np.zeros(LANDMARK_DIM, dtype=np.float32)
    rel = (hand_raw.reshape(NUM_LANDMARKS, NUM_COORDS) - center) / scale
    return rel.flatten().astype(np.float32)


def build_frame(left_raw, right_raw, face_pts) -> np.ndarray:
    """Raw (63,) hands + (N,3) face -> (253,) Vāksetu base frame."""
    z = np.zeros(LANDMARK_DIM, dtype=np.float32)
    left_raw = z if left_raw is None else np.asarray(left_raw, dtype=np.float32)
    right_raw = z if right_raw is None else np.asarray(right_raw, dtype=np.float32)

    left_n, right_n = normalize_hand(left_raw), normalize_hand(right_raw)
    center, scale = face_anchor(face_pts)
    left_r = face_relative(left_raw, center, scale)
    right_r = face_relative(right_raw, center, scale)

    prox = 1.0
    lp, rp = bool(np.any(left_raw)), bool(np.any(right_raw))
    if center is not None and (lp or rp):
        dl = float(np.linalg.norm(left_r)) if lp else np.inf
        dr = float(np.linalg.norm(right_r)) if rp else np.inf
        prox = min(dl, dr)
        if not np.isfinite(prox):
            prox = 1.0

    return np.concatenate([left_n, right_n, left_r, right_r,
                           np.array([prox], dtype=np.float32)]).astype(np.float32)


# ── MediaPipe Holistic -> base frame ─────────────────────────────────────────
def _lm_to_array(landmarks) -> np.ndarray | None:
    if not landmarks:
        return None
    return np.array([[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32)


def extract_frame_from_holistic(results, swap_hands: bool = SWAP_HOLISTIC_HANDS) -> np.ndarray:
    """Holistic results (from a mirrored frame) -> (253,) base features."""
    lh = _lm_to_array(getattr(results, "left_hand_landmarks", None))
    rh = _lm_to_array(getattr(results, "right_hand_landmarks", None))
    if swap_hands:
        lh, rh = rh, lh
    face = _lm_to_array(getattr(results, "face_landmarks", None))
    left_raw = lh.flatten() if lh is not None and lh.shape[0] == NUM_LANDMARKS else None
    right_raw = rh.flatten() if rh is not None and rh.shape[0] == NUM_LANDMARKS else None
    return build_frame(left_raw, right_raw, face)


# ── sequence helpers ─────────────────────────────────────────────────────────
def add_velocity(base_seq: np.ndarray) -> np.ndarray:
    """(T, 253) -> (T, 506): [position | frame-to-frame delta], first delta = 0."""
    base_seq = np.asarray(base_seq, dtype=np.float32)
    vel = np.zeros_like(base_seq)
    vel[1:] = base_seq[1:] - base_seq[:-1]
    return np.concatenate([base_seq, vel], axis=1).astype(np.float32)


def sample_frames(frames, num_frames: int = SEQ_LEN) -> np.ndarray:
    """Evenly pick num_frames from a list/array of base frames (Vāksetu-style)."""
    arr = np.asarray(frames, dtype=np.float32)
    if len(arr) == 0:
        return np.zeros((num_frames, BASE_DIM), dtype=np.float32)
    idx = np.linspace(0, len(arr) - 1, num_frames, dtype=int)
    return arr[idx]


def build_model_input(frame_buffer, num_frames: int = SEQ_LEN) -> np.ndarray:
    """Rolling buffer of (253,) frames -> (num_frames, 506) model input."""
    return add_velocity(sample_frames(frame_buffer, num_frames))


def hands_visible(base_frame: np.ndarray) -> bool:
    return bool(np.any(base_frame[: 2 * LANDMARK_DIM]))


def video_to_sequence(video_path: str, num_frames: int = SEQ_LEN,
                      mirror: bool = True, swap_hands: bool = SWAP_HOLISTIC_HANDS):
    """Whole video -> (num_frames, 506). Used to add your own signs to training."""
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.flip(f, 1) if mirror else f)
    cap.release()
    if not frames:
        return None
    picked = [frames[i] for i in np.linspace(0, len(frames) - 1, num_frames, dtype=int)]

    base = []
    with mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=1,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5) as hol:
        for f in picked:
            res = hol.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            base.append(extract_frame_from_holistic(res, swap_hands=swap_hands))
    return add_velocity(np.stack(base))
