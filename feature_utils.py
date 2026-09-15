"""
feature_utils.py — Shared feature extraction & model-config resolution.

FEATURE VERSIONS
────────────────
v2 (150 dims, scale- & position-invariant — used by new training):
    [   0: 60]  Left hand shape  : 20 non-wrist landmarks (indices 1..20),
                                   relative to the wrist, divided by shoulder width
    [  60: 63]  Left wrist pos   : wrist relative to mid-shoulder / shoulder width
    [  63:123]  Right hand shape : same as left
    [ 123:126]  Right wrist pos  : same as left
    [ 126:150]  Pose             : 8 landmarks x 3 (11,12 shoulders | 13,14 elbows |
                                   15,16 wrists | 23,24 hips),
                                   relative to mid-shoulder / shoulder width
    -> Invariant to where the person stands in frame and how far they are from
       the camera, so the model generalizes across people and setups.

legacy (138 dims, v1 raw normalized coordinates — kept for the original model):
    [   0: 63]  Left hand raw x,y,z   (zeros if hand missing)
    [  63:126]  Right hand raw x,y,z  (zeros if hand missing)
    [ 126:138]  Pose raw: landmarks 11,12,13,14 x,y,z
"""

import json
import os
from pathlib import Path

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_DIM_V2 = 150
FEATURE_DIM_LEGACY = 138
SEQ_LEN_DEFAULT = 60

HAND_LANDMARKS = 21
HAND_SHAPE_DIM = (HAND_LANDMARKS - 1) * 3          # 60: wrist-relative handshape
POSE_LANDMARK_IDS = (11, 12, 13, 14, 15, 16, 23, 24)  # shoulders, elbows, wrists, hips
POSE_DIM = len(POSE_LANDMARK_IDS) * 3              # 24

# Fallback "shoulder width" (MediaPipe-normalized units) when pose is undetected.
DEFAULT_SHOULDER_SCALE = 0.2

V2_MODEL_NAME = "unified_sign_model_v2.pt"
V2_CONFIG_NAME = "model_config.json"
LEGACY_MODEL_NAME = "unified_sign_model.pt"
LEGACY_CLASSES_NAME = "classes.json"


# ── v2 normalized feature extraction ─────────────────────────────────────────
def extract_features_v2(results) -> np.ndarray:
    """MediaPipe Holistic results -> (150,) normalized feature vector."""
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
    scale = float(np.linalg.norm((l_sh - r_sh)[:2]))   # x/y shoulder distance
    if scale < 1e-3:
        scale = DEFAULT_SHOULDER_SCALE

    def hand_block(landmarks) -> np.ndarray:
        """(63,) = 60 wrist-relative handshape + 3 wrist position."""
        if not landmarks:
            return np.zeros(HAND_SHAPE_DIM + 3, dtype=np.float32)
        pts = np.array([[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32)
        if pts.shape[0] != HAND_LANDMARKS:
            fixed = np.zeros((HAND_LANDMARKS, 3), dtype=np.float32)
            fixed[: min(len(pts), HAND_LANDMARKS)] = pts[:HAND_LANDMARKS]
            pts = fixed
        wrist = pts[0]
        shape = ((pts[1:] - wrist) / scale).flatten()   # (60,)
        wpos = (wrist - mid_sh) / scale                 # (3,)
        return np.concatenate([shape, wpos]).astype(np.float32)

    lh = hand_block(getattr(results, "left_hand_landmarks", None))
    rh = hand_block(getattr(results, "right_hand_landmarks", None))
    pose = np.concatenate([(P(i) - mid_sh) / scale for i in POSE_LANDMARK_IDS]).astype(np.float32)

    return np.concatenate([lh, rh, pose])               # (150,)


# ── legacy (v1) raw feature extraction — must match the original pipeline ────
def extract_features_legacy(results) -> np.ndarray:
    """MediaPipe Holistic results -> (138,) raw feature vector (original behavior)."""
    if getattr(results, "left_hand_landmarks", None):
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3, dtype=np.float32)

    if getattr(results, "right_hand_landmarks", None):
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3, dtype=np.float32)

    if getattr(results, "pose_landmarks", None):
        pose_lms = results.pose_landmarks.landmark
        upper_pose = []
        for idx in (11, 12, 13, 14):
            if idx < len(pose_lms):
                lm = pose_lms[idx]
                upper_pose.extend([lm.x, lm.y, lm.z])
            else:
                upper_pose.extend([0.0, 0.0, 0.0])
        pose = np.array(upper_pose, dtype=np.float32)
    else:
        pose = np.zeros(4 * 3, dtype=np.float32)

    return np.concatenate([lh, rh, pose])               # (138,)


def get_feature_extractor(version: int):
    """Return the extraction function matching a model config version."""
    return extract_features_v2 if int(version) >= 2 else extract_features_legacy


# ── Model / config resolution ─────────────────────────────────────────────────
def infer_input_dim(state_dict) -> int:
    """Infer LSTM input dimension from a state_dict (robust for any checkpoint)."""
    for key in ("lstm.weight_ih_l0", "lstm.weight_ih_l1"):
        if key in state_dict:
            return int(state_dict[key].shape[1])
    raise KeyError("Could not infer input_dim: 'lstm.weight_ih_l0' missing from state_dict.")


def resolve_active_model(project_root=None):
    """
    Pick the newest usable model + its config.

    Preference:
      1. model_config.json (v2) next to unified_sign_model_v2.pt  -> 150-dim v2
      2. legacy unified_sign_model.pt + classes.json              -> 138-dim v1

    Returns (model_path, config_dict) where config_dict has:
      version, input_dim, seq_len, num_classes, classes, feature, model_file
    """
    root = Path(project_root) if project_root else Path(__file__).resolve().parent

    cfg_path = root / V2_CONFIG_NAME
    if cfg_path.exists():
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            model_file = cfg.get("model_file", V2_MODEL_NAME)
            model_path = root / model_file
            if model_path.exists():
                classes = cfg.get("classes")
                if not classes:
                    with open(root / LEGACY_CLASSES_NAME, "r") as f:
                        classes = json.load(f)
                return str(model_path), {
                    "version": int(cfg.get("version", 2)),
                    "input_dim": int(cfg.get("input_dim", FEATURE_DIM_V2)),
                    "seq_len": int(cfg.get("seq_len", SEQ_LEN_DEFAULT)),
                    "num_classes": int(cfg.get("num_classes", len(classes))),
                    "classes": list(classes),
                    "feature": cfg.get("feature", "v2_normalized"),
                    "model_file": model_file,
                }
        except (json.JSONDecodeError, OSError):
            pass  # fall through to legacy

    legacy_path = root / LEGACY_MODEL_NAME
    with open(root / LEGACY_CLASSES_NAME, "r") as f:
        classes = json.load(f)
    return str(legacy_path), {
        "version": 1,
        "input_dim": FEATURE_DIM_LEGACY,
        "seq_len": SEQ_LEN_DEFAULT,
        "num_classes": len(classes),
        "classes": list(classes),
        "feature": "legacy_raw",
        "model_file": LEGACY_MODEL_NAME,
    }
