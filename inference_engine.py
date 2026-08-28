"""
inference_engine.py — Real-Time Sliding Window Inference Engine (v2.0)

Continuously processes webcam frames through the unified sign model.
No timers, no buttons, no recording windows — always listening.

How it works:
  1. Every frame → push_frame() adds 138 holistic features to a 30-frame buffer
  2. Every 3 frames (~100ms) → run inference on the buffer
  3. If confidence > threshold AND not idle AND cooldown elapsed → return result
  4. Caller (gui.py) handles TTS and sentence building
"""

import time
import numpy as np
import torch
import json
import os
import logging
from collections import deque

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Sliding window real-time inference engine for sign language recognition.

    Thread-safe: designed to be called from the webcam capture thread.
    """

    def __init__(
        self,
        model_path: str,
        labels_path: str,
        window_size: int = 30,
        stride: int = 3,
        threshold: float = 0.85,
        cooldown_sec: float = 1.5,
        repeat_cooldown_sec: float = 3.0,
    ):
        """
        Args:
            model_path         : path to TorchScript .pt model file
            labels_path        : path to labels.json
            window_size        : frames to buffer (30 @ 30fps = ~1 second window)
            stride             : run inference every N frames (3 = ~100ms @ 30fps)
            threshold          : minimum softmax confidence to trigger output (0.85 = 85%)
            cooldown_sec       : minimum seconds between ANY sign triggers
            repeat_cooldown_sec: minimum seconds before the SAME sign triggers again
        """
        # Load labels
        with open(labels_path, "r") as f:
            self.labels = json.load(f)
        self.num_classes = len(self.labels)

        # Load TorchScript model
        self.model = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()
        logger.info(f"InferenceEngine: loaded model '{model_path}' with {self.num_classes} classes")

        # Sliding window buffer
        self.buffer      = deque(maxlen=window_size)
        self.window_size = window_size
        self.stride      = stride

        # Confidence & cooldown settings
        self.threshold          = threshold
        self.cooldown_sec       = cooldown_sec
        self.repeat_cooldown    = repeat_cooldown_sec

        # State tracking
        self._frame_count       = 0
        self._last_trigger_time = 0.0
        self._last_label        = None

        # Cached probabilities for UI display (top-k bars)
        self._last_probs: list[tuple[str, float]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def push_frame(self, features: np.ndarray) -> tuple[str, float] | None:
        """
        Add one frame's 138 holistic features to the sliding buffer.
        Runs inference every `stride` frames.

        Args:
            features : numpy array of shape (138,) — holistic landmark features

        Returns:
            (label, confidence) if a sign is confirmed above threshold with cooldown
            None otherwise (most frames return None — that's normal)
        """
        self.buffer.append(features.astype(np.float32))
        self._frame_count += 1

        # Only run inference every `stride` frames to save CPU
        if self._frame_count % self.stride != 0:
            return None

        # Need a full window of 30 frames before inferring
        if len(self.buffer) < self.window_size:
            return None

        # Build (1, 30, 138) tensor and run model
        seq    = torch.from_numpy(np.array(self.buffer)).unsqueeze(0)  # (1,30,138)
        with torch.no_grad():
            logits = self.model(seq)
            probs  = torch.softmax(logits, dim=1)[0]  # (num_classes,)

        # Cache top-3 for UI confidence bars
        top3 = probs.topk(min(3, self.num_classes))
        self._last_probs = [
            (self.labels[int(i)], float(p))
            for p, i in zip(top3.values, top3.indices)
        ]

        # Get best prediction
        confidence, idx = probs.max(0)
        label      = self.labels[int(idx)]
        confidence = float(confidence)

        # ── Filter chain ──────────────────────────────────────────────────
        now = time.time()

        # 1. Skip idle state — background noise, no hand, rest pose
        if label == "idle":
            return None

        # 2. Skip low-confidence predictions
        if confidence < self.threshold:
            return None

        # 3. Global cooldown — no sign triggered less than cooldown_sec ago
        if (now - self._last_trigger_time) < self.cooldown_sec:
            return None

        # 4. Repeat cooldown — same word blocked for repeat_cooldown_sec
        if label == self._last_label and (now - self._last_trigger_time) < self.repeat_cooldown:
            return None

        # ── Trigger! ──────────────────────────────────────────────────────
        self._last_trigger_time = now
        self._last_label        = label
        logger.debug(f"Sign detected: '{label}' @ {confidence:.1%}")
        return label, confidence

    def get_top_k_probs(self, k: int = 3) -> list[tuple[str, float]]:
        """
        Return the cached top-k (label, probability) pairs.
        Called every UI refresh tick for the confidence bars.

        Returns:
            list of (label, probability) — e.g. [("hello", 0.94), ("help", 0.03)]
            Empty list if buffer not yet full.
        """
        return self._last_probs[:k]

    def is_buffer_ready(self) -> bool:
        """Returns True once the sliding window buffer has filled up (30 frames)."""
        return len(self.buffer) >= self.window_size

    def reset(self):
        """Clear the buffer — useful after app pause/resume."""
        self.buffer.clear()
        self._frame_count = 0
        self._last_probs  = []

    @property
    def buffer_fill_pct(self) -> float:
        """How full the buffer is (0.0–1.0). Used for startup progress display."""
        return len(self.buffer) / self.window_size


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    MODEL_PATH  = os.path.join("models", "unified_sign_model.pt")
    LABELS_PATH = os.path.join("models", "labels.json")

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("  Train the model first using: colab/train_model.ipynb")
        print("  Then download unified_sign_model.pt to ./models/")
        exit(1)

    engine = InferenceEngine(MODEL_PATH, LABELS_PATH)
    print(f"✓ Engine ready | {engine.num_classes} classes | window={engine.window_size}")
    print("  Pushing 35 random frames (simulating 1+ second of video)...")

    for i in range(35):
        dummy_features = np.random.randn(138).astype(np.float32)
        result = engine.push_frame(dummy_features)
        if result:
            label, conf = result
            print(f"  → Sign triggered: '{label}' ({conf:.1%})")

    top3 = engine.get_top_k_probs()
    print(f"\n  Top-3 predictions: {top3}")
    print("✓ InferenceEngine test complete")
