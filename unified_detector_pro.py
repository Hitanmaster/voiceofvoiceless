"""
unified_detector_pro.py — Enhanced Real-Time Indian Sign Language (ISL) Detector
Voice of the Voiceless Project

Features:
- Professional HUD with dynamic confidence color-coding (Green/Yellow/Red/Muted).
- Multi-version model support (v1 legacy raw, v2 normalized, v3 Vāksetu 506-dim).
- Asynchronous, non-blocking TTS engine with audio deduplication.
- Visual top-3 candidate prediction cards with animated progress bars.
- Live performance diagnostics (FPS, inference latency, buffer status).
- Interactive on-screen hotkeys (Flip, Clear, Snapshot, Repeat Speech, Threshold tuning).
"""

import os
import sys
import time
import queue
import argparse
import threading
import collections
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import mediapipe as mp

from model import BiLSTMAttentionSignClassifier
from feature_utils import resolve_active_model, get_feature_extractor


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1: VISUAL THEME & PALETTE CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

class UITheme:
    # Modern dark-mode palette (BGR format for OpenCV)
    BG_DARK = (20, 20, 22)
    BG_CARD = (30, 30, 36)
    BG_CARD_BORDER = (60, 60, 72)
    BG_BAR = (45, 45, 52)

    # Status / Confidence indicators
    CONF_HIGH = (90, 225, 110)       # Crisp Emerald Green (>80%)
    CONF_MED = (55, 215, 245)        # Warm Amber / Yellow (50-80%)
    CONF_LOW = (75, 75, 235)         # Soft Coral / Red (<50%)
    CONF_IDLE = (160, 160, 165)      # Muted Slate Gray

    # Text colors
    TEXT_MAIN = (255, 255, 255)
    TEXT_MUTED = (180, 180, 188)
    TEXT_ACCENT = (245, 195, 60)     # Cyan / Gold accent

    # Visual meters & accents
    CYAN_ACCENT = (245, 185, 40)
    VIOLET_ACCENT = (230, 100, 150)
    PILL_BG = (42, 42, 50)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2: ASYNCHRONOUS TTS ENGINE (NON-BLOCKING)
# ──────────────────────────────────────────────────────────────────────────────

class AsyncTTS:
    """
    Queue-based persistent text-to-speech engine.
    Runs inside a dedicated daemon thread to prevent any camera frame stutter.
    """
    def __init__(self, rate=150, enabled=True):
        self.enabled = enabled
        self.rate = rate
        self.queue = queue.Queue(maxsize=4)
        self.last_spoken = None
        self._is_busy = False

        if self.enabled:
            self.worker = threading.Thread(target=self._run_loop, daemon=True)
            self.worker.start()

    def speak(self, text, force=False):
        """Enqueue speech item if enabled and not an immediate repeat."""
        if not self.enabled or not text:
            return
        clean_text = text.replace("_", " ").strip()
        if not force and clean_text.lower() == str(self.last_spoken).lower():
            return

        self.last_spoken = clean_text
        try:
            # Drop older pending utterances if backlog exists
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            self.queue.put_nowait(clean_text)
        except queue.Full:
            pass

    def repeat_last(self):
        """Re-speak the last committed sign."""
        if self.last_spoken:
            self.speak(self.last_spoken, force=True)

    def _run_loop(self):
        engine = None
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
        except Exception:
            # Graceful fallback if TTS runtime/driver is absent
            engine = None

        while True:
            text = self.queue.get()
            if engine and text:
                try:
                    self._is_busy = True
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    pass
                finally:
                    self._is_busy = False
            self.queue.task_done()


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3: PROFESSIONAL HUD RENDERER
# ──────────────────────────────────────────────────────────────────────────────

class HUDRenderer:
    """
    High-performance, modern HUD graphics renderer for real-time sign detection.
    Optimized with fast NumPy ROI alpha blending.
    """

    @staticmethod
    def draw_alpha_rect(img, x1, y1, x2, y2, color, alpha=0.82):
        """Fast in-place semi-transparent rectangle overlay."""
        h, w, _ = img.shape
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return
        sub = img[y1:y2, x1:x2]
        rect = np.full_like(sub, color, dtype=np.uint8)
        img[y1:y2, x1:x2] = cv2.addWeighted(rect, alpha, sub, 1.0 - alpha, 0)

    @staticmethod
    def get_confidence_color(confidence, is_idle=False):
        """Map confidence score to ergonomic color tokens."""
        if is_idle:
            return UITheme.CONF_IDLE
        if confidence >= 0.80:
            return UITheme.CONF_HIGH
        if confidence >= 0.50:
            return UITheme.CONF_MED
        return UITheme.CONF_LOW

    @classmethod
    def draw_top_bar(cls, frame, active_sign, confidence, is_idle, buffer_cur, buffer_max, fps, infer_ms):
        """Render top status banner with sign pill, buffer meter, and FPS metrics."""
        h, w, _ = frame.shape
        banner_h = 60

        # Translucent top backdrop + bottom accent line
        cls.draw_alpha_rect(frame, 0, 0, w, banner_h, UITheme.BG_DARK, alpha=0.88)
        cv2.line(frame, (0, banner_h), (w, banner_h), UITheme.BG_CARD_BORDER, 1)

        # 1. Sign Status Badge (Left)
        conf_color = cls.get_confidence_color(confidence, is_idle)
        cls.draw_alpha_rect(frame, 12, 10, 310, banner_h - 10, UITheme.PILL_BG, alpha=0.92)
        cv2.rectangle(frame, (12, 10), (310, banner_h - 10), conf_color, 1, cv2.LINE_AA)

        # Status indicator glowing dot
        dot_center = (28, 30)
        cv2.circle(frame, dot_center, 7, conf_color, -1, cv2.LINE_AA)
        cv2.circle(frame, dot_center, 9, (255, 255, 255), 1, cv2.LINE_AA)

        # Sign Label & Confidence Text
        display_label = active_sign.replace("_", " ").upper()
        if len(display_label) > 13:
            display_label = display_label[:12] + "…"

        cv2.putText(frame, display_label, (44, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, UITheme.TEXT_MAIN, 2, cv2.LINE_AA)

        conf_str = f"{confidence * 100:4.1f}%" if not is_idle else "STANDBY"
        cv2.putText(frame, conf_str, (215, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, conf_color, 1, cv2.LINE_AA)

        # 2. Buffer Progress Meter (Center)
        buf_w = 180
        buf_x = int(w / 2 - buf_w / 2)
        buf_y = 22
        buf_h = 16
        pct = min(1.0, max(0.0, buffer_cur / max(1, buffer_max)))

        # Background slot
        cls.draw_alpha_rect(frame, buf_x, buf_y, buf_x + buf_w, buf_y + buf_h, UITheme.BG_BAR, alpha=0.9)
        cv2.rectangle(frame, (buf_x, buf_y), (buf_x + buf_w, buf_y + buf_h), UITheme.BG_CARD_BORDER, 1)

        # Fill segment
        fill_w = int(buf_w * pct)
        if fill_w > 0:
            fill_color = UITheme.CYAN_ACCENT if pct < 1.0 else UITheme.CONF_HIGH
            cv2.rectangle(frame, (buf_x + 1, buf_y + 1), (buf_x + fill_w - 1, buf_y + buf_h - 1),
                          fill_color, -1)

        buf_label = f"BUFFER: {buffer_cur}/{buffer_max} ({int(pct * 100)}%)"
        cv2.putText(frame, buf_label, (buf_x + 14, buf_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, UITheme.TEXT_MAIN, 1, cv2.LINE_AA)

        # 3. Performance Badges: FPS & Inference Latency (Right)
        perf_text = f"{fps:2.0f} FPS  |  {infer_ms:3.0f}ms"
        cls.draw_alpha_rect(frame, w - 180, 14, w - 12, banner_h - 14, UITheme.PILL_BG, alpha=0.85)
        cv2.rectangle(frame, (w - 180, 14), (w - 12, banner_h - 14), UITheme.BG_CARD_BORDER, 1)
        cv2.putText(frame, perf_text, (w - 168, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, UITheme.TEXT_ACCENT, 1, cv2.LINE_AA)

    @classmethod
    def draw_top3_card(cls, frame, top3_info):
        """Floating top-3 prediction confidence card at upper-right."""
        if not top3_info:
            return

        h, w, _ = frame.shape
        card_w = 230
        card_h = 106
        x1 = w - card_w - 16
        y1 = 74
        x2 = x1 + card_w
        y2 = y1 + card_h

        # Card Container
        cls.draw_alpha_rect(frame, x1, y1, x2, y2, UITheme.BG_CARD, alpha=0.85)
        cv2.rectangle(frame, (x1, y1), (x2, y2), UITheme.BG_CARD_BORDER, 1, cv2.LINE_AA)

        # Card Title
        cv2.putText(frame, "TOP CANDIDATES", (x1 + 14, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, UITheme.TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.line(frame, (x1 + 10, y1 + 24), (x2 - 10, y1 + 24), UITheme.BG_CARD_BORDER, 1)

        # Candidates with animated progress bars
        for idx, (label, prob) in enumerate(top3_info[:3]):
            row_y = y1 + 44 + (idx * 24)
            pct = min(1.0, max(0.0, float(prob)))

            # Rank indicator
            rank_text = f"#{idx+1}"
            cv2.putText(frame, rank_text, (x1 + 12, row_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, UITheme.TEXT_MUTED, 1, cv2.LINE_AA)

            # Class Name
            clean_name = label.replace("_", " ")
            if len(clean_name) > 10:
                clean_name = clean_name[:9] + "…"
            cv2.putText(frame, f"{clean_name:<10}", (x1 + 32, row_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, UITheme.TEXT_MAIN, 1, cv2.LINE_AA)

            # Mini Progress Bar
            bar_x = x1 + 115
            bar_w = 65
            bar_h = 8
            cls.draw_alpha_rect(frame, bar_x, row_y - 8, bar_x + bar_w, row_y, UITheme.BG_BAR, alpha=0.9)

            bar_fill = int(bar_w * pct)
            color = cls.get_confidence_color(pct) if idx == 0 else UITheme.CYAN_ACCENT
            if bar_fill > 0:
                cv2.rectangle(frame, (bar_x, row_y - 8), (bar_x + bar_fill, row_y), color, -1)

            # Percentage
            cv2.putText(frame, f"{pct * 100:4.1f}%", (bar_x + bar_w + 6, row_y - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, UITheme.TEXT_MUTED, 1, cv2.LINE_AA)

    @classmethod
    def draw_footer_and_controls(cls, frame, sentence_list, active_threshold, is_mirror):
        """Render two-tier bottom bar: sentence accumulator + hotkey hints."""
        h, w, _ = frame.shape
        footer_h = 50
        control_h = 28
        total_h = footer_h + control_h

        y_footer_top = h - total_h
        y_ctrl_top = h - control_h

        # 1. Sentence Display Bar
        cls.draw_alpha_rect(frame, 0, y_footer_top, w, y_ctrl_top, UITheme.BG_DARK, alpha=0.92)
        cv2.line(frame, (0, y_footer_top), (w, y_footer_top), UITheme.BG_CARD_BORDER, 1)

        # Sentence prefix tag
        cls.draw_alpha_rect(frame, 12, y_footer_top + 10, 115, y_ctrl_top - 10, UITheme.PILL_BG, alpha=0.9)
        cv2.rectangle(frame, (12, y_footer_top + 10), (115, y_ctrl_top - 10), UITheme.BG_CARD_BORDER, 1)
        cv2.putText(frame, "SENTENCE", (22, y_footer_top + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, UITheme.TEXT_ACCENT, 1, cv2.LINE_AA)

        # Sentence tokens
        if sentence_list:
            sentence_text = "  ".join(sentence_list).replace("_", " ").upper()
            if len(sentence_text) > 52:
                sentence_text = "…" + sentence_text[-50:]
            text_color = UITheme.TEXT_MAIN
        else:
            sentence_text = "Perform a sign to build a sentence... (Press 'C' to clear)"
            text_color = UITheme.TEXT_MUTED

        cv2.putText(frame, sentence_text, (130, y_footer_top + 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, text_color, 1 if not sentence_list else 2, cv2.LINE_AA)

        # 2. Control Hints Ribbon (Bottom-most)
        cls.draw_alpha_rect(frame, 0, y_ctrl_top, w, h, (12, 12, 14), alpha=0.98)
        cv2.line(frame, (0, y_ctrl_top), (w, y_ctrl_top), UITheme.BG_CARD_BORDER, 1)

        mirror_status = "ON" if is_mirror else "OFF"
        hints = [
            ("[Q]", "Quit"),
            ("[C]", "Clear"),
            ("[F]", f"Mirror:{mirror_status}"),
            ("[S]", "Snapshot"),
            ("[Space]", "Audio"),
            ("[+/-]", f"Thresh:{active_threshold:.2f}")
        ]

        cursor_x = 18
        for key, desc in hints:
            cv2.putText(frame, key, (cursor_x, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, UITheme.TEXT_ACCENT, 1, cv2.LINE_AA)
            cursor_x += int(len(key) * 8.5) + 3

            cv2.putText(frame, desc, (cursor_x, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, UITheme.TEXT_MUTED, 1, cv2.LINE_AA)
            cursor_x += int(len(desc) * 7.5) + 16

    @classmethod
    def draw_notification_toast(cls, frame, message):
        """Display an ephemeral feedback toast in the upper-center."""
        if not message:
            return
        h, w, _ = frame.shape
        toast_w = min(w - 60, int(len(message) * 11 + 40))
        toast_h = 36
        x1 = int(w / 2 - toast_w / 2)
        y1 = 75
        x2 = x1 + toast_w
        y2 = y1 + toast_h

        cls.draw_alpha_rect(frame, x1, y1, x2, y2, (30, 80, 45), alpha=0.92)
        cv2.rectangle(frame, (x1, y1), (x2, y2), UITheme.CONF_HIGH, 1, cv2.LINE_AA)
        cv2.putText(frame, message, (x1 + 18, y1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    @classmethod
    def draw_splash_screen(cls, frame, model_name, version, classes_count, device):
        """Render modern startup splash screen card."""
        h, w, _ = frame.shape
        cls.draw_alpha_rect(frame, 0, 0, w, h, (10, 10, 12), alpha=0.88)

        card_w, card_h = 540, 310
        x1 = int(w / 2 - card_w / 2)
        y1 = int(h / 2 - card_h / 2)
        x2 = x1 + card_w
        y2 = y1 + card_h

        cls.draw_alpha_rect(frame, x1, y1, x2, y2, UITheme.BG_CARD, alpha=0.95)
        cv2.rectangle(frame, (x1, y1), (x2, y2), UITheme.CYAN_ACCENT, 2, cv2.LINE_AA)

        # Header
        cv2.putText(frame, "VOICE OF THE VOICELESS", (x1 + 35, y1 + 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, UITheme.TEXT_MAIN, 2, cv2.LINE_AA)
        cv2.putText(frame, "Real-Time Indian Sign Language AI Detector", (x1 + 35, y1 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, UITheme.TEXT_ACCENT, 1, cv2.LINE_AA)
        cv2.line(frame, (x1 + 30, y1 + 96), (x2 - 30, y1 + 96), UITheme.BG_CARD_BORDER, 1)

        # Metadata Specs
        specs = [
            ("Active Model:", f"{model_name}"),
            ("Architecture:", f"BiLSTM + Attention (v{version})"),
            ("Vocabulary:", f"{classes_count} ISL Signs & Phrases"),
            ("Compute Hardware:", f"{device}"),
        ]

        start_y = y1 + 135
        for idx, (label, val) in enumerate(specs):
            cv2.putText(frame, label, (x1 + 40, start_y + idx * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, UITheme.TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(frame, val, (x1 + 200, start_y + idx * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, UITheme.TEXT_MAIN, 1, cv2.LINE_AA)

        # Footer Instruction
        cv2.putText(frame, "Starting camera feed... Ready to sign!", (x1 + 110, y2 - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, UITheme.CONF_HIGH, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4: REAL-TIME SIGN DETECTOR ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────────

class RealTimeSignDetector:
    """
    Orchestrates MediaPipe landmark extraction, rolling temporal buffering,
    BiLSTM neural inference, and reactive HUD drawing.
    """
    def __init__(self, model_path=None, camera_id=0, enable_tts=True, mirror=True, threshold=0.50):
        self.camera_id = camera_id
        self.mirror = mirror
        self.confidence_threshold = threshold
        self.cooldown_seconds = 1.8

        # ── 1. Resolve Model & Configuration ──
        self.model_path, self.active_config = resolve_active_model()
        if model_path:
            self.model_path = model_path

        self.feature_version = self.active_config["version"]
        self.feature_dim = self.active_config["input_dim"]
        self.seq_len = self.active_config.get("seq_len", 60)
        self.classes = self.active_config["classes"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── 2. Load Neural Network ──
        self._load_classifier()

        # ── 3. MediaPipe Holistic Pipeline ──
        self._init_mediapipe()

        # ── 4. Feature Extraction Strategy ──
        self.extract = get_feature_extractor(self.feature_version)
        self.is_v3 = int(self.feature_version) >= 3
        self.buffer_len = (self.active_config.get("window_frames") or 45) if self.is_v3 else self.seq_len

        # ── 5. State Buffers & Tracking ──
        self.sequence_buffer = collections.deque(maxlen=self.buffer_len)
        self.prediction_history = collections.deque(maxlen=7)
        self.sentence = []
        self.last_predicted_sign = None
        self.last_predict_time = 0

        # Ephemeral notification message (e.g. screenshot saved)
        self.notification_msg = None
        self.notification_expiry = 0

        # Asynchronous speech engine
        self.tts = AsyncTTS(rate=150, enabled=enable_tts)
        self.hud = HUDRenderer()

        print(f"[*] Initialized RealTimeSignDetector (Pro):")
        print(f"    - Model: {os.path.basename(self.model_path)} (v{self.feature_version})")
        print(f"    - Feature Dims: {self.feature_dim} | Target Seq: {self.seq_len} | Buffer: {self.buffer_len}")
        print(f"    - Vocabulary: {len(self.classes)} classes on {self.device}")

    def _load_classifier(self):
        """Instantiate BiLSTM classifier with inferred weight dimensions."""
        state = torch.load(self.model_path, map_location="cpu")
        inferred_dim = None
        try:
            from feature_utils import infer_input_dim
            inferred_dim = infer_input_dim(state)
        except Exception:
            pass

        self.model = BiLSTMAttentionSignClassifier(
            input_dim=inferred_dim or self.feature_dim,
            hidden_dim=self.active_config.get("hidden_dim", 128),
            num_layers=2,
            num_classes=len(self.classes)
        ).to(self.device)

        self.model.load_state_dict(state)
        self.model.eval()

    def _init_mediapipe(self):
        """Initialize MediaPipe Holistic with clean custom drawing styles."""
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Custom landmark aesthetics
        self.hand_landmark_style = self.mp_draw.DrawingSpec(
            color=(255, 255, 255), thickness=1, circle_radius=2
        )
        self.hand_conn_style = self.mp_draw.DrawingSpec(
            color=(80, 220, 110), thickness=2, circle_radius=1
        )
        self.pose_conn_style = self.mp_draw.DrawingSpec(
            color=(245, 185, 50), thickness=2, circle_radius=1
        )

    def set_notification(self, text, duration=2.5):
        """Show a temporary status banner on the HUD."""
        self.notification_msg = text
        self.notification_expiry = time.time() + duration

    def reset_after_commit(self):
        """Clear temporal sequence buffer after a recognized sign commits."""
        self.sequence_buffer.clear()
        self.prediction_history.clear()

    def process_frame(self, frame, fps=30.0):
        """
        Processes a single video frame:
        1. MediaPipe holistic keypoint detection & landmark visualization
        2. Feature vector normalization
        3. Rolling temporal buffer inference
        4. HUD overlay rendering
        """
        start_t = time.perf_counter()
        h, w, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True

        # Draw custom styled landmarks
        if results.left_hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.hand_landmark_style,
                connection_drawing_spec=self.hand_conn_style
            )
        if results.right_hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.hand_landmark_style,
                connection_drawing_spec=self.hand_conn_style
            )
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.hand_landmark_style,
                connection_drawing_spec=self.pose_conn_style
            )

        # Feature extraction
        features = self.extract(results)
        self.sequence_buffer.append(features)

        top3_info = []
        current_display_sign = "STANDBY"
        active_confidence = 0.0
        is_idle_status = True

        # Run inference once buffer is primed
        if len(self.sequence_buffer) == self.buffer_len:
            if self.is_v3:
                from vaksetu.vaksetu_features import build_model_input
                seq = build_model_input(list(self.sequence_buffer), self.seq_len)
            else:
                seq = np.array(self.sequence_buffer)

            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.model(seq_tensor)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                top3_indices = np.argsort(probs)[::-1][:3]
                top3_info = [(self.classes[idx], float(probs[idx])) for idx in top3_indices]

                pred_idx = top3_indices[0]
                confidence = float(probs[pred_idx])
                predicted_label = self.classes[pred_idx]
                active_confidence = confidence

                if confidence >= self.confidence_threshold and predicted_label.lower() != "idle":
                    is_idle_status = False
                    self.prediction_history.append(predicted_label)
                    current_display_sign = predicted_label

                    # Majority voting over recent consecutive windows
                    counter = collections.Counter(self.prediction_history)
                    most_common, count = counter.most_common(1)[0]

                    # Trigger sign commit when consensus threshold reached
                    if count >= 3 and (time.time() - self.last_predict_time > self.cooldown_seconds):
                        if most_common != self.last_predicted_sign:
                            self.last_predicted_sign = most_common
                            self.last_predict_time = time.time()
                            self.sentence.append(most_common)
                            if len(self.sentence) > 8:
                                self.sentence.pop(0)

                            self.tts.speak(most_common)
                            self.set_notification(f"Recognized: {most_common.upper()}")
                            self.reset_after_commit()
                else:
                    is_idle_status = True
                    current_display_sign = "IDLE" if predicted_label.lower() == "idle" else "SEARCHING"

        infer_ms = (time.perf_counter() - start_t) * 1000.0

        # ── Compose Professional HUD Layers ──
        # 1. Top status bar
        self.hud.draw_top_bar(
            frame,
            active_sign=current_display_sign,
            confidence=active_confidence,
            is_idle=is_idle_status,
            buffer_cur=len(self.sequence_buffer),
            buffer_max=self.buffer_len,
            fps=fps,
            infer_ms=infer_ms
        )

        # 2. Top-3 Predictions floating card
        if top3_info:
            self.hud.draw_top3_card(frame, top3_info)

        # 3. Bottom sentence & hotkeys control bar
        self.hud.draw_footer_and_controls(
            frame,
            sentence_list=self.sentence,
            active_threshold=self.confidence_threshold,
            is_mirror=self.mirror
        )

        # 4. Optional notification toast
        if self.notification_msg and time.time() < self.notification_expiry:
            self.hud.draw_notification_toast(frame, self.notification_msg)
        else:
            self.notification_msg = None

        return frame

    def run_live(self):
        """Execute interactive webcam detection loop with full hotkey support."""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"[!] Error: Could not open camera with index {self.camera_id}")
            return

        window_name = "Voice of the Voiceless — Real-Time ISL to Speech AI (Pro)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Show splash intro screen for 1.6 seconds
        splash_start = time.time()
        ret, frame = cap.read()
        if ret:
            if self.mirror:
                frame = cv2.flip(frame, 1)
            self.hud.draw_splash_screen(
                frame,
                model_name=os.path.basename(self.model_path),
                version=self.feature_version,
                classes_count=len(self.classes),
                device=str(self.device).upper()
            )
            cv2.imshow(window_name, frame)
            cv2.waitKey(1600)

        # Performance rolling metrics
        fps_buffer = collections.deque(maxlen=30)
        last_t = time.time()

        print("[*] Camera feed active. Control Keys:")
        print("    [Q] / [ESC] : Quit")
        print("    [C]         : Clear sentence buffer")
        print("    [F]         : Toggle horizontal camera mirror")
        print("    [S]         : Save annotated snapshot")
        print("    [Space]     : Repeat speech audio")
        print("    [+] / [-]   : Adjust confidence threshold")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps_buffer.append(1.0 / dt)
            current_fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 30.0

            if self.mirror:
                frame = cv2.flip(frame, 1)

            annotated = self.process_frame(frame, fps=current_fps)
            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF

            # Key Actions
            if key in (ord('q'), 27):  # 'q' or ESC
                break
            elif key == ord('c'):
                self.sentence.clear()
                self.prediction_history.clear()
                self.last_predicted_sign = None
                self.set_notification("Sentence cleared")
            elif key == ord('f'):
                self.mirror = not self.mirror
                self.set_notification(f"Camera Mirror: {'ON' if self.mirror else 'OFF'}")
            elif key == ord('s'):
                # Save snapshot to screenshots folder
                screenshots_dir = Path("screenshots")
                screenshots_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                shot_path = screenshots_dir / f"isl_detect_{timestamp}.png"
                cv2.imwrite(str(shot_path), annotated)
                self.set_notification(f"Snapshot saved: {shot_path.name}")
                print(f"[*] Saved screenshot: {shot_path}")
            elif key == 32:  # Spacebar
                self.tts.repeat_last()
                self.set_notification("Replaying audio...")
            elif key in (ord('+'), ord('=')):
                self.confidence_threshold = min(0.95, round(self.confidence_threshold + 0.05, 2))
                self.set_notification(f"Threshold: {self.confidence_threshold:.2f}")
            elif key in (ord('-'), ord('_')):
                self.confidence_threshold = max(0.20, round(self.confidence_threshold - 0.05, 2))
                self.set_notification(f"Threshold: {self.confidence_threshold:.2f}")

        cap.release()
        cv2.destroyAllWindows()
        print("[*] Session terminated gracefully.")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5: COMMAND LINE INTERFACE
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Voice of the Voiceless — Real-Time Indian Sign Language AI Detector (Pro)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to custom model checkpoint (.pt). Auto-detects if omitted."
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Webcam device index (default: 0)."
    )
    parser.add_argument(
        "--no-tts", action="store_true",
        help="Disable automatic text-to-speech engine."
    )
    parser.add_argument(
        "--no-mirror", action="store_true",
        help="Disable camera horizontal mirror flip."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.50,
        help="Minimum confidence threshold for sign confirmation (default: 0.50)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detector = RealTimeSignDetector(
        model_path=args.model,
        camera_id=args.camera,
        enable_tts=not args.no_tts,
        mirror=not args.no_mirror,
        threshold=args.threshold
    )
    detector.run_live()

