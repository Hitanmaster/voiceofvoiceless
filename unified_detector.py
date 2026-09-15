import os
import json
import time
import threading
import collections
import cv2
import numpy as np
import torch
import mediapipe as mp

from model import BiLSTMAttentionSignClassifier
from feature_utils import resolve_active_model, get_feature_extractor

# ── TTS ENGINE ─────────────────────────────────────────────────────────────────
class AsyncTTS:
    def __init__(self):
        self.lock = threading.Lock()
        self.is_busy = False

    def speak(self, text):
        if not text or self.is_busy:
            return
        t = threading.Thread(target=self._speak_worker, args=(text,), daemon=True)
        t.start()

    def _speak_worker(self, text):
        with self.lock:
            self.is_busy = True
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass
            finally:
                self.is_busy = False

# ── REAL-TIME SIGN DETECTOR ────────────────────────────────────────────────────
class RealTimeSignDetector:
    def __init__(self, model_path=None, classes_path=None, seq_len=None):
        # ── Auto-detect active model (v2 config first, legacy fallback) ──
        self.model_path, self.active_config = resolve_active_model()
        self.feature_version = self.active_config["version"]
        self.feature_dim = self.active_config["input_dim"]
        self.seq_len = seq_len or self.active_config.get("seq_len", 60)
        self.classes = self.active_config["classes"]

        if model_path:  # explicit override wins
            self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model with dims inferred from the checkpoint itself
        state = torch.load(self.model_path, map_location="cpu")
        inferred_dim = None
        try:
            from feature_utils import infer_input_dim
            inferred_dim = infer_input_dim(state)
        except Exception:
            pass
        self.model = BiLSTMAttentionSignClassifier(
            input_dim=inferred_dim or self.feature_dim,
            hidden_dim=128,
            num_layers=2,
            num_classes=len(self.classes)
        ).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        print(f"[detector] model={os.path.basename(self.model_path)} "
              f"(v{self.feature_version}, input_dim={self.feature_dim}, "
              f"classes={len(self.classes)}) on {self.device}")

        # MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Feature extractor matched to the active model
        self.extract = get_feature_extractor(self.feature_version)

        # Sequence Buffer & State
        self.sequence_buffer = collections.deque(maxlen=self.seq_len)
        self.prediction_history = collections.deque(maxlen=7)
        self.sentence = []
        self.last_predicted_sign = None
        self.last_predict_time = 0
        self.tts = AsyncTTS()

    def extract_landmarks(self, results):
        return self.extract(results)

    def reset_after_commit(self):
        """Clear stale frames so a just-committed sign isn't re-detected."""
        self.sequence_buffer.clear()
        self.prediction_history.clear()

    def process_frame(self, frame):
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True

        # Draw landmarks
        if results.left_hand_landmarks:
            self.mp_draw.draw_landmarks(frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            self.mp_draw.draw_landmarks(frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)

        # Extract features (version matched to active model)
        features = self.extract(results)
        self.sequence_buffer.append(features)

        top3_info = []
        current_sign = "Buffering frames..."

        # Inference when sequence is full
        if len(self.sequence_buffer) == self.seq_len:
            seq_tensor = torch.tensor(np.array(self.sequence_buffer), dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(seq_tensor)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                top3_indices = np.argsort(probs)[::-1][:3]
                top3_info = [(self.classes[idx], float(probs[idx])) for idx in top3_indices]

                pred_idx = top3_indices[0]
                confidence = float(probs[pred_idx])
                predicted_label = self.classes[pred_idx]

                # Threshold lowered 0.65 -> 0.50 (majority voting still guards noise)
                if confidence > 0.50 and predicted_label.lower() != "idle":
                    self.prediction_history.append(predicted_label)

                    # Majority voting over recent predictions
                    counter = collections.Counter(self.prediction_history)
                    most_common, count = counter.most_common(1)[0]

                    if count >= 3 and (time.time() - self.last_predict_time > 1.8):
                        if most_common != self.last_predicted_sign:
                            self.last_predicted_sign = most_common
                            self.last_predict_time = time.time()
                            self.sentence.append(most_common)
                            if len(self.sentence) > 6:
                                self.sentence.pop(0)
                            self.tts.speak(most_common)
                            self.reset_after_commit()
                    current_sign = f"{predicted_label} ({confidence * 100:.1f}%)"
                else:
                    current_sign = f"idle ({confidence * 100:.1f}%)" if predicted_label.lower() == "idle" else f"Low Conf ({predicted_label}: {confidence*100:.1f}%)"

        # ── UI Overlay & Debug HUD ──────────────────────────────────────────
        # Top banner
        cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
        buffer_pct = int((len(self.sequence_buffer) / self.seq_len) * 100)
        cv2.putText(frame, f"Sign: {current_sign}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 128), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Buffer: {len(self.sequence_buffer)}/{self.seq_len} ({buffer_pct}%)", (w - 240, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # Top-3 predictions panel (Top Right)
        if top3_info:
            panel_y = 65
            cv2.rectangle(frame, (w - 260, panel_y - 10), (w - 10, panel_y + 80), (15, 15, 15), -1)
            cv2.putText(frame, "Top Predictions:", (w - 250, panel_y + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            for i, (name, prob) in enumerate(top3_info):
                bar_len = int(prob * 80)
                color = (0, 255, 128) if i == 0 else (200, 200, 200)
                cv2.putText(frame, f"{name[:12]:<12} {prob*100:4.1f}%", (w - 250, panel_y + 28 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                cv2.rectangle(frame, (w - 100, panel_y + 18 + i * 20), (w - 100 + bar_len, panel_y + 26 + i * 20), (0, 200, 255), -1)

        # Sentence footer banner
        cv2.rectangle(frame, (0, h - 50), (w, h), (18, 18, 18), -1)
        sentence_str = " ".join(self.sentence) if self.sentence else "Perform a sign for ~2 sec (Press 'c' to clear, 'f' to flip)"
        cv2.putText(frame, f"Sentence: {sentence_str}", (15, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return frame

    def run_live(self):
        cap = cv2.VideoCapture(0)
        mirror = True
        print("[*] Starting real-time sign detection.")
        print("[*] Controls: 'q' = Quit | 'c' = Clear sentence | 'f' = Toggle Mirroring")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            annotated = self.process_frame(frame)
            cv2.imshow("SignLanguageAI - Real-Time Sign to Speech", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.sentence.clear()
                self.prediction_history.clear()
                self.last_predicted_sign = None
            elif key == ord('f'):
                mirror = not mirror
                print(f"[*] Camera mirroring set to: {mirror}")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = RealTimeSignDetector()
    detector.run_live()
