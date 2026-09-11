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
    def __init__(self, model_path="unified_sign_model.pt", classes_path="classes.json", seq_len=60):
        self.seq_len = seq_len
        self.feature_dim = 138
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load classes
        with open(classes_path, "r") as f:
            self.classes = json.load(f)

        # Load Model
        self.model = BiLSTMAttentionSignClassifier(
            input_dim=self.feature_dim,
            hidden_dim=128,
            num_layers=2,
            num_classes=len(self.classes)
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Sequence Buffer & State
        self.sequence_buffer = collections.deque(maxlen=self.seq_len)
        self.prediction_history = collections.deque(maxlen=7)
        self.sentence = []
        self.last_predicted_sign = None
        self.last_predict_time = 0
        self.tts = AsyncTTS()

    def extract_landmarks(self, results):
        # Left Hand (63)
        if results.left_hand_landmarks:
            lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        else:
            lh = np.zeros(21 * 3, dtype=np.float32)

        # Right Hand (63)
        if results.right_hand_landmarks:
            rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        else:
            rh = np.zeros(21 * 3, dtype=np.float32)

        # Upper Pose (12)
        if results.pose_landmarks:
            pose_lms = results.pose_landmarks.landmark
            upper_pose = []
            for idx in [11, 12, 13, 14]:
                if idx < len(pose_lms):
                    lm = pose_lms[idx]
                    upper_pose.extend([lm.x, lm.y, lm.z])
                else:
                    upper_pose.extend([0.0, 0.0, 0.0])
            pose = np.array(upper_pose, dtype=np.float32)
        else:
            pose = np.zeros(4 * 3, dtype=np.float32)

        return np.concatenate([lh, rh, pose])

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

        # Extract features
        features = self.extract_landmarks(results)
        self.sequence_buffer.append(features)

        current_sign = "Waiting for motion..."
        confidence = 0.0

        # Inference when sequence is full
        if len(self.sequence_buffer) == self.seq_len:
            seq_tensor = torch.tensor(np.array(self.sequence_buffer), dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(seq_tensor)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                pred_idx = np.argmax(probs)
                confidence = float(probs[pred_idx])
                predicted_label = self.classes[pred_idx]

                if confidence > 0.75 and predicted_label.lower() != "idle":
                    self.prediction_history.append(predicted_label)
                    
                    # Majority voting over recent predictions
                    counter = collections.Counter(self.prediction_history)
                    most_common, count = counter.most_common(1)[0]

                    if count >= 4 and (time.time() - self.last_predict_time > 1.8):
                        if most_common != self.last_predicted_sign:
                            self.last_predicted_sign = most_common
                            self.last_predict_time = time.time()
                            self.sentence.append(most_common)
                            if len(self.sentence) > 6:
                                self.sentence.pop(0)
                            self.tts.speak(most_common)
                    current_sign = f"{predicted_label} ({confidence * 100:.1f}%)"
                else:
                    current_sign = f"idle ({confidence * 100:.1f}%)" if predicted_label.lower() == "idle" else "..."

        # UI Overlay
        # Header banner
        cv2.rectangle(frame, (0, 0), (w, 60), (24, 24, 24), -1)
        cv2.putText(frame, f"Prediction: {current_sign}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 128), 2, cv2.LINE_AA)

        # Sentence footer banner
        cv2.rectangle(frame, (0, h - 50), (w, h), (18, 18, 18), -1)
        sentence_str = " ".join(self.sentence) if self.sentence else "Signs will form a sentence here..."
        cv2.putText(frame, f"Sentence: {sentence_str}", (20, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        return frame

    def run_live(self):
        cap = cv2.VideoCapture(0)
        print("[*] Starting real-time sign detection. Press 'q' to quit, 'c' to clear sentence.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            annotated = self.process_frame(frame)
            cv2.imshow("SignLanguageAI - Real-Time Sign to Speech", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.sentence.clear()
                self.last_predicted_sign = None

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = RealTimeSignDetector()
    detector.run_live()
