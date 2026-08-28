# 🎫 TICKET-03 — Backend / Integration Engineer
## Feature: Real-Time Sliding Window Inference Engine + main.py Integration

**Assigned to:** Person 3  
**Files owned:** `unified_detector.py`, `main.py` (modifications only)  
**Depends on:** TICKET-01 (`unified_sign_model.pt` + `class_labels.json` must exist)  
**Due:** End of Week 3 (Day 21)

---

## 📋 Implementation Steps

### Part A — Sliding Window Frame Buffer

- [ ] Create file `unified_detector.py` in the project root
- [ ] Import: `cv2`, `mediapipe as mp`, `numpy as np`, `torch`, `json`, `time`, `threading`, `collections.deque`, `pyttsx3`
- [ ] Define constants at the top of the file:
  ```python
  WINDOW_SIZE = 60           # frames in sliding buffer
  CONFIDENCE_THRESHOLD = 0.80
  CONSECUTIVE_REQUIRED = 3   # evaluations in a row before trigger
  COOLDOWN_SECONDS = 2.0
  INFERENCE_INTERVAL_MS = 300
  MODEL_PATH = "unified_sign_model.pt"
  LABELS_PATH = "class_labels.json"
  IDLE_CLASS = "idle"
  ```
- [ ] Initialize frame buffer: `frame_buffer = deque(maxlen=WINDOW_SIZE)`
- [ ] Load `class_labels.json` → list of class names: `labels = json.load(open(LABELS_PATH))`

---

### Part B — Model Loading with Fallback

- [ ] Write `load_model()` function:
  - [ ] Try to load `unified_sign_model.pt` using `torch.load()` + `model.load_state_dict()`
  - [ ] Set model to eval mode: `model.eval()`
  - [ ] If file not found, print warning: `"[WARNING] DL model not found. Falling back to Random Forest."`
  - [ ] In fallback: load `sign_language_model.pkl` using `pickle.load()`
  - [ ] Return `(model, 'deep_learning')` or `(rf_model, 'random_forest')`
    - > **Note:** The fallback to Random Forest only handles single-frame static signs. Dynamic gesture detection will be disabled in fallback mode. Make this clear in the printed warning.

---

### Part C — Inference Logic

- [ ] Write `run_inference(buffer, model, model_type)` function:
  - [ ] If `model_type == 'deep_learning'`:
    - [ ] Convert `deque` → numpy array shape `(60, 138)` → tensor shape `(1, 60, 138)`
    - [ ] Run `model(tensor)` → logits shape `(1, num_classes)`
    - [ ] Apply `torch.softmax(logits, dim=1)`
    - [ ] Return `(predicted_class_name, confidence_float)`
  - [ ] If `model_type == 'random_forest'`:
    - [ ] Use only the last frame: `buffer[-1]` shape `(138,)` → `(1, 138)`, but slice to first 63 features for RF compatibility
    - [ ] Return `(rf_model.predict([features])[0], rf_model.predict_proba([features]).max())`
- [ ] Write `should_trigger(prediction, confidence, consecutive_counter, last_sign, cooldown_start)` function:
  - [ ] Return `False` if `prediction == IDLE_CLASS`
  - [ ] Return `False` if `confidence < CONFIDENCE_THRESHOLD`
  - [ ] Return `False` if currently in cooldown: `time.time() - cooldown_start < COOLDOWN_SECONDS`
  - [ ] Return `False` if `consecutive_counter < CONSECUTIVE_REQUIRED`
  - [ ] Return `True` only when all conditions pass

---

### Part D — TTS Engine

- [ ] Write `speak(text)` function:
  - [ ] Try `pyttsx3` first: `engine.say(text); engine.runAndWait()`
  - [ ] If `pyttsx3` fails, fallback to `gTTS`: save to `output.mp3` and play with `playsound`
  - [ ] Run `speak()` in a **separate thread** so it doesn't block the detection loop
    - > **Note:** Never call `pyttsx3` from the main OpenCV thread directly. It can freeze the video feed. Use `threading.Thread(target=speak, args=(text,)).start()`.

---

### Part E — OpenCV Real-Time Loop

- [ ] Write `run_detector()` main function:
  - [ ] Open webcam: `cap = cv2.VideoCapture(0)`
  - [ ] Initialize MediaPipe Holistic
  - [ ] Initialize variables: `consecutive_count = 0`, `last_triggered = ""`, `cooldown_start = 0`, `last_eval_time = 0`
  - [ ] Main `while cap.isOpened()` loop:
    - [ ] Read frame, convert BGR→RGB, process with `holistic.process()`
    - [ ] Extract 138-feature vector using same `extract_features()` logic as TICKET-02
      - > **Note:** Copy or import `extract_features()` from `dataset_collector.py` — both must use identical feature extraction logic. Any mismatch = wrong predictions.
    - [ ] Append feature vector to `frame_buffer`
    - [ ] If `time.time() - last_eval_time > INFERENCE_INTERVAL_MS / 1000`:
      - [ ] If `len(frame_buffer) == WINDOW_SIZE`: run `run_inference()`
      - [ ] Update `consecutive_count`: increment if same prediction, reset to 0 if different
      - [ ] If `should_trigger()` returns True: call `speak(prediction)` in thread, reset `consecutive_count = 0`, set `cooldown_start = time.time()`
      - [ ] Update `last_eval_time = time.time()`
    - [ ] Draw overlays on frame (see Part F)
    - [ ] `cv2.imshow("Sign Language AI", frame)`
    - [ ] Break on `q` key

---

### Part F — OpenCV Frame Overlays

Draw the following on every displayed frame:
- [ ] **Predicted sign + confidence**: top-left, white text, e.g. `"hello  (94%)"` — green if confidence > 80%, yellow if 50–80%, red if < 50%
- [ ] **Confidence progress bar**: horizontal bar below the text, fill proportional to confidence %
- [ ] **Cooldown indicator**: `"Next in: 1.4s"` — shown only when in cooldown period
- [ ] **Buffer fill indicator**: `"Buffer: 60/60"` — so user knows when buffer is ready
- [ ] **"No hand detected" warning**: red text in centre of frame when no landmarks found for > 2 seconds

---

### Part G — main.py Integration

- [ ] Open `main.py`
- [ ] Find the existing menu options section
- [ ] Add **Option 10**: `"10. Unified Real-Time Detection (Deep Learning)"` → calls `unified_detector.run_detector()`
- [ ] Import `unified_detector` at the top of `main.py`
- [ ] Do NOT modify or remove Options 1–9 (legacy fallback must remain intact)

---

## ✅ Test Checklist

- [ ] `unified_detector.py` imports without errors
- [ ] Model loads successfully: `"[INFO] Loaded unified_sign_model.pt (33 classes)"` printed on start
- [ ] Fallback activates correctly when `.pt` file is deleted: `"[WARNING] DL model not found. Falling back to Random Forest."`
- [ ] Webcam opens and video feed shows at >= 20 FPS (check with `cv2.CAP_PROP_FPS`)
- [ ] Frame buffer fills to 60 and prediction text appears on screen
- [ ] Sign "hello" is detected and spoken within 2 seconds of performing the gesture
- [ ] `idle` class is correctly suppressed — resting hand does NOT trigger speech
- [ ] Cooldown timer prevents same sign from being spoken twice within 2 seconds
- [ ] `consecutive_count` resets when a different sign appears mid-gesture
- [ ] No video freeze or lag caused by TTS (runs in separate thread)
- [ ] Option 10 in `main.py` launches the detector without error

---

## 🔴 Important Notes

> **Critical Dependency:** `extract_features()` in `unified_detector.py` MUST produce the exact same output as the one in `dataset_collector.py`. If the feature ordering is different, the model will produce garbage predictions. Strongly recommend importing from a shared `feature_extractor.py` module.

> **Thread Safety:** The `deque` is written by the main loop and read by the inference step in the same thread — this is fine. But `speak()` must always be in a separate thread to prevent blocking.

> **`class_labels.json` Required:** The file must be in the same directory as `unified_sign_model.pt`. If it's missing, print an error and exit gracefully — do not crash silently.
