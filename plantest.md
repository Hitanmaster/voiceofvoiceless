# Sign Language AI System — Enhancement Plan

## 1. Current System Analysis

### What Exists Today
Your project currently has two separate, isolated pipelines controlled by a CLI menu:

| Component | Static Signs (A–Z) | Motion Signs (Hello, Thank You…) |
| :--- | :--- | :--- |
| **Algorithm** | Random Forest (scikit-learn) | Dynamic Time Warping (DTW) |
| **Input** | Single frame (63 features) | Sequence of frames (60 × 63 matrix) |
| **Trigger** | User selects Option 3 → 3-second timer → stops | User selects Option 5 → 5-second timer → stops |
| **Data** | `hand_landmarks.csv` | `motion_signs/<name>/*.npy` |
| **Detection** | MediaPipe Hands (1 hand) | MediaPipe Hands (1–2 hands) |
| **Output** | gTTS `output.mp3` | gTTS `motion_output.mp3` |

### Identified Limitations
* **Manual mode switching:** User must choose "static" or "motion" from a menu before each detection session. There is no unified mode.
* **Timer-based capture:** Static mode runs for exactly 3 seconds, motion mode for 5 seconds. There is no continuous detection.
* **No real-time auto-detection:** The system does not automatically recognize when a sign starts or ends in a continuous video stream.
* **No confidence-based triggering:** Results are only shown after the timer expires, not when a confident prediction is made.
* **No cooldown/deduplication:** If the system were continuous, it would repeatedly detect the same sign.
* **Single-hand constraint for static:** Static pipeline only uses 1 hand (63 features). Two-handed signs cannot be detected.
* **No web interface:** Only works through local CLI + OpenCV window. Cannot be demoed remotely or in a browser.
* **No deep learning model:** The planned BiLSTM + Attention architecture from `FINAL_YEAR_PROJECT_PLAN.md` has not been implemented yet.
* **No idle/background class:** The system has no way to distinguish "not signing" from "signing", leading to false positives.
* **No data augmentation:** Training data is raw, with no rotation, noise, or speed variations to improve generalization.

---

## 2. Proposed Enhancements

### Enhancement A: Unified Continuous Real-Time Detection Engine
**Goal:** Replace the two separate pipelines with a single continuous detection loop that recognizes both static and motion signs simultaneously, without any menu interaction.

**How it works:**
1. A sliding window buffer maintains the last 60 frames (~2 seconds at 30 FPS).
2. Every ~300ms (every 9th frame), the system runs inference:
   * **For static signs:** The latest single frame is checked against the classifier.
   * **For motion signs:** The full 60-frame window is checked against the sequence model.
3. The system picks the best prediction across both pipelines.
4. When confidence exceeds a threshold (default 80%) for 3 consecutive evaluations, the sign is auto-triggered and spoken via TTS.
5. A 2-second cooldown prevents the same sign from being spoken repeatedly.
6. An "Idle/No Sign" class prevents false positives when the user is not signing.

**Key components:**
```text
Camera Feed → MediaPipe → Frame Buffer (deque, maxlen=60)
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
            Single-Frame Path       Sequence Path
            (Random Forest)         (DTW / Deep Learning)
                    │                    │
                    └─────────┬──────────┘
                              ▼
                    Confidence Comparator
                              │
                              ▼
                    Auto-Trigger + Cooldown + TTS
```

### Enhancement B: Deep Learning Model (BiLSTM + Attention)
**Goal:** Replace the Random Forest + DTW combo with a single unified neural network that handles both static and motion signs in one forward pass.

**Architecture:**
| Layer | Details |
| :--- | :--- |
| **Input** | (batch, 60 frames, 138 features) — dual hand + pose from MediaPipe Holistic |
| **Feature Projection** | Linear(138 → 128) → BatchNorm1d → ReLU → Dropout(0.3) |
| **BiLSTM** | 2 layers, hidden=128, bidirectional, dropout=0.3 |
| **Self-Attention** | Learned attention weights over 60 time steps → weighted sum → context vector |
| **Classifier** | Linear(256 → 64) → ReLU → Dropout(0.3) → Linear(64 → num_classes) |
| **Output** | Softmax over all sign classes + "Idle" class |

**Why this works for both static and motion signs:**
* A **static sign** (e.g., letter "A") appears as 60 nearly identical frames → the model learns the fixed spatial shape.
* A **motion sign** (e.g., "Hello") appears as 60 frames with a movement trajectory → the model learns the temporal pattern.
* The attention mechanism automatically weights the most informative frames.

**Training pipeline:**
* **Dataset:** 30+ sequences per sign class, each 60 frames of 138 features.
* **Augmentation:** Gaussian noise (σ=0.01), random rotation (±15°), random scale (0.85×–1.15×), temporal speed warping, hand dropout.
* **Optimizer:** AdamW with learning rate scheduling.
* **Loss:** CrossEntropyLoss with class weights (to handle imbalance with the Idle class).
* **Evaluation:** Confusion matrix, precision/recall/F1 per class, loss/accuracy curves.
* **Export:** PyTorch `.pt` file, optional ONNX export for faster inference.
* **Fallback:** If no deep learning model is trained yet, the system falls back to Random Forest + DTW using the existing algorithms.

### Enhancement C: Gradio Web Interface
**Goal:** Provide a browser-based GUI that works locally or can be deployed to Google Colab, replacing the CLI-only workflow.

**Interface layout:**
| Tab | Features |
| :--- | :--- |
| **Live Detection** | Webcam feed with overlaid predictions, confidence bars, detection history log, auto-spoken audio |
| **Record New Sign** | Text input for sign name, record button, visual countdown, frame counter, save to dataset |
| **Train Model** | Button to start training, progress bar, epoch counter, accuracy display, confusion matrix image |
| **Dataset Manager** | Table showing all recorded signs with frame counts, bar chart of class distribution, delete buttons |
| **Settings** | Sliders for confidence threshold (50–99%), cooldown duration (0.5–5s), detection mode toggle (static/motion/unified), webcam device selector |

**Technical implementation:**
* **Framework:** Gradio (lightweight, Colab-compatible, no frontend code needed)
* **Webcam:** Gradio Webcam component for input
* **Audio:** Gradio Audio component for TTS playback
* Detection runs in a background thread, updating a shared state object
* Results displayed in real-time via Gradio's reactive updates

---

## 3. New Feature: MediaPipe Holistic (Dual Hand + Pose)

**Current:** MediaPipe Hands → 1 hand → 21 landmarks → 63 features per frame.
**Upgrade:** MediaPipe Holistic → 2 hands + upper body pose → 138 features per frame.

| Body Part | Landmarks | Features |
| :--- | :--- | :--- |
| **Left Hand** | 21 keypoints × 3D | 63 |
| **Right Hand** | 21 keypoints × 3D | 63 |
| **Upper Body Pose** (shoulders + elbows) | 4 keypoints × 3D | 12 |
| **Total** | | **138** |

This enables detection of two-handed signs (e.g., letters T, U, etc. in ASL) and provides body context for more accurate classification.
**Fallback:** If Holistic fails to detect both hands, the system pads missing hand features with zeros and continues with available data.

---

## 4. Implementation Phases

### Phase 1: Data Collection Infrastructure
**Files:** `dataset_collector.py`
* New script using MediaPipe Holistic for 138-feature extraction
* Records 30+ sequences per sign class (each 60 frames)
* Collects "Idle" class data (resting hands, random movements)
* Visual countdown, progress bar, frame counter in OpenCV window
* Saves sequences as `.npy` files organized by class name
* Batch collection mode for recording multiple signs in one session

### Phase 2: Deep Learning Model
**Files:** `model.py`, `train_model.py`
* `model.py`: PyTorch model definition (BiLSTM + Attention)
* `train_model.py`: Training pipeline with:
  * Data loading from `.npy` directory structure
  * Augmentation pipeline (noise, rotation, scale, speed warp, hand dropout)
  * Training loop with AdamW + CosineAnnealingLR
  * Validation after each epoch
  * Confusion matrix + classification report generation
  * Model checkpoint saving (best validation accuracy)
* Designed to run on Google Colab GPU or local GPU

### Phase 3: Continuous Detection Engine
**Files:** `unified_detector.py`
* Sliding window buffer (deque, maxlen=60)
* Inference loop at ~300ms intervals
* Dual-path detection (single-frame + sequence)
* Confidence comparison and selection
* Auto-trigger with configurable threshold
* Cooldown timer with visual indicator
* Fallback to Random Forest + DTW if no DL model available
* Real-time OpenCV overlay with:
  * Detected sign name + confidence percentage
  * Confidence progress bar
  * Detection mode indicator (Static/Motion/Unified)
  * Cooldown countdown
  * "No hand detected" warning

### Phase 4: Web Interface
**Files:** `app.py`
* Gradio-based web UI with 5 tabs
* Live webcam detection with real-time predictions
* On-the-fly sign recording from the browser
* Model training trigger with progress display
* Dataset management with charts
* Settings panel for threshold, cooldown, mode selection
* Works in Google Colab (no local display needed)

### Phase 5: Integration & Main Menu Update
**Files:** `main.py` (modify)
* Add Option 10: Continuous Real-Time Detection (Deep Learning)
* Add Option 11: Launch Web Interface
* Keep Options 1–9 as legacy fallback
* Import new modules

---

## 5. File Structure After Enhancement

```text
SignLanguageAI/
├── main.py                      # Main CLI (updated with options 10-11)
├── model.py                     # [NEW] BiLSTM + Attention PyTorch model
├── unified_detector.py          # [NEW] Continuous real-time detection engine
├── dataset_collector.py         # [NEW] Holistic data collection script
├── train_model.py               # [NEW] Training pipeline with augmentation
├── app.py                       # [NEW] Gradio web interface
├── motion_signs.py              # Existing DTW engine (kept as fallback)
├── hand_landmarks.csv           # Existing static dataset
├── sign_language_model.pkl      # Existing Random Forest model
├── unified_sign_model.pt        # [NEW] Trained PyTorch model
├── motion_signs/                # Existing motion sign templates
├── dataset/                     # [NEW] Holistic training data
│   ├── hello/
│   ├── goodbye/
│   ├── thank_you/
│   ├── idle/
│   └── ...
├── colab/                       # Existing Colab scripts
│   ├── collect_data.py
│   ├── train.py
│   └── realtime_inference.py
├── sign.ipynb                   # Existing notebook
├── FINAL_YEAR_PROJECT_PLAN.md   # Existing plan
├── PRODUCT_DOCUMENTATION.md     # Existing docs
└── README.md                    # Existing readme
```

---

## 6. New Dependencies

```bash
torch                    # PyTorch for deep learning model
torchvision              # PyTorch vision utilities
gradio                   # Web interface framework
matplotlib               # Confusion matrix and training curve plots
seaborn                  # Enhanced data visualization
```
*All other dependencies (opencv-python, mediapipe, scikit-learn, pandas, numpy, gTTS) are already in the project.*

---

## 7. Configuration Parameters

| Parameter | Default | Range | Description |
| :--- | :--- | :--- | :--- |
| `CONFIDENCE_THRESHOLD` | `0.80` | `0.50–0.99` | Minimum confidence to auto-trigger a prediction |
| `CONSECUTIVE_REQUIRED` | `3` | `1–10` | Number of consecutive matching predictions before trigger |
| `COOLDOWN_SECONDS` | `2.0` | `0.5–5.0` | Silence period after each triggered prediction |
| `INFERENCE_INTERVAL_MS` | `300` | `100–1000` | Milliseconds between model inference calls |
| `WINDOW_SIZE` | `60` | `30–90` | Number of frames in the sliding window buffer |
| `IDLE_CLASS` | `"idle"` | — | Background class label for "not signing" |
| `MODEL_FILE` | `"unified_sign_model.pt"`| — | Path to trained PyTorch model |
| `FALLBACK_ENABLED` | `True` | — | Use RF+DTW if no DL model is available |

---

## 8. Expected Outcomes

| Metric | Current System | After Enhancement |
| :--- | :--- | :--- |
| **Detection trigger** | Manual menu selection | Automatic, continuous |
| **Sign types supported** | Static OR motion (separate) | Both simultaneously |
| **User interaction needed** | Select option, wait for timer | Just show your hand |
| **Hands supported** | 1 hand (static), 2 hands (motion) | 2 hands + pose (138 features) |
| **Interface** | CLI + OpenCV window | CLI + OpenCV + Gradio Web UI |
| **Model accuracy target** | ~85% (Random Forest) | >92% (BiLSTM + Attention) |
| **Idle/false-positive handling** | None | Dedicated "Idle" class |
| **Data augmentation** | None | 5 augmentation techniques |
| **Remote demo capability** | None | Gradio web UI in Colab |

---

## 9. Risk Mitigation

| Risk | Mitigation |
| :--- | :--- |
| **Insufficient training data** | Data augmentation pipeline + batch collection mode + minimum 30 sequences per class |
| **DL model too slow on CPU** | Fallback to RF+DTW; inference throttled to every 300ms; ONNX export for optimization |
| **MediaPipe Holistic fails** | Graceful fallback to single-hand MediaPipe Hands (63 features) |
| **Low accuracy with few signs**| Confidence threshold is configurable; system shows "Low confidence" warnings |
| **Camera not available** | Gradio web UI supports file upload for pre-recorded video testing |
| **GPU unavailable for training**| Training script designed to run on Google Colab free GPU tier |

---

## 10. Deliverables for Final Year Project

* **Working system** with continuous real-time detection of both static and motion signs
* **Web-based demo** via Gradio that can be shown in presentations and deployed to Colab
* **Trained deep learning model** with documented accuracy, precision, recall, and F1 scores
* **Confusion matrix and training curves** for thesis inclusion
* **Updated technical documentation** reflecting the new architecture
* **Comparative analysis in thesis:** Legacy (RF+DTW) vs. Unified (BiLSTM+Attention)
* **System architecture diagrams** showing the new unified pipeline
