# Sign Language AI System — Master Project Documentation (`content.md`)

> **Documentation Policy & Organization Rule:**  
> **All new updates, comparisons, logs, research notes, and enhancements must always be prepended at the TOP of this file** under the `# 🆕 Latest Entries & Updates` section to maintain reverse-chronological order and immediate visibility.

---

# 🆕 Latest Entries & Updates

## Entry: [2026-08-28] 1-Month 4-Person Team Project Plan (Hybrid Strategy)

**Full Plan File:** `ONE_MONTH_PROJECT_PLAN.md` in the project root.

### Team Roles at a Glance

| Member | Role | Core Deliverable |
| :--- | :--- | :--- |
| **Person 1** | ML Engineer / Team Lead | `model.py`, `train_model.py`, Colab training, confusion matrix |
| **Person 2** | Data Engineer | `dataset_collector.py`, all .npy recording, Drive upload |
| **Person 3** | Backend / Integration | `unified_detector.py`, `main.py`, TTS, RF+DTW fallback |
| **Person 4** | UI / Documentation Lead | `app.py` (Gradio), thesis writing, slides, demo video |

### Monthly Sprint Summary

| Week | Focus | Key Output |
| :--- | :--- | :--- |
| **Week 1** (Day 1–7) | Setup + Data Collection | `dataset/` on Google Drive, 30+ seqs per class |
| **Week 2** (Day 8–14) | Model Training on Colab GPU | `unified_sign_model.pt`, accuracy > 85%, confusion matrix |
| **Week 3** (Day 15–21) | Real-Time Engine + Web UI | Live 30 FPS detection, Gradio app running at localhost |
| **Week 4** (Day 22–30) | Testing + Documentation + Demo | Thesis materials, presentation deck, GitHub release |

### Minimum Requirements Before Starting
- Google Account with Google Drive (5 GB free)
- All 4 members cloned the shared GitHub repository
- Python 3.9/3.10 installed with: `torch`, `mediapipe`, `gradio`, `opencv-python`, `pyttsx3`

## Entry: [2026-08-28] Deployment Strategy — Full Colab + ngrok/Gradio vs. Train on Colab + Run Locally

### 🏆 VERDICT: Option B — Train on Colab, Run Real-Time Inference Locally is the BETTER choice.

---

### The Two Options Explained

**Option A — Full Colab Pipeline (ngrok / Gradio public URL):**
```text
Your Webcam → Browser → ngrok tunnel → Colab GPU → MediaPipe → Model → ngrok → Browser → TTS
```

**Option B — Hybrid: Train on Colab, Infer Locally:**
```text
[TRAINING — Colab GPU, done once]
  dataset/ → BiLSTM+Attention → unified_sign_model.pt  (Download ~10 MB)

[INFERENCE — Local Machine, every demo]
  Webcam → MediaPipe (local) → Frame Buffer → unified_sign_model.pt → TTS → Speaker
```

---

### Why Option A (Full Colab + ngrok/Gradio) FAILS for Real-Time Detection

The core problem is **network round-trip latency destroying the 30 FPS sliding window requirement:**

| Problem | Impact |
| :--- | :--- |
| **Each webcam frame travels: Browser → ngrok → Colab → ngrok → Browser** | 200–800ms per frame round-trip |
| **Effective FPS drops from 30 to 2–5 FPS** | Sliding window buffer fills with stale/dropped frames |
| **MediaPipe must run on Colab server-side** | Adds another 50–150ms server compute delay |
| **ngrok Free Tier: 40 connections/min, 1 GB/month bandwidth** | Gets throttled during live demo |
| **Colab disconnects after ~90 min idle / ~12 hrs total** | Breaks demo mid-presentation |
| **No offline capability** | Requires active internet at the venue |

**Scenario where Option A is acceptable:**
- Uploading a **pre-recorded video file** for classification (not real-time).
- Running **model training** on Colab GPU (no webcam needed, no latency issue).
- Showing a **static Gradio dashboard** for dataset management or training progress.

---

### Why Option B (Train Colab, Run Locally) WINS

| Factor | Option A (Full Colab + ngrok) | Option B (Train Colab, Run Local) |
| :--- | :--- | :--- |
| **Real-time FPS** | 2–5 FPS (network-limited) | 20–30 FPS (local, zero network overhead) |
| **Webcam latency per frame** | 200–800ms | <10ms (direct USB / internal camera) |
| **MediaPipe landmark extraction** | Server-side on Colab (50–150ms lag) | Local (real-time, near-zero overhead) |
| **Colab session crash kills demo** | YES — demo stops completely | NO — local inference runs independently |
| **Requires internet during demo** | YES — always | NO — fully offline after model download |
| **ngrok dependency** | Required every demo session | Not needed after training |
| **Model file size to download** | N/A | ~5–15 MB .pt file (trivial) |
| **Confidence threshold accuracy** | Degraded (stale frames) | Full accuracy (fresh 30 FPS frames) |
| **Thesis demo reliability** | HIGH RISK of live failure | LOW RISK, predictable performance |
| **TTS audio playback** | Delivered via browser (delayed) | Direct local speaker (instant) |

---

### Recommended Hybrid Workflow (Best of Both Worlds)

```text
STEP 1 — Data Collection (Local Machine):
  Run dataset_collector.py locally with webcam
  → Saves 60-frame .npy sequences to dataset/ folder

STEP 2 — Upload & Train (Google Colab GPU — done offline, no time pressure):
  Upload dataset/ to Google Drive
  Mount Drive in Colab: drive.mount('/content/drive')
  Run train_model.py on free T4 GPU (~30–60 min for 30+ classes)
  Save checkpoint: unified_sign_model.pt to Drive

STEP 3 — Download Model (One-Time):
  Download unified_sign_model.pt (~5–15 MB) to local SignLanguageAI/ folder

STEP 4 — Live Demo (Local Machine, 100% Offline):
  Run unified_detector.py
  Local Webcam → MediaPipe Holistic → 60-frame buffer → unified_sign_model.pt → TTS → Speaker
  Full 30 FPS, zero internet dependency, works anywhere
```

### Colab Role Summary (What Colab Is Good For)

| Task | Run Where | Why |
| :--- | :--- | :--- |
| **Dataset collection** | Local | Webcam is local, zero latency |
| **Model training** | Google Colab GPU | Free T4 GPU is 10-20x faster than local CPU |
| **Model evaluation / confusion matrix** | Google Colab | Use static dataset, no webcam streaming needed |
| **Real-time sign detection** | Local Machine | 30 FPS requirement, zero latency |
| **Final presentation demo** | Local Machine | Reliable, offline, no dependency on internet/Colab uptime |

## Entry: [2026-08-28] Feasibility Analysis & Plan Comparison (`plantest.md` vs. `FINAL_YEAR_PROJECT_PLAN.md`)

### Executive Recommendation
* **Most Feasible Roadmap**: **`plantest.md`** is selected as the primary engineering roadmap for the project build.
* **Academic Reference**: **`FINAL_YEAR_PROJECT_PLAN.md`** serves as the theoretical framework, mathematical formulation, and thesis writing reference.

### Direct Comparison Matrix

| Aspect | `plantest.md` (Recommended Implementation) | `FINAL_YEAR_PROJECT_PLAN.md` (Academic Reference) |
| :--- | :--- | :--- |
| **Approach & Strategy** | **Pragmatic & Resilient**: Incremental upgrade with built-in fallbacks. | **Theoretical / All-in-one**: Direct full rewrite to Deep Learning on GPU. |
| **Sliding Window Size** | **60 Frames (~2.0s)** evaluated every 300ms (matches human gesture tempo). | **30 Frames (~1.0s)** evaluated every 100ms. |
| **Fallback Mechanism** | ✅ **Yes**: Dual-path engine falls back to Random Forest + DTW if no DL model is trained. | ❌ **None**: System fails if DL model is incomplete. |
| **User Interface** | **5-Tab Interactive Gradio Web App** (Live, Record, Train, Manage, Settings). | High-level conceptual WebRTC/Gradio mention. |
| **Landmark Handling** | **138 Holistic Landmarks** with zero-padding & missing landmark resilience. | 138 Holistic Landmarks assuming full frame visibility. |
| **Modular File Layout** | Explicitly defines 5 modular Python scripts (`model.py`, `app.py`, `train_model.py`, etc.). | Outlines high-level concepts and Colab notebook snippets. |
| **Configuration Tuning** | Complete configurable parameter table (Thresholds, Cooldowns, Strides). | Hardcoded values in text. |
| **Risk Mitigation** | Concrete mitigations for CPU limits, data scarcity, and camera loss. | Theoretical discussion. |

### Why `plantest.md` is More Feasible for the Project
1. **Guaranteed Working Demo**: The fallback architecture ensures that if deep learning training takes longer or data collection is in progress, the demo remains 100% operational using the existing Random Forest + DTW pipelines.
2. **Natural Timing**: Gestures in real life (e.g., "Thank You", "Hello") rarely complete within 1.0 second; 2.0 seconds (60 frames) provides the proper temporal window.
3. **Turnkey Web Dashboard**: The 5-tab Gradio UI covers data collection, training, live inference, and dataset management without requiring separate external tools.

---

# 🚀 Engineering Blueprint: The Feasible Enhancement Path (from `plantest.md`)

## 1. Current System Baseline & Analysis

### Current Architecture Overview
The current system operates via isolated CLI menu options:

| Component | Static Signs (A–Z) | Motion Signs (Hello, Thank You, etc.) |
| :--- | :--- | :--- |
| **Algorithm** | Random Forest (`scikit-learn`) | Dynamic Time Warping (`fastdtw` / Euclidean) |
| **Input Feature Shape** | Single frame (21 landmarks * 3D = 63 features) | Frame sequence (60 * 63 matrix) |
| **Trigger Mechanism** | Option 3 -> 3-second timer -> Stops | Option 5 -> 5-second timer -> Stops |
| **Data Storage** | `hand_landmarks.csv` | `motion_signs/<sign_name>/*.npy` |
| **Tracking Pipeline** | MediaPipe Hands (1 hand) | MediaPipe Hands (1–2 hands) |
| **Speech Output** | gTTS (`output.mp3`) | gTTS (`motion_output.mp3`) |

### Identified Limitations
* **Manual Mode Switching**: User must choose static vs. motion before signing.
* **Timer Bottleneck**: Relies on fixed countdown timers rather than natural continuous recognition.
* **No Deduplication/Cooldown**: Continuous recognition without debouncing creates duplicate speech triggers.
* **Single Hand Constraint**: Static signs cannot track 2-handed letters or gestures.
* **No Background / Idle Class**: Leads to false-positive detections when hands rest.

---

## 2. Proposed System Enhancements

### Enhancement A: Unified Continuous Real-Time Detection Engine
* **Sliding Buffer**: A rolling `deque(maxlen=60)` maintains the last 2 seconds of video at 30 FPS.
* **Periodic Evaluation**: Evaluates inference every 300ms (~9 frames) to conserve CPU/GPU cycles.
* **Dual-Path Fallback**: Simultaneously evaluates single-frame features and sequence windows, picking the highest confidence prediction.
* **Debouncing & Auto-Trigger**: Requires prediction confidence >= 80% for 3 consecutive intervals before triggering speech.
* **Cooldown Buffer**: 2.0-second cooldown suppresses repeated speech triggers for the same sign.
* **Idle Rejection**: Explicit `"idle"` class rejects background movement and resting positions.

```text
Camera Stream (30 FPS) ──► MediaPipe Holistic ──► Frame Buffer (deque maxlen=60)
                                                        │
                              ┌─────────────────────────┴─────────────────────────┐
                              ▼                                                   ▼
                    Single-Frame Path (RF)                              Sequence Path (DL / DTW)
                              │                                                   │
                              └─────────────────────────┬─────────────────────────┘
                                                        ▼
                                             Confidence Comparator
                                                        │
                                                        ▼
                                          Auto-Trigger + Cooldown + TTS
```

---

### Enhancement B: Deep Neural Network (BiLSTM + Self-Attention)

```text
Input: (Batch, 60 Frames, 138 Features)
  │
  ▼
Linear(138 → 128) + BatchNorm1d + ReLU + Dropout(0.3)
  │
  ▼
Bidirectional LSTM (2 Layers, Hidden Dim = 128, Dropout = 0.3)
  │
  ▼
Multi-Head Self-Attention Layer (Computes dynamic temporal frame weights)
  │
  ▼
Linear(256 → 64) + ReLU + Dropout(0.3)
  │
  ▼
Linear(64 → Num Classes) + Softmax
```

* **Handling Static Signs**: Stationary hand shapes yield 60 near-identical frames; the network learns fixed spatial geometry.
* **Handling Dynamic Signs**: Moving gestures produce temporal variance; the BiLSTM captures directional trajectories.
* **Data Augmentation**:
  * Gaussian coordinate noise (σ = 0.01)
  * Random 3D skeletal rotation (± 15°)
  * Landmark scaling (0.85x – 1.15x)
  * Temporal speed warping (resampling sequences)
  * Random hand dropout (single-hand robustness)

---

### Enhancement C: 5-Tab Gradio Web Dashboard

| Tab | Key Functionality |
| :--- | :--- |
| **1. Live Detection** | Live webcam streaming, bounding landmark overlays, confidence bars, spoken audio playback, detection history log. |
| **2. Record New Sign** | Sign name entry, automatic visual 3-2-1 countdown, frame capture counter, automatic `.npy` saving. |
| **3. Train Model** | One-click background training trigger, real-time epoch counter, loss/accuracy curves, confusion matrix output. |
| **4. Dataset Manager** | Class distribution charts, sequence counters per class, sample inspection, sign deletion tool. |
| **5. Settings Panel** | Sliders for confidence threshold (50–99%), cooldown timer (0.5–5.0s), inference stride, and camera index. |

---

## 3. MediaPipe Holistic Landmark Specification

| Landmark Group | Keypoints Count | Coordinate Dims | Total Features |
| :--- | :--- | :--- | :--- |
| **Left Hand** | 21 keypoints | (x, y, z) | 63 features |
| **Right Hand** | 21 keypoints | (x, y, z) | 63 features |
| **Upper Pose** (Shoulders + Elbows) | 4 keypoints | (x, y, z) | 12 features |
| **Total Input Vector per Frame** | **46 keypoints** | **3D** | **138 features** |

*Missing Hand Handling:* When one hand is off-screen, its corresponding 63 coordinates are zero-padded to maintain matrix uniformity.

---

## 4. Planned Modular Code Structure

```text
SignLanguageAI/
├── content.md                   # Master unified documentation (this file)
├── main.py                      # Main entrypoint & CLI menu (options 1-11)
├── model.py                     # PyTorch BiLSTM + Attention architecture
├── train_model.py               # Model training script with augmentations
├── unified_detector.py          # Real-time sliding window detection engine
├── dataset_collector.py         # 138-feature MediaPipe Holistic data recorder
├── app.py                       # Gradio 5-Tab Web Interface
├── motion_signs.py              # DTW template engine (kept as fallback)
├── sign_language_model.pkl      # Random Forest model (kept as fallback)
├── unified_sign_model.pt        # Trained deep learning model checkpoint
├── dataset/                     # Holistic dataset sequences (.npy files)
│   ├── hello/
│   ├── thank_you/
│   ├── idle/
│   └── ...
├── motion_signs/                # Legacy DTW sequence templates
└── hand_landmarks.csv           # Legacy static landmark dataset
```

---

## 5. Configuration Parameters

| Parameter | Default | Range | Purpose |
| :--- | :--- | :--- | :--- |
| `CONFIDENCE_THRESHOLD` | `0.80` | `0.50 – 0.99` | Minimum probability required to trigger sign output |
| `CONSECUTIVE_REQUIRED` | `3` | `1 – 10` | Consecutive evaluations matching the sign |
| `COOLDOWN_SECONDS` | `2.0` | `0.5 – 5.0` | Inactive period after speech output to avoid repetition |
| `INFERENCE_INTERVAL_MS` | `300` | `100 – 1000` | Inference loop cycle interval (ms) |
| `WINDOW_SIZE` | `60` | `30 – 90` | Rolling sequence length in frames |
| `IDLE_CLASS` | `"idle"` | — | Background / rest class name |
| `MODEL_FILE` | `"unified_sign_model.pt"` | — | Primary PyTorch neural network checkpoint |
| `FALLBACK_ENABLED` | `True` | `True / False` | Enable RF + DTW if deep learning model is not loaded |

---

# 📚 Academic & Deep Learning Foundations (from `FINAL_YEAR_PROJECT_PLAN.md`)

## 1. Mathematical Formulation & Sequence Modeling

In sign language recognition, both static poses and dynamic actions can be expressed as a continuous multivariate time-series matrix:

$$X = [x_1, x_2, \dots, x_T] \in \mathbb{R}^{T \times D}$$

Where:
* $T = 60$ (time steps / frames)
* $D = 138$ (spatial landmark features per frame)

### BiLSTM Forward and Backward Hidden States
$$\vec{h}_t = \text{LSTM}_{\text{fwd}}(x_t, \vec{h}_{t-1})$$
$$\overleftarrow{h}_t = \text{LSTM}_{\text{bwd}}(x_t, \overleftarrow{h}_{t+1})$$
$$h_t = [\vec{h}_t \,\|\,\overleftarrow{h}_t] \in \mathbb{R}^{2 \cdot H}$$

### Multi-Head Self-Attention Weighting
$$e_t = \tanh(W_a h_t + b_a)$$
$$\alpha_t = \frac{\exp(e_t^T v_a)}{\sum_{k=1}^T \exp(e_t^T v_a)}$$
$$c = \sum_{t=1}^T \alpha_t h_t$$

Where $c$ represents the final weighted context vector passed to the classification head.

---

## 2. Advanced Alternative: Spatial-Temporal Graph Convolutional Networks (ST-GCN)
* Hand and upper-body joints form a spatial graph $G = (V, E)$ based on human skeletal connectivity.
* Temporal edges connect identical joints across sequential frames.
* ST-GCN extracts spatial joint correlations and temporal dynamics simultaneously, providing state-of-the-art academic reference material for thesis comparison.

---

## 3. Thesis & Final Year Deliverables

1. **Working System**: Continuous, real-time dual-hand + pose sign-to-speech converter.
2. **Ablation Studies & Metrics**:
   - Accuracy, Precision, Recall, and F1-score across static vs. dynamic sign classes.
   - Confusion Matrix visualization.
   - Inference latency (FPS on CPU vs. GPU).
3. **Comparative Thesis Section**:
   - Machine Learning Baseline (Random Forest + DTW) vs. Unified Deep Learning (BiLSTM + Self-Attention).
4. **Interactive Cloud Demo**: Browser-accessible Gradio UI deployable via Google Colab.
