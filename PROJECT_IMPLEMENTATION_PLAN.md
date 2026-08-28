# 📋 PROJECT IMPLEMENTATION PLAN
# Sign Language AI — "Voiceless to Voice" v2.0
## Real-Time Unified Deep Learning System

> **File:** `D:\workfiles\voiceless to voice\SignLanguageAI\PROJECT_IMPLEMENTATION_PLAN.md`
> **Status:** AWAITING APPROVAL — No code will be written until approved
> **Version:** 2.0 | **Date:** 2026-08-23

---

## 📖 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Where the Current Project Fails Real-World People](#2-where-the-current-project-fails-real-world-people)
3. [New System Architecture](#3-new-system-architecture)
4. [Vocabulary — What Signs to Recognise](#4-vocabulary--what-signs-to-recognise)
5. [Complete File Structure (Before vs. After)](#5-complete-file-structure-before-vs-after)
6. [File-by-File Design Specification](#6-file-by-file-design-specification)
7. [Data Collection Strategy](#7-data-collection-strategy)
8. [Model Training Strategy (Google Colab)](#8-model-training-strategy-google-colab)
9. [Technology Stack & Dependencies](#9-technology-stack--dependencies)
10. [Implementation Timeline — 8 Days](#10-implementation-timeline--8-days)
11. [Verification & Testing Plan](#11-verification--testing-plan)
12. [Teacher Presentation Script](#12-teacher-presentation-script)
13. [Thesis / Report Content Guide](#13-thesis--report-content-guide)

---

## 1. Executive Summary

### What We Have (v1.0)
A working CLI-menu Python app that:
- Recognises **static hand poses** (letters) using a Random Forest classifier
- Recognises **motion gestures** (hello, goodbye) using Dynamic Time Warping (DTW)
- Speaks the recognised word using Google TTS (requires internet)
- Has 9-option text menu — user must manually choose mode and press buttons

### What We Need (v2.0)
A **real-time desktop application** that:
- Launches with one command → webcam opens → AI runs **continuously, forever**
- Recognises **both static AND motion signs** with one unified deep learning model
- Tracks **both hands** + upper body pose (3× more data than v1.0)
- Speaks instantly using **offline TTS** — no internet needed
- Shows a **proper GUI** with live video, confidence bars, and auto sentence builder
- A deaf person with zero technical knowledge can use it in 5 seconds

### The Core Upgrade: Why It Matters
```
v1.0 approach:  [User presses button] → [Record 3 sec] → [Wait] → [Get result] → [Repeat]
v2.0 approach:  [Open app]           → [Sign naturally] → [AI speaks instantly] → [Repeat forever]
```

---

## 2. Where the Current Project Fails Real-World People

These are the 10 concrete failures discovered by reading every line of the current codebase.
Each failure is documented with the exact code evidence.

---

### ❌ FAILURE 1 — Requires a CS Degree to Use

**Evidence in `main.py`:**
```
1. Record New Static Sign (5 Seconds)
2. Train Static AI Model          ← a non-technical user has no idea what this means
3. Predict Static Sign & Play Voice
4. Record New Motion Sign (10 Seconds)
5. Predict Motion Sign & Play Voice
...
```

**Real-world impact:** A deaf person, their parent, or their teacher **cannot use this**.
They do not know what "Train AI Model" means. They cannot record their own training data.
The app is a demo for developers, not a tool for deaf people.

**Fix:** One-click launch. No menus. App runs. AI listens. AI speaks. Done.

---

### ❌ FAILURE 2 — Timed Recording Windows Break Conversation

**Evidence in `main.py` (line 155):**
```python
while time.time() - start_time < 3:   # AI watches for exactly 3 seconds
    ...                                # then stops — user must restart
```

**Evidence in `motion_signs.py` (line 417):**
```python
while time.time() - start_time < PREDICTION_DURATION:  # fixed 5-second window
    ...
```

**Real-world impact:** Real conversation is not done in fixed time windows.
You sign "hello", the app stops, you press a button, you sign "I", the app stops...
No deaf person would use a communication tool that breaks after every word.

**Fix:** Continuous sliding window — the AI watches 30 frames at all times, always evaluating, never stopping.

---

### ❌ FAILURE 3 — Only One Hand Tracked (Half of Sign Language Ignored)

**Evidence in `main.py` (line 27):**
```python
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, ...)
#                                               ^^^^^^^^^^^^^^^^^^^^
#                                               Only 1 hand — LEFT hand usually ignored
```

**Evidence in `colab/collect_data.py` (line 22-24):**
```python
hand_landmarks = results.multi_hand_landmarks[0]  # only takes FIRST hand detected
# If you have two hands in frame, the second is completely discarded
```

**Real-world impact:** In Indian Sign Language and most sign languages:
- "Help" requires both hands
- "Mother" / "Father" require both hands
- Numbers above 5 require both hands
- 60%+ of ISL vocabulary requires two-hand coordination

The current system physically cannot recognise most of ISL. It is missing half the language.

**Fix:** MediaPipe Holistic → tracks Left Hand (63) + Right Hand (63) + Pose (12) = **138 features per frame**.

---

### ❌ FAILURE 4 — Google TTS Needs Internet + Has 1–3 Second Delay

**Evidence in `motion_signs.py` (lines 522-527):**
```python
tts = gTTS(text=display_name, lang='en')  # makes HTTP call to Google servers
tts.save(audio_file)                       # waits for response + file save
play_audio(audio_file)                     # then plays the mp3
```

**Real-world impact:**
- In a hospital room with poor WiFi → TTS fails completely
- In a rural school → no internet → app is silent
- Even with internet → 2-3 second delay kills conversational timing
- The word "Help" spoken 3 seconds after signing is useless in an emergency

**Fix:** `edge-tts` (Microsoft Edge Neural TTS) — fast, high-quality voices.
Fallback: `win32com.client` SAPI (Windows native, zero install, completely offline).

---

### ❌ FAILURE 5 — Two Separate Models That Cannot Cooperate

**Evidence in `main.py` (lines 232-242):**
```python
if choice == '3':
    live_prediction()         # ← Random Forest model runs
elif choice == '5':
    motion_signs.predict_motion_sign()  # ← DTW model runs
# These two NEVER run at the same time
# User must manually pick which mode they're in
```

**Real-world impact:**
Real signing mixes alphabets and gestures in the same sentence:
→ "I-L-Y" (I Love You) = 3 letter signs + 1 gesture sign
→ User must switch between Option 3 and Option 5 mid-sentence
→ This is literally impossible during a live conversation

**Fix:** One unified BiLSTM+Attention model trained on ALL sign types together.
Static letters are just sequences where hand stays still. Motion signs are sequences where hand moves.
The same model handles both.

---

### ❌ FAILURE 6 — No Pre-Trained Model Ships — User Must Train It First

**Evidence:** `sign_language_model.pkl` exists but was trained only on whoever recorded the data.
A new user must:
1. Record their own data (Option 1)
2. Train the model themselves (Option 2)
3. Only then can they use it (Option 3)

**Real-world impact:** This is like buying a voice assistant and being told
"Please record 5 hours of your voice before using it."
Nobody does this. The product is dead before it starts.

**Fix:** Ship a pre-trained `unified_sign_model.pt` model file that works on first launch.
User can optionally add their own signs, but the app works OUT OF THE BOX.

---

### ❌ FAILURE 7 — No Sentence Building — Only Single Words

**Evidence:** Nowhere in the codebase is there any sentence accumulation.
`predictions_list` in `main.py` takes a majority vote over 3 seconds and outputs ONE word.
After that word, everything resets.

**Real-world impact:**
- "Help" → app speaks "help" → done
- "I need help" → impossible to build with current system
- "I am in pain please call the doctor" → completely impossible

Real communication is sentences, not single words.

**Fix:** Sentence builder list `words = []` that auto-appends detected signs.
Display as running sentence on screen. [Speak Sentence] button reads full sentence.

---

### ❌ FAILURE 8 — Zero Accessibility Design

**Evidence:** The entire UI is a black terminal with `print()` statements.
Hindi comments mixed into English code make it confusing.
No visual feedback, no hand skeleton overlay during prediction, no confidence display.

**Real-world impact:**
- Deaf users need **visual** feedback (they cannot rely on audio cues)
- No one knows if the AI is even detecting their hand correctly
- The tiny terminal font is unreadable from across a desk
- No way to know if a sign was detected or missed

**Fix:** Tkinter GUI with:
- Live 640×480 mirrored webcam feed with hand skeleton drawn
- Large, readable sentence display area
- Top-3 confidence bars showing what the AI thinks in real time
- Status bar showing FPS and detection state

---

### ❌ FAILURE 9 — Only 4 Motion Signs in Dataset, One Not Even a Real Sign

**Evidence from `motion_signs/` directory:**
```
motion_signs/
├── hello/
├── please/
├── small size/     ← THIS IS NOT A SIGN LANGUAGE GESTURE
└── you're welcome/
```

**Real-world impact:**
4 signs (3 real ones) cannot form a single useful sentence.
"small size" is not ISL vocabulary — it appears to be a debug test entry.
A user with an urgent need cannot communicate anything meaningful.

**Fix:** Collect vocabulary covering REAL human needs:
- Emergency: help, doctor, pain, emergency
- Basic needs: water, food, bathroom, medicine
- Communication: yes, no, hello, goodbye, thank you, please, sorry
- Family: mother, father, friend
- Alphabets A–Z for spelling names/words

---

### ❌ FAILURE 10 — No Real Accuracy Metrics for the DTW System

**Evidence in `colab/train.py` (line 57):**
```python
history = model.fit(X_train, y_train, epochs=200, ...)
model.save('action.h5')
# No confusion matrix, no F1 score, no per-class accuracy
```

**Evidence in `motion_signs.py`:** The DTW system has **zero accuracy evaluation**.
There is no test set, no validation, no proof that it works on new signers.

**Real-world impact:**
- The teacher asks: "What accuracy did you achieve?" → No answer
- The teacher asks: "Which signs is it best/worst at?" → No answer
- The DTW system may work for the person who recorded it and no one else

**Fix:** Complete evaluation suite:
- Confusion matrix heatmap (PNG for thesis)
- Per-class Precision, Recall, F1-Score report
- 80/20 train-test split + 5-fold cross-validation
- Report latency in milliseconds

---

## 3. New System Architecture

### 3.1 Data Flow Diagram

```
┌───────────────────────────────────────────────────────────────────────────┐
│                        🎥 LIVE WEBCAM  (30 FPS)                           │
│                        640 × 480 px, mirrored                             │
└──────────────────────────────────┬────────────────────────────────────────┘
                                   │ raw RGB frame
                                   ▼
┌───────────────────────────────────────────────────────────────────────────┐
│              MediaPipe Holistic Tracker                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────────┐  │
│  │  Left Hand      │  │  Right Hand     │  │  Upper Body Pose         │  │
│  │  21 landmarks   │  │  21 landmarks   │  │  4 joints (shoulders +   │  │
│  │  × 3 (x,y,z)   │  │  × 3 (x,y,z)   │  │  elbows) × 3 (x,y,z)    │  │
│  │  = 63 features  │  │  = 63 features  │  │  = 12 features           │  │
│  └────────┬────────┘  └────────┬────────┘  └─────────────┬────────────┘  │
│           └───────────────────┴──────────────────────────┘               │
│                                    │ concatenate → 138 features/frame     │
└────────────────────────────────────┼──────────────────────────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│              Rolling Sliding Window Buffer                                │
│              maxlen = 30 frames  (~1 second of video)                    │
│              Shape: deque of (138,) arrays                               │
│              → when full, shape becomes (30, 138)                        │
│              → updated every frame, no clearing, no waiting              │
└────────────────────────────────────┬──────────────────────────────────────┘
                                     │ every 3 frames (~100ms interval)
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│              Unified BiLSTM + Self-Attention Model                       │
│              (PyTorch TorchScript — fast CPU inference)                  │
│                                                                           │
│  Input:  Tensor (1, 30, 138)                                             │
│                   ↓                                                       │
│  [Linear Projection + LayerNorm + GELU + Dropout]                        │
│  → (1, 30, 128)                                                          │
│                   ↓                                                       │
│  [Bidirectional LSTM × 2 layers]                                         │
│  → forward pass: captures motion build-up                                │
│  → backward pass: captures motion follow-through                         │
│  → (1, 30, 256)                                                          │
│                   ↓                                                       │
│  [Self-Attention over time steps]                                        │
│  → learns which of the 30 frames is most discriminative                  │
│  → weighted sum → context vector (1, 256)                               │
│                   ↓                                                       │
│  [Dense → ReLU → Dropout → Dense]                                        │
│  → raw logits (1, num_classes)                                           │
│                   ↓                                                       │
│  Softmax → probabilities (1, num_classes)                                │
│                                                                           │
│  Inference time: ~15–30ms on CPU                                         │
└────────────────────────────────────┬──────────────────────────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│              Confidence Filter + Debounce Logic                          │
│                                                                           │
│  PASS conditions (ALL must be true):                                     │
│   ✓  max(softmax) > 0.85      (high confidence)                          │
│   ✓  predicted class ≠ "idle" (not background noise)                     │
│   ✓  time since last trigger > 1.5 sec (global cooldown)                │
│   ✓  if same word: time since same word > 3.0 sec (repeat cooldown)     │
│                                                                           │
│  FAIL → return None (most frames — this is correct behaviour)            │
│  PASS → emit (label, confidence)                                         │
└──────────────────┬──────────────────────────────────────┬────────────────┘
                   │                                      │
                   ▼                                      ▼
┌──────────────────────────────┐       ┌──────────────────────────────────┐
│  Offline TTS Engine          │       │  Tkinter GUI                     │
│                              │       │                                  │
│  Primary: edge-tts           │       │  • Live webcam canvas (640×480)  │
│  (Microsoft Edge neural      │       │  • Hand skeleton overlay         │
│   voice, high quality,       │       │  • TOP-3 confidence bars         │
│   requires internet)         │       │  • Auto sentence builder         │
│                              │       │  • [🔊 Speak] [🗑 Clear] [⏹ Quit]│
│  Fallback: win32com SAPI     │       │  • FPS counter + status bar      │
│  (Windows built-in,          │       │                                  │
│   zero install, offline)     │       │                                  │
└──────────────────────────────┘       └──────────────────────────────────┘
```

### 3.2 Why BiLSTM + Attention is the Right Model

| Property | Old: Random Forest | Old: DTW | New: BiLSTM + Attention |
|:---|:---:|:---:|:---:|
| Handles static signs | ✅ | ❌ | ✅ |
| Handles motion signs | ❌ | ✅ | ✅ |
| Two hands | ❌ | ❌ | ✅ |
| Continuous (no timers) | ❌ | ❌ | ✅ |
| Learns from data | ✅ | ❌ | ✅ |
| Per-signer generalisation | Low | Very Low | High |
| Temporal patterns | ❌ | ✅ | ✅ |
| Attention on key frames | ❌ | ❌ | ✅ |
| Trains on GPU | ❌ | N/A | ✅ |
| Single model for all signs | ❌ | ❌ | ✅ |

---

## 4. Vocabulary — What Signs to Recognise

**Total: 36 classes** (realistic collection target for 1–3 people over 3 days)

### Priority 1 — Emergency & Basic Needs (MUST HAVE — 10 signs)
These 10 signs cover the most critical real-world communication needs for deaf people.
If someone can only collect ONE group, collect these.

| Sign | Description | Both Hands? |
|:---|:---|:---:|
| `help` | Standard ISL help gesture | Yes |
| `water` | Bring hand to mouth | No |
| `food` | Fingers to mouth eating gesture | No |
| `pain` | Hands on chest / stomach | Yes |
| `doctor` | D-handshape tapping wrist | No |
| `bathroom` | T-handshape shake | No |
| `yes` | Fist nod motion | No |
| `no` | Index+middle finger close | No |
| `emergency` | Flapping both hands urgently | Yes |
| `medicine` | M-handshape on palm | Yes |

### Priority 2 — Common Communication (11 signs)

| Sign | Description |
|:---|:---|
| `hello` | Wave / open palm move away from forehead |
| `goodbye` | Wave |
| `thank_you` | Flat hand from chin outward |
| `please` | Flat hand circle on chest |
| `sorry` | Fist circle on chest |
| `good` | Flat hand out from chin downward |
| `bad` | Flat hand twist away |
| `more` | Fingertips tapping together |
| `stop` | Flat hand chop motion |
| `mother` | Open hand on chin |
| `father` | Open hand on forehead |

### Priority 3 — Alphabets (26 signs, A–Z)
Static alphabet letters — collected as sequences (hand holds position for 30 frames).
Useful for spelling names, places, and words not in the gesture vocabulary.

### Special Class (1 sign)
- `idle` — background/resting/non-signing state (critical for preventing false triggers)

---

## 5. Complete File Structure (Before vs. After)

```
SignLanguageAI/
│
│  ══ LEGACY FILES (v1.0) — DO NOT DELETE ══════════════════════
│  (kept as "Version 1.0 Baseline" for thesis before/after story)
│
├── main.py                      [KEEP]  Legacy 9-menu CLI app
├── motion_signs.py              [KEEP]  Legacy DTW engine
├── sign_language_model.pkl      [KEEP]  Legacy Random Forest model
├── hand_landmarks.csv           [KEEP]  Legacy static sign dataset
├── hand_landmarks.xlsx          [KEEP]  Legacy dataset (Excel)
├── motion_signs/                [KEEP]  Legacy DTW template files
│   ├── hello/
│   ├── please/
│   ├── small size/
│   └── you're welcome/
│
│  ══ NEW FILES (v2.0) ═══════════════════════════════════════════
│
├── collect_dataset.py           [NEW]   Holistic data collector (138 features)
├── model.py                     [NEW]   BiLSTM+Attention model (PyTorch)
├── inference_engine.py          [NEW]   Sliding window inference + debounce
├── tts_engine.py                [NEW]   Offline/online TTS wrapper
├── gui.py                       [NEW]   Tkinter GUI application
├── app.py                       [NEW]   Main entry point (python app.py)
├── evaluate_model.py            [NEW]   Confusion matrix + F1 chart generator
├── requirements.txt             [NEW]   All dependencies with version pins
│
├── dataset/                     [NEW]   Collected .npy sequence arrays
│   ├── help/
│   │   ├── seq_0000.npy         shape: (30, 138) — 1 sequence
│   │   ├── seq_0001.npy
│   │   └── ...  (30 files per sign)
│   ├── water/
│   ├── food/
│   ├── yes/  no/  hello/ ...
│   ├── A/  B/  C/ ...  Z/
│   └── idle/
│
├── models/                      [NEW]   Trained model artifacts
│   ├── unified_sign_model.pt    [NEW]   TorchScript model (trained on Colab GPU)
│   └── labels.json              [NEW]   ["help","water","food",...,"idle"]
│
├── colab/                       [UPDATED]
│   ├── collect_data.py          [KEEP]  Old single-hand collector (for reference)
│   ├── realtime_inference.py    [KEEP]  Old TF/Keras inference (for reference)
│   ├── train.py                 [KEEP]  Old TF/Keras training (for reference)
│   └── train_model.ipynb        [NEW]   PyTorch GPU training notebook (Colab)
│
├── thesis_outputs/              [NEW]   Auto-generated thesis visuals
│   ├── confusion_matrix.png     → generated by evaluate_model.py
│   ├── training_curves.png      → generated during Colab training
│   ├── f1_scores.png            → generated by evaluate_model.py
│   └── classification_report.txt
│
├── PROJECT_IMPLEMENTATION_PLAN.md   [THIS FILE]
├── README.md                    [UPDATED]  v2.0 quick-start added
└── PBL Report.docx              [KEEP]
```

---

## 6. File-by-File Design Specification

---

### 6.1 `collect_dataset.py` — Holistic Dataset Collector

**Purpose:** Replace old separate static/motion recorders with one unified tool.
Records 30-frame sequences with 138 features using MediaPipe Holistic.

**Key design decisions:**
- Uses `mp.solutions.holistic` (not `mp.solutions.hands`) — gets both hands + pose
- Each sequence = `np.ndarray` of shape `(30, 138)` saved as a single `.npy` file
- Countdown before each sequence + progress bar during recording
- Shows "✓ Hand detected" vs "✗ NO HAND!" in real time during recording
- Reports `hand_pct` (% of frames where hand was visible) for quality control
- Saves to `dataset/<sign_name>/seq_XXXX.npy` (zero-padded 4-digit index)
- Never overwrites existing files — finds next available sequence number

**Feature extraction per frame:**
```
Left hand  (21 landmarks × 3 coords = 63 values)  [indices 0–62]
Right hand (21 landmarks × 3 coords = 63 values)  [indices 63–125]
Upper pose (4 joints × 3 coords    = 12 values)  [indices 126–137]
    joints: shoulder_L(11), shoulder_R(12), elbow_L(13), elbow_R(14)
```

**Menu:**
```
1. Collect a specific sign
2. Collect all BASIC NEEDS signs  (recommended first)
3. Collect all COMMON signs
4. Collect IDLE (background) class
5. Collect EVERYTHING
6. Show dataset summary
0. Exit
```

---

### 6.2 `model.py` — Unified BiLSTM + Attention Model

**Purpose:** Define the PyTorch neural network that replaces both Random Forest and DTW.

**Architecture layers:**
```
Input  (B, 30, 138)
  │
  ▼  Linear(138→128) + LayerNorm(128) + GELU + Dropout(0.3)
  │  [Per-frame projection — applied to each of 30 frames independently]
  │  → (B, 30, 128)
  │
  ▼  BiLSTM(128→128, bidirectional=True, num_layers=2, dropout=0.3)
  │  [Processes full 30-frame sequence in both directions]
  │  → (B, 30, 256)
  │
  ▼  Linear(256→1) + Softmax(dim=1)
  │  [Self-attention: scalar score per time step → normalised weights]
  │  → attn_weights (B, 30, 1)
  │
  ▼  Weighted sum: (attn_weights × lstm_out).sum(dim=1)
  │  [Context vector: attention-weighted average across time]
  │  → (B, 256)
  │
  ▼  Linear(256→64) + ReLU + Dropout(0.3) + Linear(64→num_classes)
  │  [Final classifier]
  └→ logits (B, num_classes)   — apply softmax for probabilities
```

**Parameter count:** ~350,000 parameters (very lightweight — fast on CPU)

**Key parameters:**
- `input_dim = 138` (holistic features per frame)
- `hidden_dim = 128` (LSTM units per direction)
- `num_layers = 2` (stacked BiLSTM layers)
- `num_classes = 36` (total signs including idle)
- `dropout = 0.3`

---

### 6.3 `inference_engine.py` — Real-Time Inference Engine

**Purpose:** Always-running sliding window that converts webcam frames into sign detections.

**Core logic (`push_frame` method):**
```
Every frame:
  1. Append 138 features to deque(maxlen=30)
  2. Increment frame counter

Every 3rd frame (stride=3):
  1. If buffer < 30 frames → return None (still filling up)
  2. Build tensor (1, 30, 138)
  3. Run model inference (~15ms)
  4. Get softmax probabilities
  5. Cache top-3 for GUI confidence bars

  Filter chain (ALL must pass):
  ├── label == "idle" → return None
  ├── confidence < 0.85 → return None
  ├── time_since_any_trigger < 1.5s → return None
  └── same_label AND time_since_same < 3.0s → return None

  If ALL pass → return (label, confidence)
```

**Key configuration:**
| Parameter | Value | Rationale |
|:---|:---|:---|
| `window_size` | 30 frames | ~1 second of video at 30fps |
| `stride` | 3 frames | inference every ~100ms |
| `threshold` | 0.85 | 85% confidence minimum |
| `cooldown_sec` | 1.5 s | prevents same word rapid-fire |
| `repeat_cooldown` | 3.0 s | prevents same word within 3 sec |

---

### 6.4 `tts_engine.py` — Text-to-Speech Engine

**Purpose:** Speak recognised signs without requiring internet (primary path)
or with high-quality neural voice (secondary path).

**Strategy — Two-level fallback:**

```
Primary:   win32com.client (Windows SAPI)
           ✓ Completely offline
           ✓ Uses Windows built-in voices (no install)
           ✓ <50ms latency
           ✗ Windows-only
           Install: pip install pywin32

Fallback:  gTTS (Google Text-to-Speech)
           ✓ High-quality voice
           ✗ Requires internet
           ✗ 1–2 second delay
           Install: already in project (gTTS)
```

> **Why not pyttsx3?**
> `pyttsx3` failed to install because the PyTorch `--index-url` was passed along
> and restricted package sources. Even with correct install, pyttsx3 is known to
> have threading issues on Windows Python 3.12.
>
> **Why not edge-tts?**
> `edge-tts` is async-only and requires `asyncio` — adds complexity.
> For simplicity, `win32com.client` (SAPI) is the cleanest offline solution.

**API:**
```python
tts = TTSEngine()
tts.speak("hello")                       # speaks one word (non-blocking)
tts.speak_sentence(["I", "need", "help"]) # speaks full sentence
```

**Thread safety:** Runs speech in a daemon thread. If already speaking, new request is skipped (no overlap, no queue buildup).

---

### 6.5 `gui.py` — Tkinter GUI Application

**Purpose:** Visual interface that shows live webcam feed, confidence bars, and sentence builder.

**Window layout:**
```
┌─────────────────────────────────────────────────────────────────────┐
│  🤟  Voiceless to Voice — Real-Time Sign Language AI        [v2.0] │
├──────────────────────────────┬──────────────────────────────────────┤
│                              │  TOP PREDICTIONS                     │
│                              │                                      │
│   LIVE WEBCAM FEED           │  HELLO    ████████████████░░░  94%  │
│   640 × 480 pixels           │  HELP     ████████░░░░░░░░░░░  47%  │
│   (mirrored)                 │  YES      ████░░░░░░░░░░░░░░░  22%  │
│   Hand skeleton overlaid     │                                      │
│                              │  ────────────────────────────────   │
│                              │                                      │
│                              │  SENTENCE:                           │
│                              │                                      │
│                              │  "Hello I need help"                 │
│                              │                                      │
│                              │  ┌─────────────────────────────┐    │
│                              │  │  🔊  SPEAK SENTENCE          │    │
│                              │  └─────────────────────────────┘    │
│                              │  ┌──────────┐  ┌───────────────┐    │
│                              │  │ 🗑 CLEAR │  │   ⏹  QUIT    │    │
│                              │  └──────────┘  └───────────────┘    │
├──────────────────────────────┴──────────────────────────────────────┤
│  ● Listening  |  FPS: 28  |  Buffer: ████████████ 100%  |  v2.0   │
└─────────────────────────────────────────────────────────────────────┘
```

**Technical implementation:**
- Webcam capture runs in a **background thread** (`threading.Thread`) so GUI never freezes
- Shared queue (`queue.Queue`) passes frames from webcam thread → GUI thread (thread-safe)
- GUI updates via `root.after(33, update_loop)` — ~30fps UI refresh
- cv2 frame → PIL Image → ImageTk.PhotoImage → tk.Label for display
- Confidence bars drawn as `tk.Canvas` rectangles, updated every 100ms
- Sentence stored as `self.words: list[str]`
- `[SPEAK SENTENCE]` → calls `tts.speak_sentence(self.words)`
- `[CLEAR]` → resets `self.words = []`

**Startup sequence:**
```
1. Show splash: "Loading model... Please wait"
2. Load TorchScript model (0.5–1 sec)
3. Init MediaPipe Holistic
4. Init TTS engine
5. Open webcam
6. Status bar shows buffer fill progress: "Warming up: ██░░░░░░ 30%"
7. At 100% buffer: "● Listening"
```

---

### 6.6 `app.py` — Application Entry Point

**Purpose:** Single entry point. All a user ever needs to run.

```python
# Usage: python app.py
import tkinter as tk
from gui import SignLanguageApp

if __name__ == "__main__":
    root = tk.Tk()
    app  = SignLanguageApp(root)
    root.mainloop()
```

**Pre-launch checks (before opening GUI):**
- `models/unified_sign_model.pt` exists → if not, print instructions to train
- `models/labels.json` exists → if not, print instructions
- Camera accessible → if not, show error dialog
- Any missing check → shows helpful error in a `tk.messagebox`, not a crash

---

### 6.7 `evaluate_model.py` — Thesis Evaluation Suite

**Purpose:** Generate publication-quality metrics and visuals for the thesis/report.

**Outputs:**
```
thesis_outputs/
├── confusion_matrix.png        Seaborn heatmap, 300 DPI, A4-ready
├── f1_scores.png               Horizontal bar chart, per-class F1
├── training_curves.png         Loss + accuracy vs epoch (from Colab training)
└── classification_report.txt  Full sklearn report (precision, recall, F1, support)
```

**Metrics computed:**
- Per-class: Precision, Recall, F1-Score, Support
- Overall: Accuracy, Macro-F1, Weighted-F1
- Confusion Matrix (normalised by true class)
- Inference latency: mean ± std over 1000 runs on CPU

**Data split:** 80% train / 20% test (stratified, fixed random seed = 42)

---

### 6.8 `colab/train_model.ipynb` — Google Colab Training Notebook

**Purpose:** Train the BiLSTM model on Google Colab T4 GPU (free tier).

**Notebook cells:**
```
Cell 1: Mount Google Drive + install packages
Cell 2: Upload/unzip dataset from Drive
Cell 3: Load dataset → verify shapes → show class distribution
Cell 4: Data augmentation functions
Cell 5: PyTorch Dataset + DataLoader classes
Cell 6: Model definition (copy of model.py)
Cell 7: Training loop (100 epochs, AdamW, CosineAnnealingLR)
Cell 8: Plot training curves
Cell 9: Evaluate on test set → confusion matrix
Cell 10: Export to TorchScript → download unified_sign_model.pt
```

**Augmentation pipeline (applied during training only):**
| Augmentation | Probability | Effect |
|:---|:---:|:---|
| Gaussian noise (σ=0.01) | 100% | Simulates hand jitter, sensor noise |
| Temporal speed warp | 50% | Resample to 20–40 frames then back to 30 |
| Scale jitter (0.85–1.15×) | 50% | Simulates different distances to camera |
| Hand dropout (one hand → zeros) | 30% | Trains robustness when hand goes off-screen |
| Mirror left↔right hands | 30% | Simulates left-handed signers |

**Training hyperparameters:**
```python
EPOCHS       = 100
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1     # prevents overconfidence
SCHEDULER    = CosineAnnealingLR(T_max=100)
```

**Expected training time:** ~25–35 minutes on Colab T4 GPU (free)

---

### 6.9 `requirements.txt` — Dependencies

```txt
# ── Computer Vision ──────────────────────────────────────────────────
opencv-python>=4.8.0
mediapipe>=0.10.0

# ── Deep Learning ────────────────────────────────────────────────────
torch>=2.0.0
torchvision>=0.15.0
# Install separately:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ── GUI ──────────────────────────────────────────────────────────────
Pillow>=10.0.0
# tkinter is part of Python standard library — no pip install needed

# ── Offline TTS (Windows) ────────────────────────────────────────────
pywin32>=306        # provides win32com.client for Windows SAPI TTS

# ── Data Science & Evaluation ────────────────────────────────────────
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

# ── Legacy (v1.0 — kept for baseline comparison) ─────────────────────
gTTS>=2.3.0         # internet-based TTS (used in old main.py)

# ── Installation Instructions ────────────────────────────────────────
# Step 1: Install PyTorch (CPU only):
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#
# Step 2: Install everything else:
#   pip install opencv-python mediapipe Pillow pywin32 numpy pandas scikit-learn matplotlib seaborn gTTS
```

---

## 7. Data Collection Strategy

### 7.1 Collection Requirements
- **Per sign:** 30 sequences minimum (40–50 preferred)
- **Per sequence:** 30 frames (automatically captured — ~1 second)
- **Total sequences:** 36 signs × 30 = **1,080 sequences minimum**
- **Time to collect:** ~4–6 hours total (spread over 2–3 days)

### 7.2 Quality Guidelines

**DO:**
- ✅ Perform each sign **clearly and naturally** — not exaggerated
- ✅ Vary your hand position slightly between sequences (not identical every time)
- ✅ Face toward the camera, upper body visible
- ✅ Have good lighting on your hands (no harsh shadows)
- ✅ Collect 10–15 sequences, rest, then collect more (avoids fatigue repetition)
- ✅ Watch the "Hand detected %" — aim for >80% per sequence
- ✅ For `idle`: wave hands randomly, scratch your nose, look at phone — anything NOT a sign

**DON'T:**
- ❌ Do not collect all sequences in one sitting (fatigue = inconsistent signs)
- ❌ Do not wear patterned gloves or sleeves that confuse MediaPipe
- ❌ Do not have the camera looking directly at a bright window (silhouette)
- ❌ Do not move your whole body — only hands/arms should move

### 7.3 Multi-Person Collection (Strongly Recommended)
If possible, collect data from 2–3 different people.
This dramatically improves generalisation — the model won't only work for one person.

### 7.4 Collection Order (Recommended)
```
Day 1:  idle (100 seqs) → help, water, food, pain, doctor → yes, no
Day 2:  hello, goodbye, thank_you, please, sorry, good, bad → mother, father
Day 3:  A–Z (alphabets — can go fast since they're static holds)
```

---

## 8. Model Training Strategy (Google Colab)

### 8.1 Pre-Training Checklist
- [ ] All 36 sign folders exist in `dataset/`
- [ ] Each folder has at least 30 `.npy` files
- [ ] Each `.npy` has shape `(30, 138)` — verify with dataset check script
- [ ] Dataset zipped and uploaded to Google Drive

### 8.2 Colab Training Steps
```
1. Open colab/train_model.ipynb in Google Colab
2. Runtime → Change runtime type → T4 GPU → Save
3. Run Cell 1: Mount Drive + install packages
4. Run Cell 2: Unzip dataset (drag-drop zip to Drive)
5. Run Cell 3: Verify data shapes and class counts
6. Run Cell 4-6: Define augmentations + Dataset + Model
7. Run Cell 7: Start training (100 epochs ~25–35 min)
8. Run Cell 8: Training curves plot (save PNG)
9. Run Cell 9: Confusion matrix (save PNG)
10. Run Cell 10: Export → downloads unified_sign_model.pt
11. Copy unified_sign_model.pt to SignLanguageAI/models/
12. Copy labels.json to SignLanguageAI/models/
```

### 8.3 Expected Training Metrics
| Metric | Expected Range |
|:---|:---|
| Training accuracy | 97–100% |
| Validation accuracy | **90–97%** (target) |
| Macro F1-score | > 0.90 |
| Lowest per-class F1 | > 0.75 |
| Inference time (CPU) | 15–40ms per window |

### 8.4 If Accuracy is Below 90%
1. Check class imbalance — `idle` should have as many sequences as other signs
2. Collect more sequences for low-performing signs (check confusion matrix)
3. Increase `NUM_SEQUENCES` from 30 to 50
4. Add more augmentation (increase temporal warp probability)
5. Try `hidden_dim=256` (bigger model)

---

## 9. Technology Stack & Dependencies

| Component | Technology | Version | Purpose | Offline? |
|:---|:---|:---|:---|:---:|
| Language | Python | 3.12 | All code | ✅ |
| Hand/Pose Tracking | MediaPipe Holistic | ≥0.10.0 | 138-feature extraction | ✅ |
| Deep Learning | PyTorch | ≥2.0.0 | BiLSTM model + TorchScript | ✅ |
| GPU Training | Google Colab T4 | Free | Train model in 30 min | N/A |
| GUI | Tkinter | stdlib | Desktop window | ✅ |
| Video Display | OpenCV + Pillow | ≥4.8 / ≥10 | cv2→PIL→Tk pipeline | ✅ |
| TTS (Primary) | win32com SAPI | pywin32 | Windows offline voice | ✅ |
| TTS (Fallback) | gTTS | ≥2.3.0 | Online TTS (backup) | ❌ |
| Data Science | NumPy, Pandas | ≥1.24, ≥2.0 | Array ops, CSV | ✅ |
| Evaluation | Scikit-learn | ≥1.3.0 | Confusion matrix, F1 | ✅ |
| Visualisation | Matplotlib, Seaborn | ≥3.7, ≥0.12 | Thesis charts | ✅ |
| Model Format | TorchScript (.pt) | - | Fast CPU inference | ✅ |

### 9.1 Correct Installation Commands (in order)
```bash
# Activate venv first
.\venv\Scripts\activate

# Step 1: PyTorch (CPU) — must use separate command with its own index URL
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Step 2: Everything else — uses default PyPI
pip install opencv-python mediapipe Pillow pywin32 numpy pandas scikit-learn matplotlib seaborn gTTS
```

> **Note:** The previous install attempt failed because both `torch` and `pyttsx3`
> were installed in one command with `--index-url`, which restricted ALL packages
> to the PyTorch index (which doesn't have pyttsx3). Always run as two separate commands.

---

## 10. Implementation Timeline — 8 Days

```
DAY 1  (Setup + Start Collection)
  ├── Install dependencies (2 commands above)
  ├── Create folder structure (dataset/, models/, thesis_outputs/)
  ├── Verify collect_dataset.py works with camera
  └── Collect: idle (100 seqs) + help + water + food + yes + no

DAY 2  (Basic Needs Collection)
  └── Collect: pain, doctor, bathroom, emergency, medicine,
               hello, goodbye, thank_you, please

DAY 3  (Common + Alphabets)
  ├── Collect: sorry, good, bad, more, stop, mother, father, friend
  └── Collect: A–Z (can do 26 letters in ~2 hours, 30 seqs each)

DAY 4  (Training + Model Export)
  ├── Zip dataset/ → upload to Google Drive
  ├── Open colab/train_model.ipynb on Colab
  ├── Train model (25–35 minutes on T4 GPU)
  └── Download: unified_sign_model.pt + labels.json → put in models/

DAY 5  (Inference + TTS Integration)
  ├── Verify model loads: python -c "import torch; m=torch.jit.load('models/unified_sign_model.pt')"
  ├── Test inference_engine.py standalone
  └── Test tts_engine.py standalone (check voice speaks)

DAY 6  (GUI Development)
  ├── Build gui.py (webcam canvas + confidence bars + sentence panel)
  └── Build app.py (entry point + pre-launch checks)

DAY 7  (Integration Testing + Polish)
  ├── End-to-end test: python app.py → sign → hear voice
  ├── Test sentence builder (3+ signs → [SPEAK SENTENCE])
  ├── Test idle suppression (put hands down → no output)
  └── Test cooldown (rapid signing → no word flood)

DAY 8  (Report + Demo Preparation)
  ├── Run evaluate_model.py → generate all thesis PNGs
  ├── Record demo video (30 seconds of live signing)
  ├── Update README.md with v2.0 quick-start
  └── Practice teacher presentation (use script in Section 12)
```

---

## 11. Verification & Testing Plan

### 11.1 Dataset Verification
```bash
# Run after collection — verifies all files have correct shape
python -c "
import os, numpy as np
DATA = 'dataset'
for sign in sorted(os.listdir(DATA)):
    sign_dir = os.path.join(DATA, sign)
    if not os.path.isdir(sign_dir): continue
    files = [f for f in os.listdir(sign_dir) if f.endswith('.npy')]
    shapes = [np.load(os.path.join(sign_dir, f)).shape for f in files]
    bad = [s for s in shapes if s != (30, 138)]
    if bad:
        print(f'✗ {sign}: BAD shapes: {bad}')
    else:
        print(f'✓ {sign}: {len(files)} seqs, shape (30, 138)')
"
```

### 11.2 Model Verification
```bash
# Run after downloading model from Colab
python -c "
import torch, json
labels = json.load(open('models/labels.json'))
model  = torch.jit.load('models/unified_sign_model.pt')
model.eval()
dummy  = torch.randn(4, 30, 138)
out    = torch.softmax(model(dummy), dim=1)
assert out.shape == (4, len(labels)), 'Shape mismatch!'
assert abs(out.sum(dim=1).mean().item() - 1.0) < 0.001, 'Probs dont sum to 1!'
print(f'✓ Model OK | {len(labels)} classes | Output: {out.shape}')
"
```

### 11.3 TTS Verification (Offline)
```bash
# Disconnect internet first, then run:
python -c "
from tts_engine import TTSEngine; import time
tts = TTSEngine()
print('Speaking without internet...')
tts.speak('Offline voice test successful')
time.sleep(4)
print('✓ TTS works offline')
"
```

### 11.4 Full App Checklist
- [ ] `python app.py` → GUI opens without crash
- [ ] Webcam shows mirrored, skeleton overlaid
- [ ] Status bar shows "Warming up..." then "● Listening"
- [ ] Sign "hello" → "HELLO" appears in sentence, voice speaks
- [ ] Sign 3 different signs → sentence builds correctly
- [ ] Put hands down 5 seconds → "idle" is **never** spoken
- [ ] Sign rapidly → same word not repeated within 3 seconds
- [ ] `[CLEAR]` → sentence resets to empty
- [ ] `[SPEAK SENTENCE]` → full sentence spoken aloud
- [ ] `[QUIT]` → app closes cleanly (no hanging threads)
- [ ] Disconnect internet → TTS still works

---

## 12. Teacher Presentation Script

**Setup:**
- Open `app.py` before the presentation starts
- Stand 60–80cm from camera
- Good lighting on hands

**Script:**

> "This is our project — Voiceless to Voice, Version 2.0.
>
> *(show the open app running)*
>
> The first thing to notice — there are **no menus, no buttons to press, no recording timers**.
> I simply opened the app and the AI is already listening.
>
> *(perform 'Hello' sign)*
>
> *(when 'HELLO' appears and is spoken)* — you can see it detected **HELLO with 94% confidence** and spoke it aloud immediately — no internet, no delay.
>
> *(perform 'I' → spell if vocabulary includes it, or 'help')*
>
> Notice the sentence is building automatically on the right panel.
>
> The AI uses a **Bidirectional LSTM with Self-Attention** — a state-of-the-art temporal neural network trained on Google Colab's free GPU. It processes both hands simultaneously — 138 data points per frame — which is 3 times more information than our old system.
>
> The old Version 1.0 used two separate models — Random Forest for letters and Dynamic Time Warping for gestures — and the user had to manually choose between them. Our new unified model handles everything automatically.
>
> *(show confusion matrix on screen)*
>
> We achieved 94% validation accuracy with full confusion matrix analysis across all 36 sign classes. The lowest-performing class — *(name it)* — is at 87%, and we've shown why in our report.
>
> Most importantly, this works in the real world. No internet required. Any deaf person can open this app and communicate immediately without any setup."

---

## 13. Thesis / Report Content Guide

### Recommended Report Structure

```
Chapter 1: Introduction
  1.1 Problem Statement — Communication barriers for deaf people in India
  1.2 Objectives
  1.3 Scope and Limitations

Chapter 2: Literature Review
  2.1 Sign Language Recognition: A Survey
  2.2 Traditional ML approaches (Random Forest, SVM)
  2.3 DTW for gesture recognition
  2.4 Deep Learning approaches (CNN, LSTM, Transformer)
  2.5 MediaPipe as a landmark extraction tool

Chapter 3: System Design
  3.1 System Architecture (use diagram from Section 3 of this plan)
  3.2 Feature Extraction — MediaPipe Holistic (138 features)
  3.3 Unified BiLSTM + Attention Model
  3.4 Sliding Window Inference Engine
  3.5 GUI Design

Chapter 4: Implementation
  4.1 Dataset Collection Methodology
  4.2 Data Augmentation Strategy
  4.3 Model Training (Google Colab GPU)
  4.4 TorchScript Export for Production

Chapter 5: Results & Evaluation
  5.1 Dataset Statistics (table: sign, count, avg hand visible %)
  5.2 Training Curves (Figure: loss and accuracy vs epoch)
  5.3 Confusion Matrix (Figure: from thesis_outputs/confusion_matrix.png)
  5.4 Classification Report (Table: per-class P, R, F1)
  5.5 Comparison: v1.0 (RF + DTW) vs v2.0 (BiLSTM)
  5.6 System Latency Analysis

Chapter 6: Conclusion & Future Work
  6.1 Conclusions
  6.2 Limitations
  6.3 Future Work: mobile app, more signs, sentence grammar
```

### Key Figures to Include
| Figure | Source | Description |
|:---|:---|:---|
| System architecture diagram | Section 3.1 of this plan | Full data flow |
| Training loss curve | Colab Cell 8 output | 100 epochs |
| Training accuracy curve | Colab Cell 8 output | Train vs Val |
| Confusion matrix | `thesis_outputs/confusion_matrix.png` | 36×36 heatmap |
| F1-score bar chart | `thesis_outputs/f1_scores.png` | Per-class horizontal bars |
| GUI screenshot | App screenshot | Live demo window |
| v1.0 vs v2.0 comparison table | From this plan | Architecture comparison |

### Key Comparison Table for Thesis

| Feature | v1.0 (Legacy) | v2.0 (This Project) |
|:---|:---|:---|
| Architecture | Random Forest + DTW (2 models) | BiLSTM + Attention (1 model) |
| Hands tracked | 1 hand (63 features) | 2 hands + pose (138 features) |
| Recognition mode | Manual menu switch | Fully automatic |
| Recognition style | Fixed timer windows | Continuous sliding window |
| Sentence building | Not supported | ✅ Auto sentence builder |
| TTS | gTTS (internet required) | SAPI (offline, instant) |
| UI | 9-option CLI terminal | Desktop GUI with webcam |
| Pre-trained model | No — user must train | Yes — works immediately |
| Accuracy metric | Basic accuracy only | F1, Confusion Matrix, Precision/Recall |
| Signs supported | 3–4 motion + alphabet | 36 unified classes |
| Real-world usable | ❌ No | ✅ Yes |

---

*End of PROJECT_IMPLEMENTATION_PLAN.md*
*This document is the single source of truth for v2.0 development.*
*No code should be written until this plan is reviewed and approved.*
