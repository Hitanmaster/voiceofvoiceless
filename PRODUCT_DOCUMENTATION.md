# Sign Language to Voice AI System — Product Specification & Technical Architecture

## 📋 Table of Contents
1. [Product Overview](#1-product-overview)
2. [What the Product Currently Does](#2-what-the-product-currently-does)
3. [How It Works: Technical Architecture & Methodology](#3-how-it-works-technical-architecture--methodology)
   - [Hand Landmark Extraction Pipeline](#31-hand-landmark-extraction-pipeline)
   - [Static Sign Recognition Engine](#32-static-sign-recognition-engine)
   - [Dynamic Motion Sign Recognition Engine](#33-dynamic-motion-sign-recognition-engine)
   - [Text-to-Speech (TTS) Voice Synthesis](#34-text-to-speech-tts-voice-synthesis)
4. [Complete Technology Stack](#4-complete-technology-stack)
5. [Data Flow Architecture](#5-data-flow-architecture)
6. [Project Structure & File Organization](#6-project-structure--file-organization)

---

## 1. Product Overview

The **Sign Language to Voice AI System** is an end-to-end, CPU-optimized computer vision and machine learning solution designed to assist deaf and hard-of-hearing individuals. It converts both **static hand signs** (letters, static poses) and **dynamic gesture motions** (such as "Hello", "Goodbye", "Thank You", "Yes", "No") into real-time spoken audio and on-screen text.

By capturing live webcam streams, extracting 3D hand landmark spatial features, and processing them through machine learning and temporal alignment algorithms, the product provides instant voice feedback without requiring specialized hardware or GPU acceleration.

---

## 2. What the Product Currently Does

The system provides an interactive command-line interface (CLI) driven workflow along with modular execution scripts:

### 🎙️ Core Capabilities
1. **Static Hand Sign Recognition (Letters & Poses)**
   - Captures and classifies single-frame hand shapes and poses (e.g., sign language alphabet A–Z, numbers, custom static poses).
   - Observes live camera input for 3 seconds, predicts the pose, displays real-time prediction overlays, and speaks the predicted label.

2. **Dynamic Motion Sign Recognition (Continuous Gestures)**
   - Records and tracks movement trajectories of hands over time (5 to 10 seconds duration).
   - Recognizes dynamic gestures (e.g., waving "Hello", waving "Goodbye", bowing "Thank You") regardless of variations in speed or movement distance.
   - Synthesizes and plays back speech for recognized continuous motion gestures.

3. **Data Recording & Dataset Management**
   - **Guided Dataset Creation**: Records raw 3D hand landmark coordinates directly from live webcam streams with timer countdowns.
   - **Quick Batch Collection**: Provides preset lists for fast collection of motion sign gesture sequences.
   - **Dataset Inspection & Reset**: Allows users to view recorded frame counts, check class distributions, or perform clean dataset wipes.

4. **Automated Machine Learning Model Training**
   - Trains and evaluates static pose classifiers locally on recorded dataset samples (`hand_landmarks.csv`).
   - Exports trained models for fast real-time inference.

5. **Cross-Platform Text-to-Speech (TTS)**
   - Converts recognized sign text into natural MP3 audio files (`output.mp3`, `motion_output.mp3`).
   - Plays audio automatically across Windows, macOS, and Linux native media handlers.

6. **Cloud & Notebook Integration**
   - Includes standalone scripts in `colab/` for remote execution, headless landmark extraction, and model training inside Google Colab environments.

---

## 3. How It Works: Technical Architecture & Methodology

```
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────────┐
│ Webcam Video    │ ──> │ Google MediaPipe     │ ──> │ 21 Keypoint 3D Coordinates │
│ Input Stream    │     │ Hand Tracking        │     │ (x, y, z) -> 63 Features   │
└─────────────────┘     └──────────────────────┘     └────────────────────────────┘
                                                                   │
                                ┌──────────────────────────────────┴──────────────────────────────────┐
                                ▼                                                                     ▼
                   [STATIC SIGN PIPELINE]                                                [DYNAMIC MOTION PIPELINE]
                ┌───────────────────────────┐                                         ┌────────────────────────────┐
                │ Random Forest Classifier  │                                         │ Spatial Normalization      │
                │ (scikit-learn)            │                                         │ (Wrist Origin & Scale)     │
                └─────────────┬─────────────┘                                         └─────────────┬──────────────┘
                              │                                                                     │
                              │                                                                     ▼
                              │                                                       ┌────────────────────────────┐
                              │                                                       │ Temporal Normalization     │
                              │                                                       │ (Linear Interpolation T=60)│
                              │                                                       └─────────────┬──────────────┘
                              │                                                                     │
                              │                                                                     ▼
                              │                                                       ┌────────────────────────────┐
                              │                                                       │ Dynamic Time Warping (DTW) │
                              │                                                       │ (Sakoe-Chiba Constraint)   │
                              │                                                       └─────────────┬──────────────┘
                              │                                                                     │
                              └──────────────────────────────────┬──────────────────────────────────┘
                                                                 ▼
                                                  ┌──────────────────────────────┐
                                                  │ Text Label Output & gTTS     │
                                                  │ Audio Playback (.mp3)        │
                                                  └──────────────────────────────┘
```

### 3.1 Hand Landmark Extraction Pipeline
- **Input**: Live RGB webcam frames captured via OpenCV.
- **Engine**: Google MediaPipe Hands model (`mp.solutions.hands`).
- **Feature Extraction**: Identifies 21 key hand landmarks per frame. Each landmark has 3D coordinates $(x_i, y_i, z_i)$, creating a 63-dimensional feature vector per hand frame:
  $$V = [x_0, y_0, z_0, x_1, y_1, z_1, \dots, x_{20}, y_{20}, z_{20}]$$
- **Landmark Map**:
  - Wrist: Keypoint 0
  - Thumb: Keypoints 1–4
  - Index Finger: Keypoints 5–8
  - Middle Finger: Keypoints 9–12
  - Ring Finger: Keypoints 13–16
  - Pinky Finger: Keypoints 17–20

---

### 3.2 Static Sign Recognition Engine
1. **Data Storage**: Appends 63 landmark coordinates along with a class string label to `hand_landmarks.csv`.
2. **Model Training**:
   - Uses `scikit-learn`'s `RandomForestClassifier` with 100 decision trees (`n_estimators=100`).
   - Splits data using `train_test_split` (80% train, 20% test).
   - Serializes trained model into binary format `sign_language_model.pkl` via `pickle`.
3. **Live Prediction**:
   - Samples webcam frames over a 3-second window.
   - Evaluates each frame against the trained Random Forest classifier.
   - Applies majority voting over the predicted frame buffer to determine the final prediction result.

---

### 3.3 Dynamic Motion Sign Recognition Engine
For continuous gestures that involve movement over time (e.g., waving or sliding), static single-frame matching is insufficient. The dynamic engine processes spatial and temporal sequences:

1. **Spatial Normalization**:
   - **Translation Invariance**: Shifts hand landmarks relative to the wrist origin by subtracting wrist coordinates $(x_0, y_0, z_0)$ from all 21 points:
     $$P'_i = P_i - P_{wrist}$$
   - **Scale Invariance**: Normalizes landmark coordinates by dividing by palm length (Euclidean distance between wrist `0` and middle finger MCP joint `9`):
     $$S = \|P_{middle\_mcp} - P_{wrist}\|, \quad P''_i = \frac{P'_i}{S}$$

2. **Temporal Resampling (Uniform Sequence Length)**:
   - Gesture recordings vary in speed and frame count.
   - Uses 1D linear interpolation (`np.interp`) to resample all gesture sequences to a standard length of **$T = 60$ frames** ($60 \times 63$ matrix).

3. **Dynamic Time Warping (DTW) Distance Matching**:
   - Compares live candidate sequences against reference templates stored in `motion_signs/<sign_name>/*.npy`.
   - Employs **Dynamic Time Warping (DTW)** with Sakoe-Chiba band constraints to find optimal non-linear time alignment between sequences.
   - Selects the sign label with the minimum cumulative DTW Euclidean distance.

---

### 3.4 Text-to-Speech (TTS) Voice Synthesis
- **Audio Generation**: Uses `gTTS` (Google Text-to-Speech API) to convert predicted label text into spoken voice files (`output.mp3` or `motion_output.mp3`).
- **Cross-Platform Audio Playback**:
  - **Windows**: Executes `start <filename>`
  - **macOS**: Executes `afplay <filename>`
  - **Linux**: Executes `xdg-open <filename>`

---

## 4. Complete Technology Stack

| Layer | Technology | Usage & Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.8+ | Primary development language for core application and logic |
| **Computer Vision** | OpenCV (`opencv-python`) | Video stream capture (`VideoCapture`), window UI rendering, landmark visual overlays |
| **Hand Tracking** | Google MediaPipe (`mediapipe`) | CPU-optimized real-time 21 3D hand landmark detection |
| **Machine Learning** | Scikit-Learn (`scikit-learn`) | Random Forest Classifier (`RandomForestClassifier`), dataset partitioning, accuracy evaluation |
| **Numerical & Matrix Computing** | NumPy (`numpy`) | Multi-dimensional array operations, spatial normalization, linear interpolation resampling, DTW metric calculations |
| **Data Manipulation** | Pandas (`pandas`) | CSV loading, dataset cleaning, label validation, dataset inspection |
| **Model Serialization** | Pickle (`pickle`) | Saving and loading trained machine learning models (`sign_language_model.pkl`) |
| **Speech Synthesis** | gTTS (`gTTS`) | Text-to-Speech conversion of predicted sign strings into audio files |
| **Data Formats** | CSV, Excel, NumPy `.npy` | Dataset storage (`hand_landmarks.csv`, `hand_landmarks.xlsx`, `motion_signs/*.npy`) |
| **Environment / Cloud Execution** | Jupyter Notebook, Google Colab | Remote training, experimentation, WebRTC headless inference (`sign.ipynb`, `colab/`) |

---

## 5. Data Flow Architecture

```
[Webcam Feed]
     │
     ▼
[MediaPipe Hands Detection] ──> Extracts 21 Keypoints (63 float coordinates)
     │
     ├───► Mode A: Static Signs
     │        │
     │        ├── Data Mode: Append to hand_landmarks.csv
     │        └── Predict Mode: Random Forest Classifier ──> gTTS Audio Output
     │
     └───► Mode B: Dynamic Motion Signs
              │
              ├── Wrist Normalization & Palm Scale Normalization
              ├── Resample Sequence to T = 60 Frames
              ├── Save Reference Array (.npy) OR Compare with DTW
              └── Minimum Distance Match ──> gTTS Audio Output
```

---

## 6. Project Structure & File Organization

```
SignLanguageAI/
├── main.py                     # Main interactive CLI application & static pipeline runner
├── motion_signs.py             # Dynamic gesture recorder, normalizer, and DTW prediction engine
├── hand_landmarks.csv          # Dataset file storing static sign landmark coordinates & labels
├── hand_landmarks.xlsx         # Exported Excel version of static landmark dataset
├── sign_language_model.pkl     # Serialized Random Forest model artifact
├── PBL Report.docx             # Project Based Learning comprehensive report documentation
├── PRODUCT_DOCUMENTATION.md    # Product technical specification & architecture documentation
├── README.md                   # Quickstart guide and user manual
├── sign.ipynb                  # Jupyter Notebook for experimental prototyping
├── colab/                      # Modular scripts for Google Colab & remote training
│   ├── collect_data.py         # Headless landmark collection script
│   ├── train.py                # Model training script for Colab
│   └── realtime_inference.py   # Real-time WebRTC inference handler
└── motion_signs/               # Motion gesture reference template storage (.npy arrays)
    ├── hello/
    ├── goodbye/
    └── thank you/
```
