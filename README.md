# Sign Language to Voice - AI Recognition System 🤟🔊

An end-to-end, CPU-optimized Computer Vision and Machine Learning system that translates **Static Sign Language Poses** (letters, static hand signs) and **Dynamic Motion Signs** (e.g., Hello, Goodbye, Thank You, Yes, No) into real-time speech and text.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E)
![gTTS](https://img.shields.io/badge/gTTS-Text--to--Speech-red)

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture & Mathematics](#-system-architecture--mathematics)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage Guide & CLI Options](#-usage-guide--cli-options)
- [Static vs. Motion Signs](#-static-vs-motion-signs)
- [Google Colab Integration](#-google-colab-integration)
- [Dataset Format](#-dataset-format)
- [Troubleshooting](#-troubleshooting)

---

## 🌟 Overview

The **Sign Language to Voice AI System** bridges the communication gap for individuals who are deaf or hard-of-hearing. By utilizing **Google MediaPipe** for hand landmark extraction, the system extracts 21 3D hand landmarks ($21 \times 3 = 63$ spatial feature points per hand) in real-time.

It handles two primary modes of sign language:
1. **Static Hand Signs (Letters & Poses):** Recognized using a **Random Forest Classifier** trained on spatial landmark coordinates saved in `hand_landmarks.csv`.
2. **Dynamic Motion Signs (Gestures over time):** Recognized using **Dynamic Time Warping (DTW)** with Sakoe-Chiba band constraints, temporal resampling, wrist-origin normalization, and palm-scale normalization.

Recognized signs are automatically spoken aloud using Google Text-to-Speech (`gTTS`).

---

## ⚡ Key Features

- **CPU-Optimized Performance:** No GPU required! High FPS real-time tracking using MediaPipe and fast numerical methods in NumPy.
- **Dual Recognition Pipeline:**
  - **Static Model:** Fast scikit-learn Random Forest Classifier for static alphabet signs/poses.
  - **Motion Model:** Dynamic Time Warping (DTW) distance metric for trajectory gesture matching over time.
- **Text-to-Speech (gTTS) Feedback:** Automatically converts recognized predictions into audio (`output.mp3` or `motion_output.mp3`) and plays them natively across Windows, macOS, and Linux.
- **Robust Normalization:** Position-independent (wrist origin normalization) and scale-independent (palm-distance scaling) gesture representation.
- **Interactive CLI Interface:** User-friendly terminal menu for recording, training, testing, inspecting datasets, and managing data.
- **Cloud Ready:** Modular scripts in `colab/` for remote execution and model training.

---

## 📐 System Architecture & Mathematics

### 1. MediaPipe Hand Landmark Extraction
For every video frame, MediaPipe identifies 21 key points on the hand:
- Landmark 0: Wrist
- Landmarks 1-4: Thumb
- Landmarks 5-8: Index Finger
- Landmarks 9-12: Middle Finger
- Landmarks 13-16: Ring Finger
- Landmarks 17-20: Pinky Finger

Each landmark contains 3D spatial coordinates: $(x_i, y_i, z_i)$, forming a 63-element feature vector $V = [x_0, y_0, z_0, x_1, y_1, z_1, \dots, x_{20}, y_{20}, z_{20}]$.

```
         (8)  (12)  (16)  (20)
          |    |     |     |
         (7)  (11)  (15)  (19)
    (4)   |    |     |     |
     |   (6)  (10)  (14)  (18)
    (3)   \    |     |    /
     |    (5) (9)  (13) (17)
    (2)     \  |    /   /
     |       \ |   /   /
    (1)--------(0: Wrist)
```

### 2. Gesture Normalization (Motion Signs)
To make gesture trajectory matching invariant to camera distance and screen position:
1. **Translation Invariance:** Subtract the wrist $(x_0, y_0, z_0)$ coordinate from all 21 landmarks:
   $$P'_i = P_i - P_{wrist}$$
2. **Scale Invariance:** Divide coordinates by palm length (Euclidean distance between wrist `0` and middle finger MCP `9`):
   $$S = \|P_{middle\_mcp} - P_{wrist}\|$$
   $$P''i = \frac{P'_i}{S}$$
3. **Temporal Uniformity:** Resample gesture sequences of varying frame lengths to a fixed length of $T = 60$ frames using linear interpolation:
   $$V_{resampled} \in \mathbb{R}^{60 \times 63}$$

### 3. Dynamic Time Warping (DTW)
DTW aligns live motion sequences against reference templates stored in `motion_signs/<sign_name>/`.
To accelerate DTW execution on CPU, key joint indices (fingertips + wrist + MCP joints) are extracted, and a Sakoe-Chiba band constraint window $w$ limits distance matrix evaluation:

$$\text{DTW}(A, B) = \frac{\min_{\pi} \sum_{(i,j) \in \pi} \|A_i - B_j\|}{|A| + |B|}$$

---

## 📁 Project Structure

```
SignLanguageAI/
├── main.py                    # Interactive main CLI application (Menu 0-9)
├── motion_signs.py            # Dynamic Motion Sign recorder, DTW predictor & normalizer
├── hand_landmarks.csv         # CSV dataset containing static pose landmarks & labels
├── hand_landmarks.xlsx        # Excel format export of static landmarks dataset
├── sign_language_model.pkl    # Serialized scikit-learn Random Forest model
├── PBL Report.docx            # Detailed Project Based Learning report document
├── sign.ipynb                 # Jupyter Notebook for experimentations & model prototyping
├── colab/                     # Standalone scripts for Google Colab / server execution
│   ├── collect_data.py        # Data collection module for Colab
│   ├── train.py               # Model training script
│   └── realtime_inference.py  # Realtime inference script
└── motion_signs/              # Storage directory for numpy (.npy) motion gesture sequences
    ├── hello/
    ├── goodbye/
    └── thank you/
```

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Webcam connected to your computer

### Setup Virtual Environment & Dependencies

1. **Clone or Navigate to Repository:**
   ```bash
   cd "path/to/SignLanguageAI"
   ```

2. **Create and Activate Virtual Environment:**
   - **Windows:**
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - **macOS / Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install opencv-python mediapipe scikit-learn pandas numpy gTTS
   ```

---

## 🚀 Usage Guide & CLI Options

Run the main application by executing:
```bash
python main.py
```

### Main Menu Interface
When executed, `main.py` presents the following menu:

```text
=======================================================
   SIGN LANGUAGE TO VOICE - AI RECOGNITION SYSTEM
=======================================================

  ─── Static Signs (Letters/Poses) ───
  1. Record New Static Sign (5 Seconds)
  2. Train Static AI Model
  3. Predict Static Sign & Play Voice

  ─── Motion Signs (Hello, Goodbye, etc.) ───
  4. Record New Motion Sign (10 Seconds)
  5. Predict Motion Sign & Play Voice
  6. Quick Record (Pick from list)

  ─── Utilities ───
  7. View All Recorded Signs
  8. Reset/Delete Static Dataset
  9. Reset/Delete Motion Signs Data
  0. Exit
=======================================================
```

---

### Step-by-Step CLI Walkthrough

#### 1. Static Signs (Options 1 – 3)
- **Option 1: Record New Static Sign (5 Seconds)**
  - Enter the sign name (e.g., `A`, `B`, `ThumbsUp`).
  - Press `'s'` when ready. The camera records hand landmarks for 5 seconds and appends them to `hand_landmarks.csv`.
- **Option 2: Train Static AI Model**
  - Trains a `RandomForestClassifier` on `hand_landmarks.csv`.
  - Saves the trained model to `sign_language_model.pkl` and outputs accuracy.
- **Option 3: Predict Static Sign & Play Voice**
  - Opens webcam feed for 3 seconds, detects your sign in real-time, displays prediction text on screen, and generates spoken audio (`output.mp3`).

#### 2. Motion Signs (Options 4 – 6)
- **Option 4: Record New Motion Sign (10 Seconds)**
  - Enter a gesture name (e.g., `Hello`, `Goodbye`).
  - Follow the 3-2-1 countdown, perform the gesture over 10 seconds. Saved as `.npy` array sequence under `motion_signs/<sign_name>/`.
- **Option 5: Predict Motion Sign & Play Voice**
  - Captures a 5-second video sequence, normalizes spatial & temporal features, matches sequence against reference templates using DTW, and plays `motion_output.mp3`.
- **Option 6: Quick Record (Batch List)**
  - Prompts from a predefined list of popular motion signs for fast batch dataset collection.

#### 3. Utilities (Options 7 – 9)
- **Option 7: View All Recorded Signs**
  - Summarizes recorded static signs (frame count per label) and motion sign directories.
- **Option 8 & 9: Reset / Delete Datasets**
  - Allows clean reset of static dataset (`hand_landmarks.csv`) or motion signs dataset directory (`motion_signs/`).

---

## 🔄 Static vs. Motion Signs

| Feature | Static Signs | Motion Signs |
| :--- | :--- | :--- |
| **Type** | Single image frame pose (e.g., letters A-Z, thumbs up) | Time-series movement (e.g., waving hello, swiping) |
| **Duration** | Instant / 3-5 sec observation | 5-10 sec trajectory capture |
| **Algorithm** | Random Forest Classifier (`scikit-learn`) | Dynamic Time Warping (`NumPy` DTW) |
| **Storage** | `hand_landmarks.csv` | `motion_signs/*.npy` |
| **Resampling** | Not required | Resampled to 60 uniform frames |

---

## ☁️ Google Colab Integration

For running experiments or training models in Google Colab, use the scripts inside the `colab/` directory:
1. `colab/collect_data.py`: Landmark collection tailored for headless/Colab streaming environment.
2. `colab/train.py`: Model training script saving outputs.
3. `colab/realtime_inference.py`: WebRTC / image frame inference runner for notebooks.

---

## 📊 Dataset Format

### `hand_landmarks.csv` Structure
```csv
label,x0,y0,z0,x1,y1,z1,...,x20,y20,z20
Hello,0.521,0.643,-0.002,0.489,0.612,-0.015,...
Hello,0.525,0.640,-0.001,0.491,0.610,-0.014,...
Thanks,0.410,0.520,-0.005,0.395,0.490,-0.020,...
```

### `motion_signs/` Directory Structure
```
motion_signs/
├── hello/
│   ├── rec_1.npy
│   └── rec_2.npy
├── goodbye/
│   ├── rec_1.npy
│   └── rec_2.npy
```

---

## 🛠️ Troubleshooting

- **Camera Not Opening (`[ERROR] Camera nahi khul raha!`):**
  - Ensure no other app (Zoom, Teams, Browser) is accessing your webcam.
  - On macOS, grant camera permissions to your terminal application.
- **Audio Not Playing:**
  - Verify system volume is on.
  - Windows uses `start <file>`, macOS uses `afplay <file>`, Linux uses `xdg-open <file>`. Ensure your system default media player supports `.mp3`.
- **Corrupt CSV Warning (`[CRITICAL ERROR] CSV FILE CORRUPT`):**
  - Use Option 8 to reset the static dataset and record fresh samples using Option 1.

---

## 📄 License
This project is open-source under the MIT License. Feel free to modify and adapt for research and educational purposes.
