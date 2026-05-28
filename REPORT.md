# Project Report: Sign Language to Voice - AI Recognition System

## 1. Project Overview

The **Sign Language to Voice AI Recognition System** is a computer vision-based application designed to bridge the communication gap by translating sign language gestures into spoken audio. The system operates locally and doesn't require a GPU, making it highly accessible.

It uniquely supports both:
*   **Static Signs:** Gestures that involve a fixed hand pose (e.g., letters, numbers, or specific words).
*   **Motion Signs:** Dynamic gestures that involve movement over time (e.g., "Hello," "Goodbye," "Thank You").

The application provides an interactive Command Line Interface (CLI) where users can seamlessly record their own datasets, train models, and perform live predictions. When a sign is recognized, the system uses Text-to-Speech (TTS) to vocalize the translated word.

## 2. Technology Stack

*   **Python:** Core programming language.
*   **OpenCV (`cv2`):** Used for accessing the webcam, capturing video frames, flipping/mirroring, and rendering the user interface overlay (text, progress bars, etc.) directly on the video feed.
*   **MediaPipe (`mediapipe`):** Google's open-source framework used for real-time hand tracking. It extracts 21 3D landmarks (x, y, z coordinates) per hand.
*   **Scikit-Learn (`sklearn`):** Used for machine learning on static signs. Specifically, it employs a `RandomForestClassifier` to map 63 structural data points (21 landmarks * 3 axes) to specific static sign labels.
*   **Pandas (`pandas`) & NumPy (`numpy`):** Used for dataset manipulation. Pandas handles the CSV data for static signs, while NumPy is heavily used for the normalization and Dynamic Time Warping (DTW) of multi-frame sequences in motion signs.
*   **gTTS (Google Text-to-Speech):** Converts the predicted text labels into spoken audio (`.mp3` files).
*   **Pickle (`pickle`):** Used for serializing and saving the trained Random Forest model.

## 3. Code Architecture

The codebase is elegantly divided into two main components:

### `main.py`
This is the entry point of the application and manages the core loop, menu interface, and **Static Sign** logic.

*   **Data Collection (`record_data`):** Opens the webcam for 5 seconds. In each frame, it uses MediaPipe to extract 21 hand landmarks and saves the (x, y, z) coordinates into `hand_landmarks.csv` along with the provided label.
*   **Model Training (`train_model`):** Reads `hand_landmarks.csv` using Pandas. It splits the data (80% train, 20% test) and trains a `RandomForestClassifier`. The resulting model is evaluated for accuracy and saved as `sign_language_model.pkl`.
*   **Live Prediction (`live_prediction`):** Captures 3 seconds of webcam feed, passes the hand landmarks to the trained Random Forest model, and uses the mode (most frequent prediction) as the final output. The prediction is then synthesized to speech using `gTTS`.
*   **Utilities:** Functions for playing audio OS-agnostically (`play_audio`) and managing the CSV dataset.

### `motion_signs.py`
This module is dedicated entirely to **Dynamic/Motion Signs**. Because motion signs involve time and movement, standard classification isn't sufficient.

*   **Normalization (`normalize_frame`, `normalize_sequence`):**
    *   *Spatial Normalization:* Subtracts the wrist coordinates to make the wrist the origin (0,0,0) and scales the hand by the distance between the wrist and the middle finger MCP joint. This makes the data position and scale invariant (works regardless of how far the hand is from the camera).
    *   *Temporal Normalization:* Uses linear interpolation to resample all recorded sequences to exactly 60 frames (`TARGET_FRAMES`).
*   **Data Storage:** Motion signs are recorded as 10-second clips. The normalized sequences are saved as `.npy` NumPy arrays inside `motion_signs/<sign_name>/`.
*   **Dynamic Time Warping (`dtw_distance`, `dtw_distance_fast`):** The core algorithm for matching motion signs. DTW calculates the distance between two time-series sequences. It handles variations in speed (e.g., if a user performs the "Hello" sign faster than the recorded template, DTW can still match them). `dtw_distance_fast` utilizes a Sakoe-Chiba band constraint to optimize performance.
*   **Prediction (`predict_motion_sign`):** Captures a 5-second live sequence, normalizes it, and calculates the DTW distance against all saved `.npy` templates. The sign with the lowest distance (under a `CONFIDENCE_THRESHOLD`) is selected and spoken.

## 4. Setup and Installation Instructions

To set up the project locally for development or contribution, follow these steps:

### Prerequisites
*   Python 3.8+ installed.
*   A working webcam.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv

    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    You will need to install the required libraries. If a `requirements.txt` is not provided, install them manually:
    ```bash
    pip install opencv-python mediapipe scikit-learn pandas numpy gtts
    ```

4.  **Run the Application:**
    ```bash
    python main.py
    ```

### First-Time Usage Flow
*   When starting fresh, the system has no data.
*   Select **Option 1** to record at least two different static signs (e.g., 'A' and 'B').
*   Select **Option 2** to train the Random Forest model.
*   Select **Option 3** to test static sign prediction.
*   Select **Option 4** or **6** to record templates for motion signs (record at least 3 templates per sign for accuracy).
*   Select **Option 5** to test motion sign prediction.

## 5. Limitations and Areas for Improvement

While the current implementation is functional and clever in its approach, there are several limitations that contributors should be aware of:

*   **Reliance on Audio Output Hacks:** The `play_audio` function relies on OS-level commands (`start`, `afplay`, `xdg-open`). This can be unreliable across different Linux distributions or restricted environments. Using a dedicated audio library like `pygame` or `playsound` would be more robust.
*   **Single Hand Assumption (Static Signs):** The static sign model (`main.py`) only extracts landmarks from the *first* detected hand, limiting the ability to recognize two-handed static signs.
*   **Data Quality Vulnerability (Static Signs):** The static model trains on *every* frame captured during the 5-second window. If the user moves their hand to their side halfway through, garbage data is added to the CSV, which can severely degrade model accuracy.
*   **Blocking UI & Sleep Statements:** The application uses `time.time()` loops which block execution. The transition between terminal input and OpenCV windows can feel clunky.
*   **CPU Intensity of DTW:** Dynamic Time Warping, even with the Sakoe-Chiba constraint, is mathematically expensive. Comparing a live feed against hundreds of `.npy` templates will cause significant lag. Transitioning motion signs to an LSTM (Long Short-Term Memory) neural network or an Action Recognition model would make it scalable.
*   **No Background Audio Handling:** Every time `gTTS` saves `output.mp3` or `motion_output.mp3`, it overwrites the file. If predictions happen too quickly, file locking errors or audio cutoffs can occur.
*   **Hardcoded Thresholds:** Values like `CONFIDENCE_THRESHOLD = 15.0` in `motion_signs.py` are hardcoded. These might need to be dynamic or calibrated based on the user.