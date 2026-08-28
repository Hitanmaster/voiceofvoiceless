# 🎫 TICKET-02 — Data Engineer
## Feature: MediaPipe Holistic Dataset Collector & Training Data Collection

**Assigned to:** Person 2  
**Files owned:** `dataset_collector.py`, `dataset/` folder structure  
**Depends on:** Nothing (start on Day 1)  
**Blocks:** TICKET-01 (Person 1 cannot train until dataset is on Drive)  
**Due:** End of Week 1 (Day 7) — dataset uploaded to Google Drive

---

## 📋 Implementation Steps

### Part A — Dataset Collector Script (`dataset_collector.py`)

- [ ] Create file `dataset_collector.py` in the project root
- [ ] Import: `cv2`, `mediapipe as mp`, `numpy as np`, `os`, `time`
- [ ] Initialize `mp.solutions.holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.5)`
- [ ] Write `extract_features(results)` function that:
  - [ ] Extracts **Left Hand**: 21 landmarks × 3 coords = 63 values (or zeros if not detected)
  - [ ] Extracts **Right Hand**: 21 landmarks × 3 coords = 63 values (or zeros if not detected)
  - [ ] Extracts **Upper Pose** (landmarks 11, 12, 13, 14 = left shoulder, right shoulder, left elbow, right elbow): 4 × 3 = 12 values
  - [ ] Concatenates all into a single flat array of shape `(138,)`
  - [ ] Returns zeros for any missing body part — do NOT return `None`
    - > **Note:** Pose landmarks 11–14 are the upper body. Access via `results.pose_landmarks.landmark[11]`, etc. If `pose_landmarks` is `None`, use 12 zeros.
- [ ] Write `record_sequence(sign_name, num_frames=60)` function that:
  - [ ] Opens webcam with `cv2.VideoCapture(0)`
  - [ ] Shows **3-second countdown** on screen before recording starts (`cv2.putText`)
  - [ ] Records exactly 60 frames, showing frame counter: `"Recording: Frame 23/60"`
  - [ ] Appends each frame's 138-feature vector to a list
  - [ ] Saves list as numpy array: `np.save(filepath, np.array(sequence))` — shape must be `(60, 138)`
  - [ ] Generates filename using timestamp: `f"{sign_name}_{int(time.time())}.npy"`
- [ ] Write `main()` loop:
  - [ ] Prompt user: `"Enter sign name (or 'quit' to exit): "`
  - [ ] Create directory `dataset/<sign_name>/` if it doesn't exist
  - [ ] Ask: `"How many sequences? (default 5): "`
  - [ ] Loop and record N sequences, saving each one
  - [ ] Print count of existing sequences after each save: `"Saved. Total for 'hello': 12"`

---

### Part B — Data Quality Validation

- [ ] After recording, write a `validate_dataset()` function that:
  - [ ] Scans all files in `dataset/*/` directories
  - [ ] Loads each `.npy` file and checks shape == `(60, 138)`
  - [ ] Prints a report: class name, sequence count, any malformed files
  - > **Note:** A file is malformed if its shape is NOT `(60, 138)`. Delete and re-record any malformed files.
- [ ] Run validation and fix all issues before uploading to Drive

---

### Part C — Mass Data Collection (Days 5–7)

Follow this collection order (most important classes first):

**Priority 1 — CRITICAL (collect 60+ sequences each, all 4 members record):**
- [ ] `idle` — Resting hands in lap, touching face, random arm movement, sitting still

**Priority 2 — HIGH (collect 40+ sequences each):**
- [ ] `hello` — Wave hand side to side
- [ ] `thank_you` — Flat hand from chin moving forward
- [ ] `goodbye` — Wave hand
- [ ] `please` — Circular motion on chest
- [ ] `help` — Thumb up placed on open palm, move upward
- [ ] `yes` — Fist nodding motion
- [ ] `no` — Index and middle finger wagging side to side
- [ ] `i_love_you` — ILY handshape (index, pinky, thumb extended)

**Priority 3 — MEDIUM (collect 30+ sequences each, split A–M and N–Z between pairs):**
- [ ] Static alphabets A through Z (one static pose held for 60 frames)
  - > **Note:** For static signs, just hold the hand pose still. The model learns from the consistent spatial pattern across all 60 frames.

**Collection Rules:**
- [ ] Each member records in their own lighting conditions (mix bright + dim)
- [ ] Vary distance from camera (close, medium, far)
- [ ] Vary hand angle slightly between sequences
- [ ] Do NOT record more than 10 sequences in a row without taking a break (hand fatigue affects quality)

---

### Part D — Google Drive Upload

- [ ] Zip the entire `dataset/` folder: `dataset.zip`
- [ ] Upload `dataset.zip` to the shared Google Drive folder: `SignLanguageAI/`
- [ ] Unzip on Drive OR share the folder path with Person 1 directly
- [ ] Confirm with Person 1 that they can access the dataset from their Colab notebook
- [ ] Commit `dataset_collector.py` to GitHub (do NOT commit the `.npy` data files — add `dataset/` to `.gitignore`)

---

## ✅ Test Checklist

- [ ] Run `dataset_collector.py` — webcam opens without error
- [ ] Countdown appears on screen for 3 seconds before recording
- [ ] Frame counter updates in real-time on screen during recording
- [ ] After recording, file `dataset/hello/hello_<timestamp>.npy` exists
- [ ] File shape is exactly `(60, 138)`: `np.load('...').shape == (60, 138)`
- [ ] Validation script shows 0 malformed files in `dataset/`
- [ ] `idle` class has at least 60 sequences before upload
- [ ] All other sign classes have at least 30 sequences before upload
- [ ] Google Drive folder is accessible by Person 1 in Colab

---

## 🔴 Important Notes

> **Most Critical Class:** `idle` is the most important class in the entire dataset. If it has too few samples, the model will produce false positives constantly. Aim for 60+ sequences of idle, recorded by all 4 team members.

> **Zero-Padding is Expected:** It is perfectly fine if MediaPipe misses one hand in some frames. The `extract_features()` function handles this by returning zeros — do not re-record sequences just because one hand was occasionally missed.

> **Do NOT commit `.npy` files to GitHub.** Add `dataset/` to `.gitignore`. Only the script goes to GitHub. The data goes to Google Drive.

> **Coordinate with Person 1:** Let Person 1 know the exact Google Drive path where `dataset/` is stored so they can set `DATASET_PATH` in the Colab notebook correctly.
