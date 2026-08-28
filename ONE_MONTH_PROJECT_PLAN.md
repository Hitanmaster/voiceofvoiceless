# 🗓️ 1-Month Final Year Project Plan — Sign Language AI System
## Team of 4 | Hybrid Strategy: Train on Google Colab GPU → Run Real-Time Locally

---

> **Project Goal:**  
> Build a complete, working **real-time sign language to speech converter** using a unified BiLSTM + Attention deep learning model trained on Google Colab and deployed locally for zero-latency live inference.

---

## 👥 Team Roles & Ownership

| Member | Role | Primary Files & Responsibilities |
| :--- | :--- | :--- |
| **Person 1** | 🧠 ML Engineer / Team Lead | `model.py`, `train_model.py`, Colab notebooks, Google Drive setup, model evaluation, Confusion Matrix, F1-Score |
| **Person 2** | 📦 Data Engineer | `dataset_collector.py`, `collect_dataset.py`, data organization, `.npy` recording, idle class collection, data quality review |
| **Person 3** | ⚙️ Backend / Integration Engineer | `unified_detector.py`, `main.py`, `motion_signs.py`, `inference_engine.py`, RF+DTW fallback integration, TTS engine |
| **Person 4** | 🖥️ UI / Documentation Lead | `app.py` (Gradio), `README.md`, `content.md` updates, thesis architecture diagrams, system demo video, final presentation slides |

---

## 🧰 What You Need Before Starting

### Hardware
- [ ] Any laptop/PC with webcam (for data collection and local inference)
- [ ] Stable internet connection (for Google Colab training sessions only)

### Accounts & Cloud Tools
- [ ] **Google Account** with Google Drive (minimum 5 GB free space for dataset)
- [ ] **Google Colab** (free tier is sufficient — T4 GPU available)
- [ ] **GitHub** repository shared among all 4 team members (for code collaboration)

### Local Software to Install
```bash
# Python 3.9 or 3.10 recommended
pip install opencv-python mediapipe torch torchvision
pip install scikit-learn pandas numpy
pip install gtts pyttsx3 playsound
pip install gradio matplotlib seaborn
```

### Google Colab Packages (run in first notebook cell)
```python
!pip install torch torchvision mediapipe opencv-python-headless
!pip install scikit-learn matplotlib seaborn
from google.colab import drive
drive.mount('/content/drive')
```

### File Structure to Set Up on Day 1
```text
SignLanguageAI/
├── dataset/                  # [NEW] Holistic training sequences
│   ├── hello/                # 30+ .npy files per sign class
│   ├── thank_you/
│   ├── goodbye/
│   ├── please/
│   ├── help/
│   ├── yes/
│   ├── no/
│   ├── A/ B/ C/ ... Z/       # Static alphabet signs
│   └── idle/                 # Background/resting pose class
├── colab/
│   ├── train.ipynb           # Training notebook
│   └── evaluate.ipynb        # Evaluation notebook
├── model.py                  # BiLSTM + Attention architecture
├── train_model.py            # Training pipeline script
├── dataset_collector.py      # 138-feature Holistic recorder
├── unified_detector.py       # Real-time sliding window engine
├── app.py                    # Gradio 5-tab web interface
├── unified_sign_model.pt     # [OUTPUT] Trained model checkpoint
└── content.md                # Master documentation
```

---

## 📅 Week-by-Week Plan

---

### 📅 WEEK 1 (Days 1–7): Environment Setup + Data Collection

**Goal:** Get all tools installed, GitHub repo shared, and collect clean dataset for all sign classes.

#### Day 1–2: Project Setup (ALL MEMBERS)
| Task | Owner | Done? |
| :--- | :--- | :--- |
| Create shared GitHub repo, all 4 members clone it | Person 1 | [ ] |
| Set up Python virtual environment (`venv/`) locally | All | [ ] |
| Install all local dependencies (see above) | All | [ ] |
| Create Google Drive folder: `SignLanguageAI/dataset/` | Person 1 | [ ] |
| Share Google Drive folder access with all members | Person 1 | [ ] |
| Test Colab GPU access: Runtime → Change runtime → T4 GPU | Person 1 | [ ] |
| Review and understand existing codebase (`main.py`, `motion_signs.py`, `inference_engine.py`) | All | [ ] |

#### Day 3–4: Build Dataset Collector (Person 2 leads, Person 3 assists)
| Task | Owner | Done? |
| :--- | :--- | :--- |
| Write `dataset_collector.py` using MediaPipe Holistic (138 features: Left + Right + Pose) | Person 2 | [ ] |
| Test zero-padding for missing hand landmarks | Person 2 | [ ] |
| Add OpenCV countdown overlay (3-2-1 before recording each sequence) | Person 2 | [ ] |
| Add visual frame counter showing progress (e.g., "Frame 14/60") | Person 2 | [ ] |
| Save sequences as `dataset/<sign_name>/<timestamp>.npy` (shape: 60, 138) | Person 2 | [ ] |
| Test with 3 sample classes: `hello`, `A`, `idle` | Person 2 + Person 3 | [ ] |

#### Day 5–7: Mass Data Collection (Person 2 leads, ALL contribute)
**Target: 30+ sequences per class, 60 frames each**

| Sign Classes to Collect | Type | Priority |
| :--- | :--- | :--- |
| `idle` (resting hands, random head scratching, walking) | Background | CRITICAL |
| `hello`, `goodbye`, `thank_you`, `please`, `help` | Dynamic gestures | HIGH |
| `yes`, `no`, `i_love_you` | Dynamic gestures | HIGH |
| `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `J` | Static alphabet | MEDIUM |
| `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T` | Static alphabet | MEDIUM |
| `U`, `V`, `W`, `X`, `Y`, `Z` | Static alphabet | MEDIUM |

**Collection Tips:**
- Record in different lighting conditions (bright room, dim room)
- Record from slightly different distances and angles
- Each team member records their own sequences (adds person-variety = better generalization)
- Minimum 30 sequences per class; 50+ is ideal

| Task | Owner | Done? |
| :--- | :--- | :--- |
| Collect `idle` class (minimum 60 sequences, most important class) | All 4 members | [ ] |
| Collect 5 dynamic gestures (30 seq each = 150 files) | Person 2 + Person 4 | [ ] |
| Collect A–M static alphabets (30 seq each) | Person 1 + Person 3 | [ ] |
| Collect N–Z static alphabets (30 seq each) | Person 2 + Person 4 | [ ] |
| Verify all `.npy` files have shape `(60, 138)` | Person 2 | [ ] |
| Upload complete `dataset/` folder to Google Drive | Person 2 | [ ] |
| Commit `dataset_collector.py` to GitHub | Person 2 | [ ] |

**Week 1 Deliverable:** `dataset/` folder on Google Drive with 30+ sequences per class for all target signs.

---

### 📅 WEEK 2 (Days 8–14): Model Design + Colab GPU Training

**Goal:** Build the BiLSTM + Attention model, train it on Colab GPU, evaluate accuracy, and download `unified_sign_model.pt`.

#### Day 8–9: PyTorch Model Architecture (Person 1 leads)
| Task | Owner | Done? |
| :--- | :--- | :--- |
| Write `model.py`: BiLSTM + Self-Attention class | Person 1 | [ ] |
| Test model forward pass with dummy tensor `(4, 60, 138)` | Person 1 | [ ] |
| Write `train_model.py` with data loading from `dataset/` directory | Person 1 | [ ] |
| Implement 5 augmentation techniques in training loop | Person 1 | [ ] |
| Implement 80/20 train/validation split | Person 1 | [ ] |
| Add AdamW optimizer + CosineAnnealingLR scheduler | Person 1 | [ ] |
| Add CrossEntropyLoss with class weights (for idle class imbalance) | Person 1 | [ ] |

**Model Architecture Reference:**
```python
class UnifiedSignModel(nn.Module):
    # Input:  (batch, 60, 138)
    # Layer 1: Linear(138 -> 128) + BatchNorm1d + ReLU + Dropout(0.3)
    # Layer 2: BiLSTM(128, hidden=128, layers=2, dropout=0.3)
    # Layer 3: Self-Attention over 60 time steps -> context vector (256,)
    # Layer 4: Linear(256 -> 64) + ReLU + Dropout(0.3)
    # Output:  Linear(64 -> num_classes) [Softmax at inference]
```

#### Day 10–11: Colab Training Notebook (Person 1 leads, Person 2 assists)
| Task | Owner | Done? |
| :--- | :--- | :--- |
| Create `colab/train.ipynb` Colab notebook | Person 1 | [ ] |
| Mount Google Drive in notebook | Person 1 | [ ] |
| Load dataset from Drive into PyTorch DataLoader | Person 1 | [ ] |
| Run training on T4 GPU (50–100 epochs, monitor val accuracy) | Person 1 | [ ] |
| Save best checkpoint: `unified_sign_model.pt` to Drive | Person 1 | [ ] |
| Verify training loss decreasing and val accuracy > 85% | Person 1 | [ ] |

#### Day 12–13: Model Evaluation (Person 1 leads, Person 4 assists for charts)
| Task | Owner | Done? |
| :--- | :--- | :--- |
| Create `colab/evaluate.ipynb` Colab notebook | Person 1 | [ ] |
| Load trained model, run on test set | Person 1 | [ ] |
| Generate Confusion Matrix (matplotlib/seaborn) | Person 1 + Person 4 | [ ] |
| Generate Classification Report (Precision, Recall, F1 per class) | Person 1 | [ ] |
| Plot training loss and accuracy curves | Person 1 + Person 4 | [ ] |
| Save all charts as PNG files for thesis | Person 4 | [ ] |
| Download `unified_sign_model.pt` to local `SignLanguageAI/` folder | Person 1 | [ ] |
| Commit `model.py`, `train_model.py`, notebooks to GitHub | Person 1 | [ ] |

#### Day 14: Accuracy Review & Retraining if Needed
| Task | Owner | Done? |
| :--- | :--- | :--- |
| If validation accuracy < 85%: collect more data for weak classes | Person 2 | [ ] |
| If certain classes confused: review and re-record those classes | Person 2 | [ ] |
| Re-run training with updated dataset if needed | Person 1 | [ ] |

**Week 2 Deliverable:** `unified_sign_model.pt` downloaded locally with val accuracy > 85%, confusion matrix and training charts saved.

---

### 📅 WEEK 3 (Days 15–21): Real-Time Inference Engine + Web Interface

**Goal:** Build the live detection engine that runs the downloaded model locally at 30 FPS, and the Gradio web UI.

#### Day 15–16: Unified Real-Time Detector (Person 3 leads)
| Task | Owner | Done? |
| :--- | :--- | :--- |
| Write `unified_detector.py` with `deque(maxlen=60)` sliding window | Person 3 | [ ] |
| Integrate MediaPipe Holistic (138 features, zero-padding fallback) | Person 3 | [ ] |
| Load `unified_sign_model.pt` with PyTorch `torch.load()` | Person 3 | [ ] |
| Add RF fallback: if `.pt` not found, load `sign_language_model.pkl` | Person 3 | [ ] |
| Implement confidence threshold check (>= 80%) | Person 3 | [ ] |
| Implement 3-consecutive-evaluation debouncing before trigger | Person 3 | [ ] |
| Implement 2.0s cooldown timer after each speech output | Person 3 | [ ] |
| Reject predictions where class == `"idle"` | Person 3 | [ ] |

#### Day 17–18: OpenCV Overlay + TTS Integration (Person 3 leads, Person 2 assists)
| Task | Owner | Done? |
| :--- | :--- | :--- |
| Add OpenCV overlay: predicted sign + confidence % text on video frame | Person 3 | [ ] |
| Add visual confidence progress bar drawn on frame | Person 3 | [ ] |
| Add cooldown countdown indicator ("Next in: 1.2s") | Person 3 | [ ] |
| Add "No hand detected" warning when no landmarks found | Person 3 | [ ] |
| Integrate TTS: `pyttsx3` for offline speech, `gTTS` as fallback | Person 3 + Person 2 | [ ] |
| Test full pipeline: webcam -> detection -> speech output | Person 3 | [ ] |
| Update `main.py`: add Option 10 (Unified Real-Time Detection) | Person 3 | [ ] |

#### Day 19–20: Gradio Web Interface (Person 4 leads)
| Task | Owner | Done? |
| :--- | :--- | :--- |
| Write `app.py` with Gradio 5-tab layout | Person 4 | [ ] |
| Tab 1 — Live Detection: webcam input, prediction output, confidence display | Person 4 | [ ] |
| Tab 2 — Record Sign: name input, countdown, save .npy | Person 4 | [ ] |
| Tab 3 — Train Model: trigger training, show progress output | Person 4 | [ ] |
| Tab 4 — Dataset Manager: table of classes + sequence counts | Person 4 | [ ] |
| Tab 5 — Settings: confidence slider, cooldown slider, mode toggle | Person 4 | [ ] |
| Update `main.py`: add Option 11 (Launch Gradio Web UI) | Person 3 | [ ] |
| Test Gradio app runs locally at `http://127.0.0.1:7860` | Person 4 | [ ] |

#### Day 21: Integration Testing (ALL MEMBERS)
| Task | Owner | Done? |
| :--- | :--- | :--- |
| Run full system end-to-end: data collection -> training -> local inference | All | [ ] |
| Test all 4 team members' hand sizes and signing styles | All | [ ] |
| Identify and fix any false positives or missed detections | Person 3 | [ ] |
| Verify idle class correctly suppresses non-signing frames | All | [ ] |
| Record a short 30-second demo video of the system working | Person 4 | [ ] |

**Week 3 Deliverable:** Fully working real-time sign language detection running locally. Gradio web UI running at localhost. Demo video recorded.

---

### 📅 WEEK 4 (Days 22–30): Testing, Polish & Final Documentation

**Goal:** Harden the system, fix edge cases, produce all thesis materials, and finalize the presentation.

#### Day 22–24: System Hardening & Edge Case Fixes (Person 3 + Person 2)
| Task | Owner | Done? |
| :--- | :--- | :--- |
| Test with poor lighting conditions | Person 2 | [ ] |
| Test with occluded hands (only 1 hand visible) | Person 2 | [ ] |
| Test with fast vs. slow signing speeds | All | [ ] |
| Test with different skin tones and hand sizes | All | [ ] |
| Adjust confidence threshold if too many false positives | Person 3 | [ ] |
| Profile FPS: ensure >= 20 FPS on local machine | Person 3 | [ ] |
| Fix any crashes or memory leaks in detection loop | Person 3 | [ ] |

#### Day 25–26: Thesis Documentation (Person 4 leads, ALL contribute)
| Task | Owner | Done? |
| :--- | :--- | :--- |
| Write system architecture section with block diagram | Person 4 | [ ] |
| Write comparative analysis: Legacy (RF+DTW) vs. Unified (BiLSTM+Attention) | Person 1 + Person 4 | [ ] |
| Write data collection methodology section | Person 2 + Person 4 | [ ] |
| Insert confusion matrix and training curves into thesis | Person 4 | [ ] |
| Write results & discussion section (accuracy, FPS, limitations) | Person 1 + Person 4 | [ ] |
| Update `README.md` with complete setup and run instructions | Person 4 | [ ] |
| Update `content.md` with final system summary entry | Person 4 | [ ] |

#### Day 27–28: Final Presentation Preparation (ALL)
| Task | Owner | Done? |
| :--- | :--- | :--- |
| Create 15–20 slide presentation deck | Person 4 | [ ] |
| Slide 1: Project title, team names | Person 4 | [ ] |
| Slide 2–3: Problem statement and motivation | Person 4 | [ ] |
| Slide 4–5: System architecture diagram | Person 1 + Person 4 | [ ] |
| Slide 6–7: Dataset collection methodology | Person 2 | [ ] |
| Slide 8–10: Model architecture + math formulation | Person 1 | [ ] |
| Slide 11–12: Training results (accuracy, confusion matrix) | Person 1 | [ ] |
| Slide 13–14: Live demo screenshots / video | Person 4 | [ ] |
| Slide 15: Limitations and future work | All | [ ] |
| Rehearse live demo: make sure model loaded, webcam working | All | [ ] |

#### Day 29–30: Final Review & Submission
| Task | Owner | Done? |
| :--- | :--- | :--- |
| Final code review: remove debug prints, clean up comments | All | [ ] |
| Final GitHub commit with all files | Person 1 | [ ] |
| Tag GitHub release: `v1.0-final-year-project` | Person 1 | [ ] |
| Final dry-run of live demo on presentation laptop | All | [ ] |
| Submit thesis document | Person 4 | [ ] |

**Week 4 Deliverable:** Complete, polished system. All thesis materials. GitHub release tagged. Presentation rehearsed.

---

## 📊 Summary Timeline at a Glance

```text
WEEK 1 (Days 1–7):   Setup + Data Collection
  Day 1–2: Environment setup, GitHub, Google Drive
  Day 3–4: Build dataset_collector.py (Holistic 138-feature recorder)
  Day 5–7: Mass data collection (all 4 members record sequences)

WEEK 2 (Days 8–14):  Model Design + GPU Training
  Day 8–9:  Write model.py + train_model.py
  Day 10–11: Train on Colab T4 GPU, save unified_sign_model.pt
  Day 12–13: Evaluate: confusion matrix, F1-scores, download model
  Day 14:   Retraining if accuracy < 85%

WEEK 3 (Days 15–21): Real-Time Engine + Web UI
  Day 15–16: Build unified_detector.py (sliding window, debounce, cooldown)
  Day 17–18: OpenCV overlay + TTS integration, test end-to-end
  Day 19–20: Build Gradio app.py (5 tabs)
  Day 21:   Full integration test by all 4 members

WEEK 4 (Days 22–30): Testing, Documentation, Presentation
  Day 22–24: Edge case testing and system hardening
  Day 25–26: Write thesis sections, insert charts
  Day 27–28: Build presentation deck, rehearse demo
  Day 29–30: Final review, GitHub release, submission
```

---

## 🎯 Target Metrics

| Metric | Minimum Target | Ideal Target |
| :--- | :--- | :--- |
| Model validation accuracy | > 85% | > 92% |
| Real-time inference FPS (local) | > 20 FPS | 28–30 FPS |
| Sign classes supported | 15 classes | 33 classes (A–Z + 7 gestures) |
| Sequences per class | 30 sequences | 50+ sequences |
| Idle class false-positive rate | < 15% | < 5% |
| TTS trigger latency after sign | < 1.0 second | < 0.5 second |

---

## 🔗 Key Tools & Links

| Tool | Purpose | Link |
| :--- | :--- | :--- |
| **Google Colab** | Free T4 GPU for model training | https://colab.research.google.com |
| **Google Drive** | Dataset cloud storage (shared folder) | https://drive.google.com |
| **GitHub** | Version control & team collaboration | https://github.com |
| **MediaPipe Holistic** | 138-landmark hand + pose extraction | https://developers.google.com/mediapipe |
| **PyTorch** | Deep learning framework | https://pytorch.org |
| **Gradio** | Web UI framework | https://www.gradio.app |
| **pyttsx3** | Offline text-to-speech (local) | https://pypi.org/project/pyttsx3 |

---

## ⚠️ Risk Management

| Risk | Probability | Mitigation |
| :--- | :--- | :--- |
| Colab session times out during training | Medium | Enable "Stay connected" in Colab; use model checkpointing every 10 epochs |
| Not enough training data | Medium | All 4 members each record sequences; data augmentation adds synthetic samples |
| Model accuracy < 85% | Medium | Collect more data for weak classes; increase Dropout rate; re-train |
| Someone's laptop too slow for MediaPipe + inference | Low | Reduce inference interval to 500ms; use ONNX export for optimization |
| GitHub merge conflicts | Medium | Use separate branches per feature; merge on Friday of each week |
| MediaPipe fails to detect hand | Low | Zero-pad missing hand features; show "No hand detected" warning |
