# 🎫 TICKET-01 — ML Engineer / Team Lead
## Feature: PyTorch BiLSTM + Attention Model, Colab Training & Evaluation

**Assigned to:** Person 1  
**Files owned:** `model.py`, `train_model.py`, `colab/train.ipynb`, `colab/evaluate.ipynb`  
**Depends on:** TICKET-02 (dataset must be available on Google Drive before Day 10)  
**Due:** End of Week 2 (Day 14)

---

## 📋 Implementation Steps

### Part A — Model Architecture (`model.py`)

- [ ] Create file `model.py` in the project root
- [ ] Define class `UnifiedSignModel(nn.Module)` using PyTorch
- [ ] Add **Feature Projection layer**: `Linear(138 → 128)` + `BatchNorm1d(128)` + `ReLU` + `Dropout(0.3)`
  - > **Note:** `BatchNorm1d` expects input shape `(batch * seq, features)` — you must reshape before and after this layer. Flatten `(B, 60, 138)` → `(B*60, 138)`, apply BatchNorm, then reshape back to `(B, 60, 128)`.
- [ ] Add **BiLSTM layer**: `nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)`
- [ ] Add **Self-Attention layer**: `nn.Linear(256, 1)` — computes scalar score per time step, apply `softmax` across 60 steps, compute weighted sum → context vector `(B, 256)`
- [ ] Add **Classifier head**: `Linear(256 → 64)` + `ReLU` + `Dropout(0.3)` + `Linear(64 → num_classes)`
- [ ] Implement `forward(self, x)` method accepting shape `(batch, 60, 138)`
- [ ] Add `get_num_classes()` helper that reads `dataset/` folder and counts subdirectories

> **Note:** Do NOT apply `Softmax` in the `forward()` method. `nn.CrossEntropyLoss` expects raw logits. Only apply `torch.softmax()` at inference time.

---

### Part B — Training Pipeline (`train_model.py`)

- [ ] Create file `train_model.py` in the project root
- [ ] Write `SignDataset(Dataset)` class that:
  - [ ] Reads all `.npy` files from `dataset/<class_name>/` subdirectories
  - [ ] Assigns integer label to each class (sorted alphabetically for consistency)
  - [ ] Returns tensor of shape `(60, 138)` and integer label
  - [ ] Handles missing/corrupt `.npy` files gracefully with a `try/except`
- [ ] Write `augment_sequence(seq)` function applying all 5 augmentations:
  - [ ] Gaussian noise: `seq += np.random.normal(0, 0.01, seq.shape)`
  - [ ] Random rotation (±15°): rotate the `(x,y)` coordinates of each landmark
  - [ ] Random scale (0.85–1.15×): multiply all coordinates by a random scalar
  - [ ] Temporal speed warp: resample the 60-frame sequence to 48 or 72 frames then resize back to 60
  - [ ] Hand dropout: randomly zero-out all 63 left-hand features OR all 63 right-hand features (not both)
- [ ] Compute class weights for `CrossEntropyLoss` using `sklearn.utils.class_weight.compute_class_weight`
  - > **Note:** The `idle` class will have significantly more samples. Class weights prevent the model from just predicting "idle" for everything.
- [ ] Implement 80/20 train/validation split using `torch.utils.data.random_split`
- [ ] Use `DataLoader` with `batch_size=32`, `shuffle=True`, `num_workers=0`
  - > **Note:** Set `num_workers=0` on Windows to avoid multiprocessing errors.
- [ ] Training loop: `AdamW` optimizer (`lr=1e-3`, `weight_decay=1e-4`) + `CosineAnnealingLR` scheduler
- [ ] Save **best checkpoint only** (highest val accuracy): `torch.save(model.state_dict(), 'unified_sign_model.pt')`
- [ ] Save `class_labels.json` (list of class names in label order) alongside the `.pt` file
  - > **Note:** `class_labels.json` is CRITICAL — the inference engine needs it to map model output index → sign name.
- [ ] Print epoch number, train loss, val loss, val accuracy each epoch

---

### Part C — Colab Training Notebook (`colab/train.ipynb`)

- [ ] Create notebook `colab/train.ipynb`
- [ ] Cell 1: Install packages — `!pip install torch torchvision mediapipe opencv-python-headless scikit-learn`
- [ ] Cell 2: Mount Google Drive — `drive.mount('/content/drive')`
- [ ] Cell 3: Set paths — `DATASET_PATH = '/content/drive/MyDrive/SignLanguageAI/dataset/'`
- [ ] Cell 4: Copy `model.py` and `train_model.py` into Colab session from Drive
- [ ] Cell 5: Run training — `!python train_model.py --dataset $DATASET_PATH --epochs 100`
- [ ] Cell 6: Save outputs to Drive — copy `unified_sign_model.pt` and `class_labels.json` back to Drive
- [ ] Verify runtime is set to **GPU (T4)** before running — Runtime → Change runtime type → T4 GPU
  - > **Note:** Colab free tier disconnects after ~90 min of idle. Keep the browser tab active during training. Consider enabling "Stay awake" browser extension.

---

### Part D — Evaluation Notebook (`colab/evaluate.ipynb`)

- [ ] Create notebook `colab/evaluate.ipynb`
- [ ] Load `unified_sign_model.pt` with `model.load_state_dict(torch.load(...))`
- [ ] Run model on the validation split (held-out 20%)
- [ ] Generate **Confusion Matrix** using `sklearn.metrics.confusion_matrix` + `seaborn.heatmap`
  - [ ] X-axis: Predicted class, Y-axis: True class
  - [ ] Use class names as tick labels (from `class_labels.json`)
- [ ] Generate **Classification Report**: `sklearn.metrics.classification_report` (Precision, Recall, F1 per class)
- [ ] Plot **training loss curve** and **validation accuracy curve** on the same figure
- [ ] Save all plots: `confusion_matrix.png`, `training_curves.png` — export to Drive for thesis

---

## ✅ Test Checklist

- [ ] `model.py` imports without errors: `python -c "from model import UnifiedSignModel; print('OK')"`
- [ ] Model forward pass works with dummy input: `model(torch.randn(4, 60, 138))` returns shape `(4, num_classes)`
- [ ] `train_model.py` runs without crash on a small dummy dataset (3 classes, 5 sequences each)
- [ ] Training loss decreases over the first 10 epochs (not stuck or increasing)
- [ ] Validation accuracy reaches > 85% by epoch 50–80
- [ ] `unified_sign_model.pt` file exists and size is between 2 MB and 30 MB
- [ ] `class_labels.json` exists and contains all class names in correct order
- [ ] Confusion matrix image is readable with class names on both axes
- [ ] F1-score for `idle` class > 0.90 (most important class to get right)
- [ ] Model correctly classifies at least 3 signs from a manually prepared test batch

---

## 🔴 Important Notes

> **Blocker:** You cannot start Colab training (Part C) until TICKET-02 (Person 2) has uploaded the dataset to Google Drive. Coordinate with Person 2 by end of Day 9.

> **Critical File:** `class_labels.json` must be committed to GitHub alongside `model.py`. Without it, Person 3's inference engine cannot map predictions to sign names.

> **Retraining:** If val accuracy < 85% after 100 epochs, first check the confusion matrix for which classes are being confused. Share the confusion matrix with Person 2 so they can collect more data for those specific classes before retraining.
