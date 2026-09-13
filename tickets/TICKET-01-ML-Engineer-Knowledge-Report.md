# 🧠 ML Engineer Comprehensive Knowledge Report
## Person 1 — What to Know, What to Use, What to Study

> This document covers every tool, library, concept, algorithm, and topic that Person 1 (ML Engineer / Team Lead)
> must understand to successfully build, train, and evaluate the BiLSTM + Attention model for the Sign Language AI System.

---

## 📦 SECTION 1: Tools & Libraries to Install and Use

### 1.1 Core Deep Learning — PyTorch

| What | Why Needed | Where Used |
| :--- | :--- | :--- |
| `torch` | The main deep learning framework | `model.py`, `train_model.py` |
| `torch.nn` | Building blocks: Linear, LSTM, Dropout, BatchNorm | `model.py` |
| `torch.optim` | Optimizers: AdamW, LR schedulers | `train_model.py` |
| `torch.utils.data` | Dataset, DataLoader, random_split | `train_model.py` |
| `torchvision` | Utility transforms (optional) | `train_model.py` |

**Install:**
```bash
pip install torch torchvision
```

> **Note on PyTorch Versions:** Use PyTorch 2.x (latest stable). On Google Colab, PyTorch comes pre-installed.
> Locally on Windows, install the CPU version if no NVIDIA GPU: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

---

### 1.2 Scientific Computing

| What | Why Needed | Where Used |
| :--- | :--- | :--- |
| `numpy` | Array operations, data augmentation math | `train_model.py` |
| `scipy.interpolate` | Temporal speed warping (resample sequences) | `train_model.py` → augmentation |

**Install:**
```bash
pip install numpy scipy
```

---

### 1.3 Data & Machine Learning Utilities

| What | Why Needed | Where Used |
| :--- | :--- | :--- |
| `sklearn.utils.class_weight` | Compute class weights to handle idle class imbalance | `train_model.py` |
| `sklearn.metrics` | Confusion matrix, classification report, F1-score | `colab/evaluate.ipynb` |
| `json` | Save/load `class_labels.json` | `train_model.py`, `model.py` |
| `os`, `pathlib` | File scanning, directory creation | `train_model.py` |

**Install:**
```bash
pip install scikit-learn
```

---

### 1.4 Visualization

| What | Why Needed | Where Used |
| :--- | :--- | :--- |
| `matplotlib.pyplot` | Plot training loss and accuracy curves | `colab/evaluate.ipynb` |
| `seaborn` | Beautiful confusion matrix heatmap | `colab/evaluate.ipynb` |

**Install:**
```bash
pip install matplotlib seaborn
```

---

### 1.5 Google Colab Specific

| What | Why Needed |
| :--- | :--- |
| `google.colab.drive` | Mount Google Drive to access dataset |
| Colab T4 GPU runtime | Free GPU for 10–20x faster training than CPU |
| `tqdm` | Progress bar for training loop epochs |

```python
# First cell in every Colab notebook:
from google.colab import drive
drive.mount('/content/drive')
```

---

## 🏗️ SECTION 2: Architecture Topics — What to Know and Understand

### 2.1 What is a Neural Network? (Foundation)

A neural network is a stack of mathematical transformations (layers) that learns to map an input to an output by adjusting internal parameters (weights) during training.

**Key concepts to know:**
- **Layer:** A mathematical operation (e.g., `y = W·x + b`)
- **Activation function:** Adds non-linearity (ReLU, Sigmoid, Tanh, Softmax)
- **Forward pass:** Input flows through all layers → produces output (prediction)
- **Backward pass:** Computes gradients of loss with respect to weights
- **Gradient Descent:** Updates weights in the direction that reduces loss

---

### 2.2 Recurrent Neural Networks (RNN) — The Foundation of LSTM

Sign language gestures are **sequences** — the order of frames matters. Standard (feedforward) neural networks cannot handle sequences because they treat each input independently. RNNs process sequences step-by-step, maintaining a **hidden state** that carries information from past steps.

```
Frame_1 → [RNN Cell] → hidden_1
Frame_2 → [RNN Cell] ← hidden_1 → hidden_2
Frame_3 → [RNN Cell] ← hidden_2 → hidden_3
...
Frame_60 → [RNN Cell] ← hidden_59 → Output
```

**Problem with basic RNNs:** They suffer from the **vanishing gradient problem** — gradients shrink exponentially during backpropagation through many time steps, making it impossible to learn long-range dependencies (e.g., how frame 1 relates to frame 60).

---

### 2.3 LSTM (Long Short-Term Memory) — The Fix for Vanishing Gradients

LSTM is a special type of RNN cell designed to remember information over long sequences. It does this using **gates** that control what to keep, what to forget, and what to output.

**The 3 LSTM Gates:**

| Gate | Formula | Purpose |
| :--- | :--- | :--- |
| **Forget Gate** | $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$ | Decides what to erase from cell state |
| **Input Gate** | $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$ | Decides what new info to add to cell state |
| **Output Gate** | $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$ | Decides what hidden state to output |

**Cell state update:**
$$C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t$$

**Hidden state output:**
$$h_t = o_t \cdot \tanh(C_t)$$

**What Person 1 needs to know practically:**
- LSTM can learn which frames in a sign are important and carry that information forward
- A static sign (letter "A") will show low variance across 60 frames → LSTM learns the fixed shape
- A motion sign ("Hello") will show high variance → LSTM learns the trajectory

**PyTorch usage:**
```python
self.lstm = nn.LSTM(
    input_size=128,       # features per frame (after projection)
    hidden_size=128,      # hidden state dimensions
    num_layers=2,         # stacking 2 LSTM layers
    batch_first=True,     # input shape: (batch, seq, features)
    bidirectional=True,   # read sequence forward AND backward
    dropout=0.3           # regularization between layers
)
# Output shape: (batch, 60, 256) — 256 because bidirectional doubles hidden_size
```

---

### 2.4 Bidirectional LSTM (BiLSTM) — Why "Bi"?

A standard LSTM only reads frames **left → right** (past to future). A **BiLSTM** reads the sequence in **both directions simultaneously** — one LSTM reads forward (frame 1 → 60), another reads backward (frame 60 → 1).

```
Forward LSTM:   Frame_1 → Frame_2 → ... → Frame_60
Backward LSTM:  Frame_60 → Frame_59 → ... → Frame_1
                                    ↓
              Concatenate outputs: [forward_h_t || backward_h_t] → shape (256,)
```

**Why it helps for signs:**
- For motion signs like "Thank You", the ending pose (frame 60) gives context to understand the beginning (frame 10).
- BiLSTM captures both "what came before" and "what comes after" each frame.

---

### 2.5 Self-Attention Mechanism — Focusing on What Matters

Not all 60 frames contribute equally to recognizing a sign:
- For "Hello" (waving), the frames at peak extension matter most
- For "A" (static letter), all frames are equally important
- For "Thank You", the starting chin-touch frames are the key identifier

**Self-Attention** learns to assign a weight (importance score) to each frame automatically:

$$e_t = \tanh(W_a \cdot h_t + b_a) \quad \text{(score for each frame)}$$
$$\alpha_t = \text{softmax}(e_t) \quad \text{(normalized weights summing to 1.0)}$$
$$c = \sum_{t=1}^{60} \alpha_t \cdot h_t \quad \text{(weighted average = context vector)}$$

**PyTorch implementation:**
```python
self.attention = nn.Linear(256, 1)

# In forward():
attn_scores = self.attention(lstm_out)      # (B, 60, 1)
attn_weights = torch.softmax(attn_scores, dim=1)  # normalize over 60 frames
context = torch.sum(attn_weights * lstm_out, dim=1)  # (B, 256)
```

**What to know:** The attention weights become interpretable — you can visualize which frames the model focused on for each sign. This is excellent thesis material.

---

### 2.6 BatchNorm1d — Stabilizing Training

**Batch Normalization** normalizes the input to each layer so that it has mean ≈ 0 and variance ≈ 1. This prevents one feature from dominating the training (e.g., a large z-coordinate value swamping smaller x,y values).

**Critical gotcha in this project:**
`BatchNorm1d` expects 2D input `(N, features)`, but the sequence input is 3D `(B, 60, 138)`.

**Solution — reshape around BatchNorm:**
```python
B, S, F = x.shape          # (batch, 60, 138)
x = x.view(B * S, F)       # reshape to (B*60, 138)
x = self.bn(x)             # apply BatchNorm1d
x = x.view(B, S, -1)       # reshape back to (B, 60, 128)
```

---

### 2.7 Dropout — Preventing Overfitting

**Dropout** randomly sets some neuron outputs to zero during training (with probability `p=0.3`). This forces the network to not rely too heavily on any single neuron, improving generalization.

- During training: `Dropout(0.3)` is active
- During inference/evaluation: Dropout is automatically turned off when you call `model.eval()`

> **Important:** Always call `model.eval()` before running inference. Forgetting this is a common bug that causes lower accuracy in production vs. training.

---

### 2.8 Softmax & CrossEntropyLoss — The Output Layer

**Softmax** converts raw output scores (logits) into probabilities that sum to 1.0:
$$P(\text{class}_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**CrossEntropyLoss** combines Softmax + negative log likelihood into one:
$$\mathcal{L} = -\log(P(\text{true class}))$$

**Critical rule:** Do NOT put `Softmax` in `model.forward()`. `nn.CrossEntropyLoss` applies it internally. Only add `torch.softmax()` when doing inference (predicting).

```python
# Training: no softmax
loss = criterion(logits, labels)

# Inference: add softmax manually
probs = torch.softmax(model(x), dim=1)
confidence, predicted_class = probs.max(dim=1)
```

---

## 🔢 SECTION 3: Training Concepts — What to Know

### 3.1 Optimizer: AdamW

**AdamW** (Adam with Weight Decay) is an optimizer that adapts the learning rate for each parameter individually.

- **Adam** combines momentum (smooth updates) + RMSprop (adaptive learning rates)
- **Weight Decay** (L2 regularization) adds a small penalty for large weights, preventing overfitting

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,           # initial learning rate
    weight_decay=1e-4  # L2 regularization strength
)
```

---

### 3.2 Learning Rate Scheduler: CosineAnnealingLR

The learning rate should decrease over time — large steps early (explore), small steps later (fine-tune). **CosineAnnealingLR** reduces the LR following a cosine curve from `lr_max` to `lr_min`:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,   # total epochs
    eta_min=1e-5 # minimum learning rate
)

# After each epoch:
scheduler.step()
```

---

### 3.3 Train/Validation Split

- **Training set (80%):** Model sees this data and updates weights from it
- **Validation set (20%):** Model never trains on this — used only to measure generalization

```python
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
```

> **Why not just use training accuracy?** Training accuracy will always be high — the model memorizes training data. Validation accuracy tells you if it generalizes to new, unseen signers.

---

### 3.4 Class Weights — Handling Imbalanced Data

The `idle` class has 60+ sequences while others may have only 30. Without correction, the model will learn to just predict "idle" most of the time and achieve misleadingly high accuracy.

**Solution:** Assign higher loss penalty to underrepresented classes.

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_labels),
    y=all_labels
)
weight_tensor = torch.FloatTensor(class_weights)
criterion = nn.CrossEntropyLoss(weight=weight_tensor)
```

---

### 3.5 Data Augmentation for Sequences

Since collecting 1000+ sequences per class is not feasible, augmentation artificially expands the dataset by applying realistic variations:

| Augmentation | What It Simulates | Code |
| :--- | :--- | :--- |
| **Gaussian Noise** | Minor hand tremor, sensor jitter | `seq += np.random.normal(0, 0.01, seq.shape)` |
| **Random Scale** | Signer closer/farther from camera | `seq *= np.random.uniform(0.85, 1.15)` |
| **Random Rotation** | Tilted camera or body angle | Rotate (x,y) by angle θ using rotation matrix |
| **Temporal Warp** | Fast vs. slow signing speed | Resample from 60→48 or 60→72 frames, then resize back to 60 |
| **Hand Dropout** | One hand partially hidden | Zero out features 0:63 OR 63:126 randomly |

**Rotation augmentation (2D rotation in x-y plane):**
```python
def rotate_landmarks(seq, angle_deg):
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    # Apply R to every (x, y) pair in every frame
    for frame_idx in range(seq.shape[0]):
        for lm in range(0, seq.shape[1], 3):  # x at lm, y at lm+1
            xy = seq[frame_idx, lm:lm+2]
            seq[frame_idx, lm:lm+2] = R @ xy
    return seq
```

---

### 3.6 Saving & Loading Model Checkpoints

```python
# Save only the best model (when val_accuracy improves):
best_val_acc = 0.0
if val_acc > best_val_acc:
    best_val_acc = val_acc
    torch.save(model.state_dict(), 'unified_sign_model.pt')

# Load for inference:
model = UnifiedSignModel(input_dim=138, hidden_dim=128, num_classes=33)
model.load_state_dict(torch.load('unified_sign_model.pt', map_location='cpu'))
model.eval()
```

> **`map_location='cpu'`** is critical — if the model was trained on GPU (Colab T4) and loaded on a CPU-only local machine, this parameter prevents a crash.

---

## 📊 SECTION 4: Evaluation Topics — What to Know

### 4.1 Confusion Matrix

A grid showing how many times the model predicted each class vs. the actual true class.

- **Rows** = True class (what the sign actually was)
- **Columns** = Predicted class (what the model said)
- **Diagonal** = Correct predictions
- **Off-diagonal** = Mistakes (e.g., model predicted "B" when it was "D")

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted'); plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
```

---

### 4.2 Precision, Recall, F1-Score

These three metrics give a more complete picture than accuracy alone.

| Metric | Formula | What It Measures |
| :--- | :--- | :--- |
| **Precision** | TP / (TP + FP) | Of all times model predicted class X, how many were correct? |
| **Recall** | TP / (TP + FN) | Of all true class X samples, how many did the model catch? |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean of Precision and Recall |

**For sign language:**
- High **Recall** for `idle` is critical — missing a non-sign (false negative) causes false TTS triggers
- High **Precision** for all signs — wrong predictions are embarrassing in a live demo

```python
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, target_names=class_labels))
```

---

### 4.3 Training & Validation Curves

Plot loss and accuracy over epochs to diagnose training:

| Pattern | What It Means | What to Do |
| :--- | :--- | :--- |
| Both curves improving → plateau | Normal training | Good, model converged |
| Train loss low, val loss high | **Overfitting** | Add more Dropout, collect more data |
| Both losses high & not decreasing | **Underfitting** | Increase hidden_dim, add more LSTM layers |
| Val loss spikes then recovers | Unstable | Lower learning rate |
| Val accuracy jumps between epochs | Too small dataset | Collect more sequences |

```python
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend(); plt.title('Loss Curves')

plt.subplot(1,2,2)
plt.plot(val_accuracies, label='Val Accuracy')
plt.axhline(y=0.85, color='r', linestyle='--', label='85% target')
plt.legend(); plt.title('Validation Accuracy')
plt.savefig('training_curves.png', dpi=150)
```

---

## 📚 SECTION 5: Study Topics & Learning Path

These are the topics Person 1 should study (in this order) to understand what they're building:

### Priority 1 — Must Know Before Starting (Days 1–3)
- [ ] **Python classes** and `__init__`, `__len__`, `__getitem__` dunder methods (needed for `Dataset` class)
- [ ] **NumPy**: `np.load`, `np.save`, array shapes, `np.concatenate`, `np.zeros`
- [ ] **PyTorch tensors**: `torch.tensor()`, `.shape`, `.view()`, `.to(device)`, difference from NumPy
- [ ] **`nn.Module` in PyTorch**: how `__init__` and `forward()` work, what `super().__init__()` does

### Priority 2 — Need Before Writing model.py (Days 4–6)
- [ ] **LSTM concept**: what hidden state and cell state are, why it solves vanishing gradient
- [ ] **BiLSTM**: how forward and backward LSTMs concatenate outputs
- [ ] **Dropout**: what it does during training vs. inference, `model.train()` vs. `model.eval()`
- [ ] **BatchNorm1d**: what it normalizes, why input shape matters (2D not 3D)
- [ ] **Softmax vs. CrossEntropyLoss**: why you don't apply Softmax in forward()

### Priority 3 — Need Before Training (Days 7–9)
- [ ] **PyTorch Dataset & DataLoader**: how `__getitem__` feeds data into training loop
- [ ] **`random_split`**: how to split dataset into train/val
- [ ] **AdamW optimizer**: conceptually what it does, `lr` and `weight_decay` parameters
- [ ] **LR Scheduler**: what CosineAnnealingLR does to learning rate over epochs
- [ ] **Class weights**: why imbalanced classes need correction, how to compute them

### Priority 4 — Need for Evaluation (Days 10–12)
- [ ] **Confusion matrix**: how to read it, what off-diagonal values mean
- [ ] **Precision, Recall, F1**: formulas, when each matters
- [ ] **`matplotlib`**: `plt.plot()`, `plt.subplot()`, `plt.savefig()` basics
- [ ] **`seaborn.heatmap()`**: for confusion matrix visualization

### Priority 5 — Nice to Know for Thesis (Days 12–14)
- [ ] **Attention mechanism**: what the attention weights mean, how to visualize them
- [ ] **Overfitting vs. Underfitting**: how to diagnose from loss curves
- [ ] **ONNX export**: `torch.onnx.export()` for optimized inference (optional)
- [ ] **Transfer learning** (conceptual): why we train from scratch vs. fine-tuning a pre-trained model

---

## 🔗 SECTION 6: Recommended Learning Resources

| Topic | Resource | Type |
| :--- | :--- | :--- |
| PyTorch basics | https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html | Official Tutorial |
| LSTM explained visually | https://colah.github.io/posts/2015-08-Understanding-LSTMs | Blog (highly recommended) |
| Attention mechanism | https://jalammar.github.io/visualizing-neural-machine-translation | Visual Blog |
| BiLSTM for sequence classification | https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html | Official Tutorial |
| PyTorch Dataset & DataLoader | https://pytorch.org/tutorials/beginner/data_loading_tutorial.html | Official Tutorial |
| Confusion matrix & metrics | https://scikit-learn.org/stable/modules/model_evaluation.html | Scikit-learn Docs |
| Google Colab GPU tips | https://research.google.com/colaboratory/faq.html | Official FAQ |

---

## ⚡ SECTION 7: Quick Reference — Complete model.py Skeleton

```python
import torch
import torch.nn as nn

class UnifiedSignModel(nn.Module):
    def __init__(self, input_dim=138, hidden_dim=128, num_classes=33):
        super().__init__()

        # Step 1: Project 138 features → 128 (with normalization)
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Step 2: BiLSTM — learns temporal patterns
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Step 3: Self-Attention — score each of the 60 frames
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Step 4: Classifier — maps context vector to class probabilities
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
            # NO Softmax here — CrossEntropyLoss handles it
        )

    def forward(self, x):
        # x shape: (batch, 60, 138)
        B, S, F = x.shape

        # Apply feature projection (requires 2D input for BatchNorm1d)
        x_flat = x.reshape(B * S, F)
        x_proj = self.feature_proj(x_flat)
        x_seq = x_proj.reshape(B, S, -1)       # (B, 60, 128)

        # BiLSTM
        lstm_out, _ = self.lstm(x_seq)          # (B, 60, 256)

        # Self-Attention
        attn_scores = self.attention(lstm_out)  # (B, 60, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (B, 256)

        # Classification
        logits = self.classifier(context)       # (B, num_classes)
        return logits


# Quick test — run this to verify the model works:
if __name__ == "__main__":
    model = UnifiedSignModel(input_dim=138, hidden_dim=128, num_classes=33)
    dummy = torch.randn(4, 60, 138)   # batch of 4, 60 frames, 138 features
    output = model(dummy)
    print(f"Output shape: {output.shape}")  # Expected: torch.Size([4, 33])
    print("Model architecture OK!")
```

---

## 📋 Summary Checklist for Person 1

### Before Writing Code
- [ ] Understand LSTM gates (conceptually — no need to memorize formulas)
- [ ] Understand why BiLSTM is better than unidirectional LSTM for gestures
- [ ] Know the difference between `model.train()` and `model.eval()`
- [ ] Know why Softmax is NOT in `forward()` when using CrossEntropyLoss

### While Writing Code
- [ ] Test model forward pass before writing any training loop
- [ ] Print output shapes at each layer during development
- [ ] Always set `map_location='cpu'` when loading `.pt` file on local machine
- [ ] Save `class_labels.json` in the same folder as `unified_sign_model.pt`

### After Training
- [ ] Validate accuracy > 85% on the held-out validation set
- [ ] Check confusion matrix for the most confused class pairs
- [ ] Share `confusion_matrix.png` and `training_curves.png` with Person 4 for thesis
- [ ] Commit `model.py`, `train_model.py`, `class_labels.json` to GitHub
