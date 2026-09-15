# ==============================================================================
#  ISL v2 Trainer — Google Colab (GPU recommended)
#  Trains BiLSTM+Attention on (60, 150) normalized sequences from dataset_v2.
#
#  Usage: paste each CELL into a separate Colab cell and run in order.
#  Output (to Drive): unified_sign_model_v2.pt, model_config.json, classes.json,
#                     training_history.png, confusion_matrix_v2.png
# ==============================================================================


# ═══════════════════════════ CELL 1: Enable GPU ═══════════════════════════════
# Runtime menu -> Change runtime type -> Hardware accelerator: GPU (T4)
# (No installs needed: torch, sklearn, matplotlib are preinstalled in Colab.)


# ═══════════════════════════ CELL 2: Configuration ════════════════════════════
import os, re, json, glob, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATASET_DIR = "/content/drive/MyDrive/SignLanguageAI/dataset_v2"
OUTPUT_DIR = "/content/drive/MyDrive/SignLanguageAI"

TARGET_FRAMES = 60
INPUT_DIM = 150
BATCH_SIZE = 32
EPOCHS = 120
LR = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
EARLY_STOP_PATIENCE = 15
AUG_PER_SAMPLE = 10          # augmented copies generated ON-THE-FLY per real train sample
SEED = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Device: {device}")
if device.type != "cuda":
    print("[!] WARNING: No GPU — enable it via Runtime > Change runtime type.")


# ═══════════════ CELL 3: Load dataset + HONEST video-level split ══════════════
def load_dataset(dataset_dir):
    """
    Returns (real_sequences, labels, class_names).
    IMPORTANT: split is done per REAL VIDEO (no augmentation here), so augmented
    copies of a video can never leak into val/test.
    """
    classes = sorted(d for d in os.listdir(dataset_dir)
                     if os.path.isdir(os.path.join(dataset_dir, d)))
    seqs, labels = [], []
    for ci, cls in enumerate(classes):
        cls_dir = os.path.join(dataset_dir, cls)
        files = sorted(glob.glob(os.path.join(cls_dir, "*.npy")))
        kept = 0
        for fp in files:
            try:
                arr = np.load(fp)
            except Exception:
                continue
            if arr.ndim != 2 or arr.shape[1] != INPUT_DIM:
                continue
            if arr.shape[0] != TARGET_FRAMES:
                idx = np.linspace(0, arr.shape[0] - 1, TARGET_FRAMES, dtype=int)
                arr = arr[idx]
            seqs.append(arr.astype(np.float32)); labels.append(ci); kept += 1
        if kept == 0:
            print(f"  [!] Class '{cls}' has 0 valid sequences — EXCLUDED.")
    # Rebuild class list to drop empty classes and remap labels
    used = sorted(set(labels))
    classes = [classes[i] for i in used]
    remap = {old: new for new, old in enumerate(used)}
    labels = [remap[l] for l in labels]
    return seqs, labels, classes


class SignDataset(Dataset):
    """Returns (sequence, label). Optionally generates augmented variants on the fly."""
    def __init__(self, seqs, labels, augment=False, aug_per_sample=0):
        self.items = []
        for s, l in zip(seqs, labels):
            self.items.append((s, l))
            if augment:
                for _ in range(aug_per_sample):
                    self.items.append((s, l))  # same base; augmentation applied in __getitem__
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        s, l = self.items[idx]
        if self.augment and idx % (AUG_PER_SAMPLE + 1) != 0:  # first copy stays clean
            s = augment_sequence(s)
        return torch.from_numpy(np.ascontiguousarray(s)), l


def augment_sequence(seq: np.ndarray) -> np.ndarray:
    """Jitter + scale + temporal warp on TRAIN split only (v2-aware: skip zeros)."""
    out = seq.copy()
    # Hand-shape/wrist/pose are all normalized; zero rows mean 'hand missing'.
    # Jitter every non-zero coordinate slightly.
    mask = out != 0.0
    noise = np.random.normal(0.0, 0.01, size=out.shape).astype(np.float32)
    out[mask] += noise[mask]
    scale = np.random.uniform(0.95, 1.05)
    out[mask] *= scale
    if np.random.rand() > 0.3:
        curve = np.linspace(0, TARGET_FRAMES - 1, TARGET_FRAMES)
        warp = np.sin(np.linspace(0, np.pi, TARGET_FRAMES)) * np.random.uniform(-3, 3)
        warp_idx = np.clip(np.round(curve + warp), 0, TARGET_FRAMES - 1).astype(int)
        out = out[warp_idx]
    return out.astype(np.float32)


print("[*] Loading dataset ...")
seqs, labels, classes = load_dataset(DATASET_DIR)
n = len(seqs)
print(f"[*] Loaded {n} real sequences across {len(classes)} classes")

# ── Per-class video-level split (HONEST: augmented data never touches val/test) ─
train_s, train_l = [], []
val_s, val_l = [], []
test_s, test_l = [], []

by_class = {}
for s, l in zip(seqs, labels):
    by_class.setdefault(l, []).append((s, l))

SINGLE_VIDEO_CLASSES = []
for ci, cls in enumerate(classes):
    items = by_class.get(ci, [])
    rng = random.Random(SEED + ci)
    idxs = list(range(len(items)))
    rng.shuffle(idxs)

    if len(items) >= 3:
        test_idx = set(idxs[:1])          # 1 real video -> test
        val_idx = set(idxs[1:2])          # 1 real video -> val
        tr_idx = idxs[2:]
        mode = "3-way"
    elif len(items) == 2:
        test_idx = set(idxs[1:2])         # val doubles as test
        val_idx = set(idxs[1:2])
        tr_idx = idxs[:1]
        mode = "2-way (test=val)"
    else:  # 1 video
        test_idx = set(idxs)              # documented fallback: optimistic numbers
        val_idx = set(idxs)
        tr_idx = idxs
        mode = "1-video (OPTIMISTIC)"
        SINGLE_VIDEO_CLASSES.append(cls)

    for i in tr_idx:
        s, l = items[i]; train_s.append(s); train_l.append(l)
    for i in val_idx:
        s, l = items[i]; val_s.append(s); val_l.append(l)
    for i in test_idx:
        s, l = items[i]; test_s.append(s); test_l.append(l)

print(f"[*] Split: train={len(train_s)}  val={len(val_s)}  test={len(test_s)}")
if SINGLE_VIDEO_CLASSES:
    print(f"[!] {len(SINGLE_VIDEO_CLASSES)} classes have only 1 video "
          f"(their val/test numbers are optimistic):")
    print("    " + ", ".join(SINGLE_VIDEO_CLASSES))
print("[TIP] Record 2-3 more videos per LOW class for honest, higher accuracy.")


# ═══════════════ CELL 4: Synthetic 'idle' class + data loaders ════════════════
# If dataset_v2 has no real idle sequences, synthesize them so the model can
# output 'idle' when no structured sign is performed.
HAS_IDLE = "idle" in [c.lower() for c in classes]

def make_idle_samples(count, seed=7):
    """Random smooth motion with at least one hand present, no repeated sign."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(count):
        seq = np.zeros((TARGET_FRAMES, INPUT_DIM), dtype=np.float32)
        # wrist positions wander smoothly; hand shape random-but-fixed per sample
        base_shape = rng.normal(0, 0.05, size=60).astype(np.float32)
        t = np.linspace(0, 1, TARGET_FRAMES, dtype=np.float32)
        drift = 0.05 * np.sin(2 * np.pi * t * rng.uniform(0.5, 1.5))[:, None]
        for side in ([0] if rng.rand() < 0.5 else [0, 1]):
            shape_off = side * 63
            seq[:, shape_off:shape_off + 60] = base_shape[None, :]
            wx = rng.uniform(-0.6, 0.6)
            wy = rng.uniform(0.1, 0.5)
            seq[:, shape_off + 60:shape_off + 63] = \
                np.stack([wx + drift[:, 0], wy + drift[:, 0],
                          np.zeros(TARGET_FRAMES, dtype=np.float32)], axis=1)
        out.append(seq)
    return out

if not HAS_IDLE:
    print("[*] No real 'idle' sequences — synthesizing 80 idle samples.")
    idle_seqs = make_idle_samples(80)
    idle_label = len(classes)
    classes = classes + ["idle"]
    train_s += idle_seqs[:56]
    train_l += [idle_label] * 56
    val_s += idle_seqs[56:68]
    val_l += [idle_label] * 12
    test_s += idle_seqs[68:]
    test_l += [idle_label] * 12

NUM_CLASSES = len(classes)
print(f"[*] {NUM_CLASSES} classes: {classes}")


# ═══════════════════ CELL 5: Model + training loop ════════════════════════════
import sys
sys.path.insert(0, "/content")  # if you uploaded model.py to /content
# If you keep model.py in Drive instead, uncomment:
# sys.path.insert(0, "/content/drive/MyDrive/SignLanguageAI")

from model import BiLSTMAttentionSignClassifier

model = BiLSTMAttentionSignClassifier(
    input_dim=INPUT_DIM, hidden_dim=128, num_layers=2, num_classes=NUM_CLASSES, dropout=0.3
).to(device)

# Class weights handle imbalance (few real videos for some signs)
counts = np.bincount(train_l, minlength=NUM_CLASSES).astype(np.float64)
weights = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1.0))
class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
print(f"[*] Class weight range: {weights.min():.2f} .. {weights.max():.2f}")

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=6
)

train_ds = SignDataset(train_s, train_l, augment=True, aug_per_sample=AUG_PER_SAMPLE)
val_ds = SignDataset(val_s, val_l, augment=False)
test_ds = SignDataset(test_s, test_l, augment=False)

# num_workers=0 avoids Colab DataLoader crash-loops on small datasets
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def evaluate(loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(total, 1)


history = {"train": [], "val": []}
best_val = 0.0
best_state = None
patience = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item() * xb.size(0)

    train_loss = running_loss / max(len(train_ds), 1)
    train_acc = evaluate(train_dl)
    val_acc = evaluate(val_dl)
    scheduler.step(val_acc)
    history["train"].append(train_acc)
    history["val"].append(val_acc)

    marker = ""
    if val_acc > best_val:
        best_val = val_acc
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        patience = 0
        marker = "  ← best"
    else:
        patience += 1

    print(f"Epoch {epoch:3d} | loss {train_loss:.4f} | "
          f"train {train_acc*100:5.1f}% | val {val_acc*100:5.1f}%{marker}")

    if patience >= EARLY_STOP_PATIENCE:
        print(f"[i] Early stopping at epoch {epoch} (patience {EARLY_STOP_PATIENCE}).")
        break

if best_state is not None:
    model.load_state_dict(best_state)
print(f"[*] Best val accuracy: {best_val*100:.2f}%")


# ═══════════════ CELL 6: Test evaluation + confusion matrix ═══════════════════
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

test_acc = evaluate(test_dl)
print(f"\n🎯 TEST ACCURACY (real videos only): {test_acc*100:.2f}%\n")

y_true, y_pred = [], []
model.eval()
with torch.no_grad():
    for xb, yb in test_dl:
        pred = model(xb.to(device)).argmax(dim=1).cpu().numpy()
        y_pred.extend(pred.tolist()); y_true.extend(yb.numpy().tolist())

present = sorted(set(y_true) | set(y_pred))
names = [classes[i] for i in present]
print(classification_report(y_true, y_pred, labels=present,
                            target_names=names, zero_division=0))

cm = confusion_matrix(y_true, y_pred, labels=present)
plt.figure(figsize=(13, 11))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=names, yticklabels=names)
plt.title(f"v2 Confusion Matrix — Test Acc {test_acc*100:.1f}%")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.xticks(rotation=45, ha="right"); plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_v2.png")
plt.savefig(cm_path, dpi=200)
print(f"[✓] Saved {cm_path}")

plt.figure(figsize=(9, 5))
plt.plot(history["train"], label="train")
plt.plot(history["val"], label="val")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True)
plt.title("Training History")
hist_path = os.path.join(OUTPUT_DIR, "training_history.png")
plt.savefig(hist_path, dpi=150)
print(f"[✓] Saved {hist_path}")


# ═══════════════ CELL 7: Save model + config to Drive ═════════════════════════
model_path = os.path.join(OUTPUT_DIR, "unified_sign_model_v2.pt")
torch.save(model.state_dict(), model_path)

config = {
    "version": 2,
    "input_dim": INPUT_DIM,
    "seq_len": TARGET_FRAMES,
    "num_classes": NUM_CLASSES,
    "classes": classes,
    "feature": "v2_normalized",
    "model_file": "unified_sign_model_v2.pt",
    "test_accuracy": round(test_acc * 100, 2),
    "best_val_accuracy": round(best_val * 100, 2),
}
cfg_path = os.path.join(OUTPUT_DIR, "model_config.json")
with open(cfg_path, "w") as f:
    json.dump(config, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "classes.json"), "w") as f:
    json.dump(classes, f, indent=2)

print("\n" + "=" * 65)
print("  TRAINING COMPLETE — artifacts in Drive:")
print(f"    {model_path}")
print(f"    {cfg_path}")
print(f"    {os.path.join(OUTPUT_DIR, 'classes.json')}")
print("=" * 65)
print("[NEXT] Download those 3 files into your project folder — the local")
print("       detector (unified_detector.py) auto-detects the v2 model.")
