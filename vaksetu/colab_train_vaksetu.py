# ==============================================================================
#  Voiceless to Voice — v3 trainer on the Vāksetu ISL Feature Dataset (300 signs)
#  Google Colab, GPU (T4) recommended. Paste each CELL into its own Colab cell.
#
#  Dataset: "Vāksetu ISL Feature Dataset (v1.0)", Zenodo, Oct 5 2026,
#           DOI 10.5281/zenodo.23004815, CC BY 4.0 (credit it in the report).
#
#  What this does
#    * downloads the 2.2 GB HDF5 + manifest + quality flags to Drive (once)
#    * de-duplicates every array by SHA-256 BEFORE splitting (the set has
#      ~45k exact duplicates; skipping this gives fake-high test accuracy)
#    * drops tracking-failure / static rows (quality_flags.csv) and arrays
#      that carry conflicting labels
#    * merges all 23 negative categories into one "idle" reject class
#      (the live detector already ignores "idle")
#    * optionally merges your own signs extracted with extract_own_videos.py
#    * trains the same BiLSTM + attention model on (20, 506) input
#    * reports overall test accuracy AND accuracy on INCLUDE-derived
#      (other-signer) samples, which is the honest generalisation number
#
#  Output (OUTPUT_DIR): unified_sign_model_v3.pt, model_config.json,
#                       classes.json, confusion_top_errors.txt, history.png
# ==============================================================================


# ═══════════════════════════ CELL 1: Setup ════════════════════════════════════
# Runtime -> Change runtime type -> T4 GPU
# !pip -q install h5py
from google.colab import drive
drive.mount("/content/drive")


# ═══════════════════════════ CELL 2: Configuration ════════════════════════════
import os, sys, json, hashlib, random, subprocess
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = "/content/drive/MyDrive/SignLanguageAI"
DATA_DIR = f"{ROOT}/vaksetu_data"
OUTPUT_DIR = f"{ROOT}/vaksetu_v3"
PROJECT_DIR = f"{ROOT}/voiceofvoiceless"   # repo copy with model.py + vaksetu/
EXTRA_NPY_DIR = f"{ROOT}/own_signs_v3"     # optional: class folders of (20,506) .npy

ZENODO = "https://zenodo.org/records/23004815/files"
FILES = ["dataset_consolidated.h5", "manifest_consolidated.csv", "quality_flags.csv"]

USE_PHASE2 = True        # add refining_phase2 positives (300 classes)
USE_ADAPTER = False      # refining_adapter: 87% model-labelled -> noisy, off by default
REJECT_NAME = "idle"     # name of the merged negative class
MAX_REJECT = 6000        # cap on reject samples after dedupe

SEQ_LEN, INPUT_DIM, BASE_DIM = 20, 506, 253
HIDDEN = 128
BATCH = 256
EPOCHS = 60
LR = 2e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
PATIENCE = 8
SEED = 42

os.makedirs(DATA_DIR, exist_ok=True); os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[*] device:", device)


# ═══════════════════════════ CELL 3: Download (once) ══════════════════════════
for fn in FILES:
    dst = f"{DATA_DIR}/{fn}"
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print("[=] have", fn); continue
    print("[↓]", fn)
    subprocess.run(["wget", "-q", "--show-progress", "-O", dst,
                    f"{ZENODO}/{fn}?download=1"], check=True)
H5 = f"{DATA_DIR}/dataset_consolidated.h5"


# ═══════════════════ CELL 4: Load, label, de-duplicate ════════════════════════
def _col(df, *keys):
    """Find a manifest column by keyword (column names are matched loosely)."""
    for c in df.columns:
        lc = c.lower()
        if all(k in lc for k in keys):
            return c
    return None

manifest = pd.read_csv(f"{DATA_DIR}/manifest_consolidated.csv")
M_GROUP, M_FILE = _col(manifest, "group"), _col(manifest, "file")
M_CLASS = _col(manifest, "class")
M_DOMAIN = _col(manifest, "domain")
print("[*] manifest columns:", list(manifest.columns))

flags = pd.read_csv(f"{DATA_DIR}/quality_flags.csv")
F_SHA = _col(flags, "sha")
BAD_SHA = set(flags[F_SHA].astype(str).str.lower()) if F_SHA else set()
print(f"[*] quality-flagged arrays to drop: {len(BAD_SHA)}")

def _s(x):
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)

def group_labels(f, group):
    """Per-row class names + domains for one HDF5 group."""
    g = f[group]
    n = g["data"].shape[0]
    names = [_s(x) for x in g["filenames"][:]]
    if group == "main_positive" and "labels" in g and "class_names" in f:
        cls_names = [_s(x) for x in f["class_names"][:]]
        classes = [cls_names[i] for i in g["labels"][:]]
    elif group == "main_positive" and "labels" in g and "class_names" in g:
        cls_names = [_s(x) for x in g["class_names"][:]]
        classes = [cls_names[i] for i in g["labels"][:]]
    else:
        sub = manifest[manifest[M_GROUP] == group].set_index(M_FILE)
        classes = [str(sub.at[fn, M_CLASS]) if fn in sub.index else None for fn in names]
    if "domains" in g:
        domains = [_s(x) for x in g["domains"][:]]
    elif M_DOMAIN:
        sub = manifest[manifest[M_GROUP] == group].set_index(M_FILE)
        domains = [str(sub.at[fn, M_DOMAIN]) if fn in sub.index else "unknown" for fn in names]
    else:
        domains = ["unknown"] * n
    return classes, domains

seen = {}            # sha -> index in X
X, Y, D, HS = [], [], [], [] # arrays, class names, domains, hashes
conflict = set()

def ingest(f, group, reject=False, chunk=2048):
    classes, domains = group_labels(f, group)
    data = f[group]["data"]
    n = data.shape[0]
    kept = dup = bad = 0
    for s in range(0, n, chunk):
        block = data[s:s + chunk]
        for j, arr in enumerate(block):
            i = s + j
            cls = REJECT_NAME if reject else classes[i]
            if cls is None or cls == "nan":
                continue
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            h = hashlib.sha256(arr.tobytes()).hexdigest()
            if h in BAD_SHA and not reject:
                bad += 1; continue
            if h in seen:
                if Y[seen[h]] != cls:
                    conflict.add(h)
                dup += 1; continue
            seen[h] = len(X); X.append(arr); Y.append(cls); D.append(domains[i]); HS.append(h)
            kept += 1
    print(f"  {group:18s} rows={n:6d} kept={kept:6d} dup={dup:6d} flagged={bad}")

with h5py.File(H5, "r") as f:
    print("[*] groups:", list(f.keys()))
    ingest(f, "main_positive")
    if USE_PHASE2:  ingest(f, "refining_phase2")
    if USE_ADAPTER: ingest(f, "refining_adapter")
    ingest(f, "negative", reject=True)
    ingest(f, "refining_negative", reject=True)

# drop arrays whose label disagrees across groups
if conflict:
    keep = [i for i in range(len(X)) if HS[i] not in conflict]
    X = [X[i] for i in keep]; Y = [Y[i] for i in keep]; D = [D[i] for i in keep]
    HS = [HS[i] for i in keep]
    print(f"[*] dropped {len(conflict)} label-conflict arrays")

# optional: your own signs, extracted with extract_own_videos.py
if os.path.isdir(EXTRA_NPY_DIR):
    added = 0
    for cls in sorted(os.listdir(EXTRA_NPY_DIR)):
        cdir = os.path.join(EXTRA_NPY_DIR, cls)
        if not os.path.isdir(cdir): continue
        for fn in sorted(os.listdir(cdir)):
            if not fn.endswith(".npy"): continue
            arr = np.load(os.path.join(cdir, fn)).astype(np.float32)
            if arr.shape != (SEQ_LEN, INPUT_DIM): continue
            X.append(arr); Y.append(cls); D.append("own"); HS.append(fn); added += 1
    print(f"[*] added {added} of your own sequences from {EXTRA_NPY_DIR}")

# cap the reject class so it doesn't dominate
rej = [i for i, y in enumerate(Y) if y == REJECT_NAME]
if len(rej) > MAX_REJECT:
    drop = set(random.Random(SEED).sample(rej, len(rej) - MAX_REJECT))
    keep = [i for i in range(len(X)) if i not in drop]
    X = [X[i] for i in keep]; Y = [Y[i] for i in keep]; D = [D[i] for i in keep]

X = np.stack(X).astype(np.float32)
classes = sorted(set(Y) - {REJECT_NAME}) + [REJECT_NAME]
cidx = {c: i for i, c in enumerate(classes)}
y = np.array([cidx[c] for c in Y], dtype=np.int64)
D = np.array(D)
print(f"[*] unique samples: {len(X)}  classes: {len(classes)}  ({X.nbytes/1e9:.2f} GB)")


# ═══════════════════ CELL 5: Split AFTER de-duplication ═══════════════════════
# Stratified 80/10/10. Every INCLUDE (MVI, other signers) sample is pushed to
# test where its class still has training data, so we get an honest
# "new signer" number alongside the overall one.
rng = np.random.RandomState(SEED)
tr, va, te = [], [], []
for c in range(len(classes)):
    idx = np.where(y == c)[0]
    mvi = [i for i in idx if D[i].lower().startswith("mvi")]
    rest = [i for i in idx if not D[i].lower().startswith("mvi")]
    rng.shuffle(rest)
    n = len(rest)
    if n < 3:
        tr += rest; te += mvi; continue
    n_te = max(1, int(round(0.1 * n))); n_va = max(1, int(round(0.1 * n)))
    te += rest[:n_te] + mvi
    va += rest[n_te:n_te + n_va]
    tr += rest[n_te + n_va:]
tr, va, te = map(np.array, (tr, va, te))
n_mvi = int(np.char.startswith(np.char.lower(D[te].astype(str)), "mvi").sum())
print(f"[*] split train={len(tr)} val={len(va)} test={len(te)} (test incl. {n_mvi} INCLUDE samples)")


# ═══════════════════ CELL 6: Data loaders + augmentation ══════════════════════
def augment(seq):
    """Augment the 253 base features, then recompute velocity (keeps them consistent)."""
    base = seq[:, :BASE_DIM].copy()
    mask = base != 0
    base[mask] += np.random.normal(0, 0.01, size=mask.sum()).astype(np.float32)
    base[mask] *= np.random.uniform(0.95, 1.05)
    if np.random.rand() < 0.5:                      # temporal warp / speed change
        k = np.random.randint(16, 25)
        src = np.linspace(0, SEQ_LEN - 1, k).round().astype(int)
        base = base[src][np.linspace(0, k - 1, SEQ_LEN).round().astype(int)]
    if np.random.rand() < 0.1:                      # drop one hand for a few frames
        s = np.random.randint(0, SEQ_LEN - 4); hand = np.random.randint(2)
        base[s:s + 4, hand * 63:(hand + 1) * 63] = 0
        base[s:s + 4, 126 + hand * 63:126 + (hand + 1) * 63] = 0
    vel = np.zeros_like(base); vel[1:] = base[1:] - base[:-1]
    return np.concatenate([base, vel], 1).astype(np.float32)

class SeqDS(Dataset):
    def __init__(self, idx, aug=False): self.idx, self.aug = idx, aug
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        i = self.idx[k]; s = X[i]
        if self.aug: s = augment(s)
        return torch.from_numpy(np.ascontiguousarray(s)), int(y[i])

train_dl = DataLoader(SeqDS(tr, aug=True), batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)
val_dl = DataLoader(SeqDS(va), batch_size=512, num_workers=2)
test_dl = DataLoader(SeqDS(te), batch_size=512, num_workers=2)


# ═══════════════════ CELL 7: Model + training ═════════════════════════════════
sys.path.insert(0, PROJECT_DIR)
from model import BiLSTMAttentionSignClassifier

model = BiLSTMAttentionSignClassifier(input_dim=INPUT_DIM, hidden_dim=HIDDEN, num_layers=2,
                                      num_classes=len(classes), dropout=0.3).to(device)
counts = np.bincount(y[tr], minlength=len(classes)).astype(np.float64)
w = counts.sum() / (len(classes) * np.maximum(counts, 1))
w = np.clip(w, 0.2, 5.0)
crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device),
                           label_smoothing=LABEL_SMOOTHING)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, epochs=EPOCHS,
                                            steps_per_epoch=len(train_dl))

def predict(loader):
    model.eval(); P, T = [], []
    with torch.no_grad():
        for xb, yb in loader:
            P.append(model(xb.to(device)).argmax(1).cpu().numpy()); T.append(yb.numpy())
    return np.concatenate(P), np.concatenate(T)

hist, best, best_state, bad = {"val": [], "loss": []}, 0.0, None, 0
for ep in range(1, EPOCHS + 1):
    model.train(); tot = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); loss = crit(model(xb), yb); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        tot += loss.item() * xb.size(0)
    p, t = predict(val_dl); acc = (p == t).mean()
    hist["val"].append(acc); hist["loss"].append(tot / len(tr))
    mark = ""
    if acc > best:
        best, bad, mark = acc, 0, "  <- best"
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
    print(f"epoch {ep:3d} | loss {tot/len(tr):.4f} | val {acc*100:5.2f}%{mark}")
    if bad >= PATIENCE:
        print("[i] early stop"); break
model.load_state_dict(best_state)


# ═══════════════════ CELL 8: Test + honest numbers ════════════════════════════
p, t = predict(test_dl)
test_acc = (p == t).mean()
is_mvi = np.char.startswith(np.char.lower(D[te].astype(str)), "mvi")
mvi_acc = (p[is_mvi] == t[is_mvi]).mean() if is_mvi.any() else float("nan")
idle = cidx[REJECT_NAME]
rej_recall = (p[t == idle] == idle).mean() if (t == idle).any() else float("nan")
print(f"\nTEST accuracy (de-duplicated): {test_acc*100:.2f}%")
print(f"INCLUDE / other-signer accuracy: {mvi_acc*100:.2f}%  (n={int(is_mvi.sum())})")
print(f"idle (reject) recall: {rej_recall*100:.2f}%")

from collections import Counter
errs = Counter((classes[a], classes[b]) for a, b in zip(t, p) if a != b)
with open(f"{OUTPUT_DIR}/confusion_top_errors.txt", "w") as fh:
    for (a, b), n in errs.most_common(50):
        fh.write(f"{a} -> {b}: {n}\n")

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4)); plt.plot(hist["val"]); plt.title("val accuracy")
plt.xlabel("epoch"); plt.grid(True); plt.savefig(f"{OUTPUT_DIR}/history.png", dpi=120)


# ═══════════════════ CELL 9: Save for the live detector ═══════════════════════
torch.save(model.state_dict(), f"{OUTPUT_DIR}/unified_sign_model_v3.pt")
cfg = {
    "version": 3,
    "feature": "vaksetu_506",
    "input_dim": INPUT_DIM,
    "base_dim": BASE_DIM,
    "seq_len": SEQ_LEN,
    "window_frames": 45,          # live buffer (~1.5 s @30fps) sampled down to 20
    "hidden_dim": HIDDEN,
    "num_classes": len(classes),
    "classes": classes,
    "model_file": "unified_sign_model_v3.pt",
    "test_accuracy": round(float(test_acc) * 100, 2),
    "include_other_signer_accuracy": round(float(mvi_acc) * 100, 2),
    "dataset": "Vaksetu ISL Feature Dataset v1.0, DOI 10.5281/zenodo.23004815, CC BY 4.0",
}
json.dump(cfg, open(f"{OUTPUT_DIR}/model_config.json", "w"), indent=2)
json.dump(classes, open(f"{OUTPUT_DIR}/classes.json", "w"), indent=2)
print("\n[✓] saved to", OUTPUT_DIR)
print("    copy unified_sign_model_v3.pt + model_config.json + classes.json")
print("    into the project root; unified_detector.py picks v3 up automatically.")
