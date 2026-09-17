# 🚀 ISL v2 Training Guide — Google Colab GPU

Complete walkthrough to retrain your sign language model with **normalized v2 features** (150 dims) for much better accuracy. All heavy work runs on Colab's free GPU.

---

## What's Different in v2?

| | v1 (old model) | v2 (new pipeline) |
|---|---|---|
| Features | 138 raw coordinates | **150 normalized** (wrist-relative handshape + 8 pose landmarks) |
| Position/scale invariant | ❌ | ✅ works for different people & camera distances |
| Augmentation | pre-saved (leaks into test) | on-the-fly, **train split only** |
| Validation | none/optimistic | honest video-level split |
| Class weights | no | yes (handles class imbalance) |
| Early stopping | no | yes (patience 15) |

---

## Part 0 — Prerequisites

1. Your ISL videos uploaded to Google Drive at:
   `MyDrive/ISL_Videos/` (the same folder `colab_download_isl.py` creates)
2. A Google account with Colab access (free tier is fine)

---

## Part 1 — Extract Landmarks (Colab, ~15-30 min)

> Extraction is CPU-bound; you don't need GPU for this part.

1. Open https://colab.research.google.com → **New notebook**
2. Open `colab_extract_v2.py` in this repo — each `CELL n:` block goes into one notebook cell
3. Run in order:
   - **CELL 1** — installs compatible `numpy<2` and `mediapipe==0.10.14`. **Note:** If NumPy was downgraded, click **Runtime → Restart session** (or `Runtime -> Restart runtime`), then proceed to CELL 2.
   - **CELL 2** — authorize Google Drive access
   - **CELL 3** — config; check the printed paths/classes
   - **CELL 4** — defines the v2 extractor and initializes MediaPipe Holistic
   - **CELL 5** — the actual extraction loop; watch per-class counts
   - **CELL 6** — optional cleanup

**Expected output:** per-class table like
```
  [OK ] Thank You           : 3 video sequence(s)
  [LOW] Cat-1               : 1 video sequence(s)
```

**Interpretation:**
- `[OK ]` — 3+ real videos → honest val/test possible
- `[LOW]` — 1-2 videos → val/test numbers will be optimistic (see Part 4)

Output lands in `MyDrive/SignLanguageAI/dataset_v2/`.

---

## Part 2 — Train the Model (Colab GPU, ~20-60 min)

1. **Runtime → Change runtime type → GPU (T4)** ← don't skip!
2. Upload `model.py` from this repo to Colab: use the Files sidebar (left) → upload to `/content/`
3. Paste cells from `colab_train_v2.py` and run in order:
   - **CELL 1** — GPU reminder (no install needed)
   - **CELL 2** — hyperparameters; defaults are sane
   - **CELL 3** — loads dataset + honest video-level split; shows split sizes
   - **CELL 4** — synthesizes `idle` sequences if none exist; builds loaders
   - **CELL 5** — training loop with early stopping; watch val accuracy
   - **CELL 6** — test accuracy + confusion matrix PNG
   - **CELL 7** — saves model + config to Drive

**Expected logs:**
```
[*] Device: cuda
[*] Split: train=...  val=...  test=...
Epoch   1 | loss 3.5... | train  12.0% | val  10.0%  ← best
...
Epoch  45 | loss 0.4... | train  92.3% | val  88.0%  ← best
[i] Early stopping at epoch 60.
🎯 TEST ACCURACY (real videos only): 85.4%
```

---

## Part 3 — Deploy Back to Your Project

Download **3 files** from `MyDrive/SignLanguageAI/` into your project root:

| File | Purpose |
|------|---------|
| `unified_sign_model_v2.pt` | trained weights |
| `model_config.json` | classes + dims; lets local code auto-detect v2 |
| `classes.json` | backup of class list |

Then locally:
```bash
python test_video.py videos/Bus.mp4     # sanity check: auto-uses v2 model
python app.py                            # Gradio demo, auto-uses v2 model
python unified_detector.py               # live webcam, auto-uses v2 model
```

The local code checks `model_config.json` first and falls back to the legacy `unified_sign_model.pt` — your old model keeps working until you delete it.

---

## Part 4 — Reading Your Results Honestly

- **Test accuracy is the real number.** Val accuracy is used for early stopping; classes with 1 video have optimistic val/test numbers because their test sample came from the same video.
- **Confusion matrix** (`confusion_matrix_v2.png`): rows = true sign, columns = prediction. Dark off-diagonal cells = confused pairs.
  - Confused pairs that are *genuinely similar* (e.g. "Eat"/"Food") → consider dropping one
  - Scattered errors on `[LOW]` classes → record more videos for those signs
- **If test accuracy is low but val is high** → data leakage or overfitting to augmentation; check the split printout

---

## Part 5 — Getting to 90%+

1. **Record your own videos** — 5+ clips per sign, different distances/backgrounds. This beats everything else.
2. **Drop near-duplicate signs** (keep the 20-25 most distinct) — accuracy rises sharply.
3. **Re-run Part 1 + Part 2** after adding videos; the pipeline handles the rest.
4. Realistic targets: 35 dictionary classes ≈ 75-85% · 20-25 distinct classes + your own recordings ≈ 90%+

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | Colab ships NumPy 2.x by default, which breaks MediaPipe's C-bindings. Run `!pip install "numpy<2" "protobuf<4.26.0" "mediapipe==0.10.14" opencv-python-headless tqdm`, then click **Runtime → Restart session**, and re-run from **CELL 2**. |
| Most classes `SKIP (no landmarks)` | videos too small/dark/hands out of frame → check a few manually |
| `CUDA out of memory` | lower `BATCH_SIZE` to 16 in CELL 2 |
| Training stuck at low acc | check CELL 3 printed real split counts (not all-1-video classes) |
| Colab disconnects | Drive artifacts from CELL 7 are safe; re-run from CELL 5 after reconnect |
| Local `test_video.py` loads old model | confirm `model_config.json` + `unified_sign_model_v2.pt` are both in project root |
