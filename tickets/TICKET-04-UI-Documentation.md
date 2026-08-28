# 🎫 TICKET-04 — UI / Documentation Lead
## Feature: Gradio 5-Tab Web Interface & Project Documentation

**Assigned to:** Person 4  
**Files owned:** `app.py`, `README.md`, thesis charts integration  
**Depends on:** TICKET-03 (inference engine must work before wiring into Gradio), TICKET-01 (charts from training)  
**Due:** End of Week 3 (Day 21) for `app.py`; End of Week 4 (Day 26) for docs

---

## 📋 Implementation Steps

### Part A — Gradio App Setup (`app.py`)

- [ ] Create file `app.py` in the project root
- [ ] Install and import Gradio: `import gradio as gr`
- [ ] Import: `cv2`, `numpy as np`, `torch`, `json`, `os`, `time`, `threading`
- [ ] At the top of `app.py`, write `load_model()` — same logic as TICKET-03's model loader
  - > **Note:** Keep model loading in a function, not at module level. This prevents crash-on-import if `.pt` file is missing.
- [ ] Create a `shared_state` dictionary to hold the current prediction across threads:
  ```python
  shared_state = {
      "prediction": "—",
      "confidence": 0.0,
      "history": [],
      "recording": False
  }
  ```

---

### Part B — Tab 1: Live Detection

- [ ] Create function `process_webcam_frame(frame)`:
  - [ ] Accept an image frame from Gradio Webcam input (numpy array, RGB)
  - [ ] Run MediaPipe Holistic on the frame
  - [ ] Extract 138-feature vector
  - [ ] Append to a module-level `deque(maxlen=60)`
  - [ ] If buffer is full, run inference
  - [ ] Return annotated frame (with sign text + confidence drawn on it), current prediction string, confidence value
- [ ] Build Tab 1 layout:
  - [ ] `gr.Image(source="webcam", streaming=True)` — live camera input
  - [ ] `gr.Label()` — displays predicted sign name
  - [ ] `gr.Slider(0, 1)` — shows confidence value (read-only, updated live)
  - [ ] `gr.Textbox(lines=5)` — detection history log (last 10 detections with timestamps)
  - > **Note:** Gradio's streaming webcam processes one frame at a time via `gr.Interface`. Use `live=True` on the interface for real-time updates.

---

### Part C — Tab 2: Record New Sign

- [ ] Build Tab 2 layout:
  - [ ] `gr.Textbox(label="Sign Name")` — user types name (e.g. `"hello"`)
  - [ ] `gr.Number(label="Number of Sequences", value=5)`
  - [ ] `gr.Button("▶ Start Recording")`
  - [ ] `gr.Textbox(label="Status", interactive=False)` — shows countdown + frame counter
- [ ] On button click, call `start_recording(sign_name, num_sequences)`:
  - [ ] Opens webcam (or reuses open stream)
  - [ ] Runs the same recording logic as `dataset_collector.py`
  - [ ] Updates `Status` textbox with `"Countdown: 3..."`, `"Recording: Frame 23/60"`, `"Saved! (Seq 2/5)"`
  - [ ] Saves `.npy` files to `dataset/<sign_name>/`
  - > **Note:** Recording blocks the UI thread. Use `gr.Progress()` or run in a background thread and update state via `gr.update()`.

---

### Part D — Tab 3: Train Model

- [ ] Build Tab 3 layout:
  - [ ] `gr.Button("🚀 Start Training")`
  - [ ] `gr.Textbox(label="Training Log", lines=15, interactive=False)` — live output
  - [ ] `gr.Number(label="Epochs", value=100)`
- [ ] On button click, call `start_training(epochs)`:
  - [ ] Run `python train_model.py --epochs {epochs}` as a subprocess
  - [ ] Stream stdout line-by-line into the Training Log textbox
  - [ ] When training finishes, show: `"✅ Training complete! Model saved as unified_sign_model.pt"`
  - > **Note:** Use `subprocess.Popen` with `stdout=subprocess.PIPE` to stream output. Display in the textbox using `yield` with `gr.update()`.

---

### Part E — Tab 4: Dataset Manager

- [ ] Build Tab 4 layout:
  - [ ] `gr.Dataframe()` — table with columns: Sign Name | Sequence Count | Status
  - [ ] `gr.Button("🔄 Refresh")` — rescans `dataset/` folder and updates table
  - [ ] `gr.Plot()` — bar chart showing sequence counts per class (use `matplotlib`)
  - [ ] `gr.Textbox(label="Sign to Delete")` + `gr.Button("🗑️ Delete Class")` — removes `dataset/<name>/`
- [ ] Write `get_dataset_stats()` function:
  - [ ] Scan `dataset/` directory
  - [ ] Count `.npy` files per subdirectory
  - [ ] Return as list of `[sign_name, count, "✅ Ready" if count >= 30 else "⚠️ Needs more data"]`
  - > **Note:** Warn on classes with < 30 sequences by showing "⚠️" in the Status column.

---

### Part F — Tab 5: Settings

- [ ] Build Tab 5 layout:
  - [ ] `gr.Slider(0.5, 0.99, value=0.80, label="Confidence Threshold")`
  - [ ] `gr.Slider(0.5, 5.0, value=2.0, label="Cooldown (seconds)")`
  - [ ] `gr.Slider(100, 1000, value=300, step=50, label="Inference Interval (ms)")`
  - [ ] `gr.Radio(["Unified (DL)", "Static Only (RF)", "Motion Only (DTW)"], label="Detection Mode")`
  - [ ] `gr.Number(label="Webcam Device Index", value=0)`
  - [ ] `gr.Button("💾 Apply Settings")` — updates `shared_state` config values
  - > **Note:** Settings changes take effect on the next inference cycle — no restart needed.

---

### Part G — Launch & main.py Integration

- [ ] At the bottom of `app.py`, add:
  ```python
  if __name__ == "__main__":
      demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
  ```
- [ ] In `main.py`, add **Option 11**: `"11. Launch Gradio Web Interface"` → runs `os.system("python app.py")` or `subprocess.Popen`
- [ ] Test that `http://127.0.0.1:7860` opens in the browser and all 5 tabs are visible

---

### Part H — Thesis Documentation Assets

- [ ] Collect `confusion_matrix.png` and `training_curves.png` from Person 1 after training
- [ ] Place in `docs/` folder for thesis inclusion
- [ ] Write system architecture block diagram (can use draw.io, Mermaid, or PowerPoint)
- [ ] Update `README.md` with:
  - [ ] Project overview (2–3 sentences)
  - [ ] Requirements: `pip install -r requirements.txt`
  - [ ] How to collect data: `python dataset_collector.py`
  - [ ] How to train: `python train_model.py` or use Colab
  - [ ] How to run live detection: `python main.py` → Option 10
  - [ ] How to launch web UI: `python app.py` or `python main.py` → Option 11
  - [ ] Screenshots of the Gradio UI tabs

---

## ✅ Test Checklist

- [ ] `python app.py` starts without errors
- [ ] Browser opens `http://127.0.0.1:7860` and all 5 tabs are visible
- [ ] Tab 1: Webcam feed shows and prediction label updates while signing
- [ ] Tab 2: Recording a sign creates `.npy` file in `dataset/<sign_name>/`
- [ ] Tab 3: Clicking "Start Training" shows log output updating in real-time
- [ ] Tab 4: Dataset table lists all classes with correct sequence counts
- [ ] Tab 4: Clicking "Refresh" updates counts after new recording
- [ ] Tab 5: Changing confidence slider and clicking "Apply" updates detection behavior
- [ ] Option 11 in `main.py` launches the Gradio app without error
- [ ] `README.md` has working setup and run instructions (tested on a clean venv)

---

## 🔴 Important Notes

> **Gradio Webcam in Streaming Mode:** Use `gr.Interface(fn=process_webcam_frame, inputs=gr.Image(source='webcam', streaming=True), live=True)`. Without `live=True`, the webcam won't update in real-time.

> **Do not run `cv2.imshow` in `app.py`:** Gradio manages its own UI — use return values from functions to display images, not OpenCV windows.

> **Wait for Person 1's charts:** Do not finalize the thesis section until you receive `confusion_matrix.png` and `training_curves.png` from Person 1. Leave placeholder sections in the thesis document in the meantime.
