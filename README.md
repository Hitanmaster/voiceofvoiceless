# voiceofvoiceless

A small prototype for real-time hand-gesture recognition using MediaPipe and an LSTM sequence model.

**Summary**
- **Purpose:** Detect three hand gestures (`hello`, `thanks`, `iloveyou`) from webcam video using MediaPipe hand landmarks and an LSTM model.
- **Realtime demo:** `main.py` loads a trained model and predicts gestures from a rolling 30-frame window.

**Repository files**
- [data_collection.py](data_collection.py): capture webcam frames, extract hand keypoints with MediaPipe, and save 30-frame sequences as NumPy arrays under `MP_Data/<action>/<sequence>/<frame>.npy`.
- [train_model.py](train_model.py): load saved sequences, build and train an LSTM Keras model, and save the trained model as `action.h5`.
- [main.py](main.py): load `action.h5`, perform real-time landmark extraction and gesture prediction, and overlay the predicted label on the video feed.

**Data layout**
- `MP_Data/<action>/<sequence>/<frame>.npy` — each `.npy` file contains a 126-length keypoint vector (left+right hand: 21 points × 3 coords each × 2 hands) for a single frame.
- Each training sample is a sequence of 30 frames → shape `(30, 126)`.

**Quick start**
1. Install dependencies (recommended in a virtual env):

```bash
pip install opencv-python numpy mediapipe tensorflow scikit-learn
```

2. Collect data (webcam required):

```bash
python data_collection.py
```

3. Train the model (creates `action.h5`):

```bash
python train_model.py
```

4. Run realtime detection:

```bash
python main.py
```

Press `q` in the OpenCV window to quit.

**Notes & tips**
- Camera index is hardcoded to device 0; change `cv2.VideoCapture(0)` if needed.
- The detection threshold in `main.py` is 0.8 — lower it to increase sensitivity.
- `action.h5` must exist before running `main.py`.
- Add more actions or more training data to improve accuracy.

**Possible improvements**
- Add a `requirements.txt` and a more detailed `README` section for model evaluation.
- Add CLI options for paths, camera index, and threshold.
- Save model checkpoints, log validation accuracy, and provide a small evaluation script.

**License / Author**
- Project scaffold only — add license and author details as appropriate.
