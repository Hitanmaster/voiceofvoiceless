# v3: 300 ISL signs with the Vāksetu dataset

This branch adds a third model version trained on the **Vāksetu ISL Feature Dataset v1.0**
(Goa College of Engineering, Zenodo, 5 Oct 2026, DOI [10.5281/zenodo.23004815](https://doi.org/10.5281/zenodo.23004815), CC BY 4.0).
That's 300 sign classes and a reject class, compared with the 35 signs + idle in v2.

v1 and v2 still work. The detector picks whichever `model_config.json` is in the project root.

## What changed

| File | Change |
|---|---|
| `vaksetu/vaksetu_features.py` | Live feature extraction in the Vāksetu format: 253 base features per frame (both hands wrist-centred and scaled, the same hands relative to the face, a hand-to-face proximity value) plus 253 frame-to-frame velocities, giving **(20, 506)**. Ported from the dataset author's own extractor and checked to match it exactly. |
| `vaksetu/colab_train_vaksetu.py` | Colab trainer: downloads the data, de-duplicates by SHA-256 **before** splitting, drops flagged and label-conflict rows, merges the 23 negative categories into `idle`, then trains the same BiLSTM + attention model. |
| `vaksetu/extract_own_videos.py` | Converts our own recorded videos into the same (20, 506) format, so our signs can be trained alongside the 300. |
| `feature_utils.py` | Recognises `version: 3` configs and passes `hidden_dim` / `window_frames` through. |
| `unified_detector.py` | For v3 it keeps a ~1.5 s buffer of 253-dim frames, samples 20 of them, and adds velocity before each prediction. v1 and v2 behave as before. |
| `app.py` | For v3, uploaded videos are mirrored to match the webcam view. The specs tab now mentions v3. |

## How to train (Colab, T4)

1. Copy this repo folder to `MyDrive/SignLanguageAI/voiceofvoiceless` (the trainer imports `model.py` from there).
2. Optional: to add our own signs, run
   `python vaksetu/extract_own_videos.py --videos <class folders of videos> --out /content/drive/MyDrive/SignLanguageAI/own_signs_v3`
3. Open `vaksetu/colab_train_vaksetu.py` and run its cells in order. The first run downloads about 2.2 GB to Drive.
4. Copy `unified_sign_model_v3.pt`, `model_config.json` and `classes.json` from `MyDrive/SignLanguageAI/vaksetu_v3/` into the project root.
5. Run `python unified_detector.py` or `python app.py`. The startup log should read `v3, input_dim=506`.

## Numbers to report honestly

The trainer prints three numbers:
- **Test accuracy**, measured on de-duplicated data. Without de-duplication, copies of the same array land in both train and test and the score is inflated.
- **INCLUDE / other-signer accuracy**, from the ~920 samples extracted from INCLUDE videos of other people. This is the real indicator of how the model will do on new signers. Expect it to be much lower, because almost all of the remaining data comes from one person.
- **idle recall**, how well the model rejects non-signs.

## Hand check (do this once)

Vāksetu was recorded on a mirrored webcam image using MediaPipe's HandLandmarker. Our detector uses Holistic, which labels left and right the opposite way on a mirrored image. `SWAP_HOLISTIC_HANDS = True` in `vaksetu_features.py` corrects for this.
If one-handed signs keep coming out wrong when you test live, set it to `False` and compare.

## Credit

Dataset: Joseph Jonathan Fernandes, *Vāksetu ISL Feature Dataset (v1.0)*, Zenodo, 2026, CC BY 4.0.
INCLUDE-derived samples: Sridhar, Ganesan, Kumar & Khapra, ACM Multimedia 2020, CC BY 4.0.
