"""
extract_own_videos.py — convert your own sign videos into Vāksetu-format
(20, 506) sequences so your signs can be trained together with the 300
Vāksetu classes.

Input  : VIDEO_DIR/<ClassName>/*.mp4   (any of mp4/avi/mov/webm)
Output : OUT_DIR/<ClassName>/<video>.npy, each (20, 506) float32

Run locally or in Colab:
    python vaksetu/extract_own_videos.py --videos dataset_videos --out own_signs_v3
Then point EXTRA_NPY_DIR in colab_train_vaksetu.py at the output folder.
Tip: record each sign 3+ times, ideally by different team members.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vaksetu_features import video_to_sequence  # noqa: E402

EXTS = (".mp4", ".avi", ".mov", ".webm", ".mkv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder of class sub-folders with videos")
    ap.add_argument("--out", required=True, help="output folder for .npy sequences")
    ap.add_argument("--no-mirror", action="store_true",
                    help="videos are already mirrored (selfie recordings)")
    args = ap.parse_args()

    done = skipped = 0
    for cls in sorted(os.listdir(args.videos)):
        cdir = os.path.join(args.videos, cls)
        if not os.path.isdir(cdir):
            continue
        odir = os.path.join(args.out, cls)
        os.makedirs(odir, exist_ok=True)
        for fn in sorted(os.listdir(cdir)):
            if not fn.lower().endswith(EXTS):
                continue
            seq = video_to_sequence(os.path.join(cdir, fn), mirror=not args.no_mirror)
            if seq is None or not np.any(seq[:, :126]):
                print(f"  [skip] {cls}/{fn}: no hands found")
                skipped += 1
                continue
            np.save(os.path.join(odir, os.path.splitext(fn)[0] + ".npy"), seq)
            done += 1
        print(f"[{cls}] ok")
    print(f"[✓] {done} sequences written, {skipped} skipped -> {args.out}")


if __name__ == "__main__":
    main()
