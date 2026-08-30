# ============================================================
#  ISL Video Downloader for Google Colab + Google Drive
#  Run each cell in order. Runtime: GPU/CPU doesn't matter.
# ============================================================

# ── CELL 1: Install dependencies ────────────────────────────
# !pip install -q yt-dlp

# ── CELL 2: Mount Google Drive ──────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ── CELL 3: Configure paths ─────────────────────────────────
import os

# Change this to wherever you want videos saved in your Drive
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/ISL_Videos"

# Number of parallel download threads (keep <= 6 to avoid YT throttling)
NUM_WORKERS = 6

os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)
print(f"[OK] Videos will be saved to: {DRIVE_OUTPUT_DIR}")

# ── CELL 4: Load video list from GitHub ─────────────────────
import json, re, time, os, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import yt_dlp

# Load isl_videos.json directly from your public GitHub repo
JSON_URL = "https://raw.githubusercontent.com/Hitanmaster/voiceofvoiceless/main/isl_videos.json"

print("[*] Fetching video list from GitHub ...")
with urllib.request.urlopen(JSON_URL) as resp:
    RAW_ITEMS = json.loads(resp.read().decode())

print(f"[OK] Loaded {len(RAW_ITEMS)} entries from isl_videos.json")

# ── CELL 5: Build download queue ────────────────────────────
def sanitize(name):
    name = re.sub(r'[\\/*?:"<>|]', '', name)
    return name.strip()

def build_queue(raw_items):
    seen = {}
    queue = []
    for item in raw_items:
        name  = item.get("name", "").strip()
        url   = item.get("video_url", "").strip()
        if not url or "Not found" in url or not url.startswith("http"):
            continue
        safe = sanitize(name)
        if safe in seen:
            seen[safe] += 1
            fname = f"{safe}_{seen[safe]}.mp4"
        else:
            seen[safe] = 1
            fname = f"{safe}.mp4"
        queue.append({"name": name, "filename": fname, "url": url})
    return queue

QUEUE = build_queue(RAW_ITEMS)
print(f"[OK] {len(QUEUE)} videos queued for download")

# ── CELL 6: Download function ───────────────────────────────
def download_video(item, out_dir):
    fname = item["filename"]
    url   = item["url"]
    fpath = os.path.join(out_dir, fname)

    # Skip already-downloaded files larger than 1 KB
    if os.path.exists(fpath) and os.path.getsize(fpath) > 1024:
        return item["name"], fname, "SKIP"

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": os.path.join(out_dir, fname.rsplit(".", 1)[0] + ".%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "retries": 5,
        "fragment_retries": 5,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios", "web_embedded", "mweb"]
            }
        },
    }

    for attempt in range(3):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return item["name"], fname, "OK"
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
            else:
                return item["name"], fname, f"FAIL: {str(e)[:80]}"

# ── CELL 7: Run downloads ───────────────────────────────────
ok = skip = fail = done = 0
total = len(QUEUE)

print("=" * 60)
print(f"  ISL Video Downloader - {total} videos")
print(f"  Saving to: {DRIVE_OUTPUT_DIR}")
print(f"  Workers  : {NUM_WORKERS}")
print("=" * 60)

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
    futures = {ex.submit(download_video, item, DRIVE_OUTPUT_DIR): item for item in QUEUE}
    for future in as_completed(futures):
        done += 1
        name, fname, status = future.result()
        if   status == "OK":   ok   += 1; tag = "[OK]  "
        elif status == "SKIP": skip += 1; tag = "[SKIP]"
        else:                  fail += 1; tag = "[FAIL]"
        print(f"[{done:4d}/{total}] {tag} {fname:<40s} {'' if status in ('OK','SKIP') else status}")

print("\n" + "=" * 60)
print(f"  Downloaded : {ok}")
print(f"  Skipped    : {skip}")
print(f"  Failed     : {fail}")
print(f"  Saved to   : {DRIVE_OUTPUT_DIR}")
print("=" * 60)
