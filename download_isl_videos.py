"""
Indian Sign Language Video Downloader
Downloads all sign videos listed in isl_videos.txt or isl_videos.json
and saves them as MP4 files named by their sign word in the 'videos/' folder.
Uses mobile & embedded player clients to bypass YouTube bot/sign-in blocks.
"""

import os
import re
import sys
import json
import time
import warnings
import imageio_ffmpeg
from concurrent.futures import ThreadPoolExecutor, as_completed

# Filter deprecation warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

class StderrFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
    def write(self, s):
        if "Deprecated Feature: Support for Python" in s or "n challenge solving failed" in s:
            return
        self.original_stderr.write(s)
    def flush(self):
        self.original_stderr.flush()

sys.stderr = StderrFilter(sys.stderr)

import yt_dlp

INPUT_TXT = "isl_videos.txt"
INPUT_JSON = "isl_videos.json"
OUTPUT_DIR = "videos"
NUM_WORKERS = 4   # Safe concurrency to avoid YouTube IP throttling

# Get ffmpeg binary path
try:
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = None

def sanitize_filename(name):
    """Remove illegal Windows characters from filenames."""
    name = re.sub(r'[\\/*?:"<>|]', '', name)
    name = name.strip().replace('  ', ' ')
    return name

def load_video_list():
    """Load items from isl_videos.txt (or isl_videos.json)."""
    items = []
    seen_names = {}

    if os.path.exists(INPUT_TXT):
        with open(INPUT_TXT, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if " : " in line:
                    sign_name, video_url = line.split(" : ", 1)
                    sign_name = sign_name.strip()
                    video_url = video_url.strip()
                    
                    if video_url.startswith("http") and "Not found" not in video_url:
                        safe_name = sanitize_filename(sign_name)
                        if safe_name in seen_names:
                            seen_names[safe_name] += 1
                            filename = f"{safe_name}_{seen_names[safe_name]}.mp4"
                        else:
                            seen_names[safe_name] = 1
                            filename = f"{safe_name}.mp4"
                        
                        items.append({
                            "name": sign_name,
                            "filename": filename,
                            "url": video_url
                        })
    return items

def download_video(item, output_dir):
    """Download a single video using yt-dlp with mobile client fallback."""
    filename = item["filename"]
    video_url = item["url"]
    filepath = os.path.join(output_dir, filename)

    # Skip if already downloaded and file size > 1KB
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1024:
        return item["name"], filename, "ALREADY_EXISTS"

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best/18',
        'outtmpl': os.path.join(output_dir, filename.rsplit('.', 1)[0] + '.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'retries': 5,
        'fragment_retries': 5,
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'ios', 'web_embedded', 'mweb']
            }
        }
    }
    
    if FFMPEG_EXE:
        ydl_opts['ffmpeg_location'] = FFMPEG_EXE

    # Attempt download with retry
    for attempt in range(2):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            return item["name"], filename, "SUCCESS"
        except Exception as e:
            if attempt == 0:
                time.sleep(2)
            else:
                return item["name"], filename, f"ERROR: {str(e)[:80]}"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    items = load_video_list()
    total = len(items)

    print("=" * 65)
    print(f"[*] Indian Sign Language Video Downloader")
    print(f"[*] Total videos to download: {total}")
    print(f"[*] Target directory: '{os.path.abspath(OUTPUT_DIR)}'")
    print(f"[*] Concurrent download threads: {NUM_WORKERS}")
    print("=" * 65 + "\n")

    if total == 0:
        print("[!] No videos found to download.")
        return

    success_count = 0
    skipped_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_item = {
            executor.submit(download_video, item, OUTPUT_DIR): item
            for item in items
        }

        completed = 0
        for future in as_completed(future_to_item):
            completed += 1
            sign_name, filename, status = future.result()

            if status == "SUCCESS":
                success_count += 1
                status_str = "[OK]"
            elif status == "ALREADY_EXISTS":
                skipped_count += 1
                status_str = "[SKIP]"
            else:
                failed_count += 1
                status_str = "[FAIL]"

            print(f"[{completed}/{total}] {status_str} {filename:35s} | {sign_name}")

    print("\n" + "=" * 65)
    print(f"[*] DOWNLOAD SUMMARY")
    print(f"[*] Total Processed : {completed}")
    print(f"[*] Newly Downloaded: {success_count}")
    print(f"[*] Already Existed : {skipped_count}")
    print(f"[*] Failed          : {failed_count}")
    print(f"[*] Saved to folder : '{os.path.abspath(OUTPUT_DIR)}'")
    print("=" * 65)

if __name__ == "__main__":
    main()
