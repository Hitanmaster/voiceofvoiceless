"""
Motion Sign Recognition Module
================================
Recognizes dynamic/motion signs (Hello, Goodbye, Please, Thank You, Yes, No, etc.)
by recording hand landmark sequences over time and comparing using DTW.

Works on CPU only - no GPU needed!
"""

import cv2
import os
import time
import numpy as np
import platform
import warnings
from gtts import gTTS
import mediapipe as mp

warnings.filterwarnings("ignore")

# =========================================================================
# MEDIAPIPE SETUP
# =========================================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

MOTION_DIR = 'motion_signs'
RECORDING_DURATION = 10  # seconds
TARGET_FRAMES = 60       # resample all recordings to this many frames
PREDICTION_DURATION = 5  # seconds for prediction capture
CONFIDENCE_THRESHOLD = 15.0  # DTW distance threshold (below = confident match)


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================
def play_audio(filename):
    """OS ke hisaab se audio play karta hai."""
    if platform.system() == "Windows":
        os.system(f"start {filename}")
    elif platform.system() == "Darwin":
        os.system(f"afplay {filename}")
    else:
        os.system(f"xdg-open {filename}")


def ensure_motion_dir():
    """Motion signs directory banata hai agar nahi hai."""
    if not os.path.exists(MOTION_DIR):
        os.makedirs(MOTION_DIR)


# =========================================================================
# NORMALIZATION - Position & Scale Independent
# =========================================================================
def normalize_frame(landmarks_63):
    """
    Single frame ke 63 values (21 landmarks x 3 coords) ko normalize karta hai.
    - Wrist (landmark 0) ko origin banata hai
    - Palm size se scale karta hai
    """
    coords = np.array(landmarks_63).reshape(21, 3)

    # Wrist ko origin banao
    wrist = coords[0].copy()
    coords = coords - wrist

    # Palm size se normalize karo (wrist to middle finger MCP distance)
    palm_size = np.linalg.norm(coords[9])  # landmark 9 = middle finger MCP
    if palm_size > 0.001:  # avoid division by zero
        coords = coords / palm_size

    return coords.flatten()


def normalize_sequence(sequence):
    """
    Puri sequence ko normalize karta hai:
    1. Har frame ko independently normalize karta hai
    2. Fixed number of frames mein resample karta hai
    """
    if len(sequence) == 0:
        return np.zeros((TARGET_FRAMES, 63))

    # Normalize each frame
    normalized = np.array([normalize_frame(frame) for frame in sequence])

    # Resample to TARGET_FRAMES using linear interpolation
    if len(normalized) == TARGET_FRAMES:
        return normalized

    original_indices = np.linspace(0, 1, len(normalized))
    target_indices = np.linspace(0, 1, TARGET_FRAMES)

    resampled = np.zeros((TARGET_FRAMES, 63))
    for feat_idx in range(63):
        resampled[:, feat_idx] = np.interp(target_indices, original_indices, normalized[:, feat_idx])

    return resampled


# =========================================================================
# DTW (Dynamic Time Warping) - Pure NumPy Implementation
# =========================================================================
def dtw_distance(seq1, seq2):
    """
    Do sequences ke beech DTW distance compute karta hai.
    Ye algorithm time warping handle karta hai - matlab agar sign
    thoda fast ya slow kiya toh bhi match karega.

    seq1, seq2: shape (T, 63) - T frames, 63 features per frame
    Returns: float distance (lower = more similar)
    """
    n, m = len(seq1), len(seq2)

    # Use only key landmarks for faster computation
    # Fingertips (4,8,12,16,20) + wrist (0) + MCP joints (5,9,13,17)
    key_indices = []
    for lm_idx in [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]:
        key_indices.extend([lm_idx * 3, lm_idx * 3 + 1, lm_idx * 3 + 2])

    s1 = seq1[:, key_indices]
    s2 = seq2[:, key_indices]

    # DTW cost matrix
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = np.linalg.norm(s1[i - 1] - s2[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    return cost[n, m] / (n + m)  # Normalize by path length


def dtw_distance_fast(seq1, seq2, window=10):
    """
    Faster DTW with Sakoe-Chiba band constraint.
    Only checks within a window around the diagonal - much faster for long sequences.
    """
    n, m = len(seq1), len(seq2)

    # Use key landmarks
    key_indices = []
    for lm_idx in [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]:
        key_indices.extend([lm_idx * 3, lm_idx * 3 + 1, lm_idx * 3 + 2])

    s1 = seq1[:, key_indices]
    s2 = seq2[:, key_indices]

    # DTW with window constraint
    w = max(window, abs(n - m))  # window must be at least |n-m|
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m, i + w) + 1
        for j in range(j_start, j_end):
            d = np.linalg.norm(s1[i - 1] - s2[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    return cost[n, m] / (n + m)


# =========================================================================
# RECORDING MOTION SIGNS
# =========================================================================
def record_motion_sign():
    """
    Webcam khol ke motion sign record karta hai (10 seconds).
    Hand landmarks ki sequence ko .npy file mein save karta hai.
    """
    sign_name = input("\nKaunsa motion sign record karna hai? (e.g., Hello, Goodbye, Yes, No): ").strip()
    if not sign_name:
        print("[ERROR] Sign ka naam dena zaroori hai!")
        return

    ensure_motion_dir()
    sign_dir = os.path.join(MOTION_DIR, sign_name.lower())
    if not os.path.exists(sign_dir):
        os.makedirs(sign_dir)

    # Count existing recordings
    existing = [f for f in os.listdir(sign_dir) if f.endswith('.npy')]
    rec_num = len(existing) + 1

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera nahi khul raha!")
        return

    print(f"\n[INFO] Camera khul raha hai...")
    print(f"-> '{sign_name}' sign karne ke liye tayyar ho jayein.")
    print(f"-> Camera window par click karein aur 's' dabayein.")
    print(f"-> Recording {RECORDING_DURATION} seconds ki hogi.")

    # Wait for user to get ready
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # Mirror

        # Draw instructions
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(frame, f"Recording: '{sign_name}' (#{rec_num})", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Press 's' to START", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Get Ready - Motion Sign', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    cv2.destroyAllWindows()

    # Countdown 3-2-1
    print("\n[INFO] 3... 2... 1... GO!")
    countdown_start = time.time()
    while time.time() - countdown_start < 3:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        count_num = 3 - int(time.time() - countdown_start)
        # Big countdown number in center
        text = str(max(count_num, 1))
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 4, 5)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
        cv2.imshow('Get Ready!', frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    # Record the motion sign
    print(f"\n[RECORDING] '{sign_name}' ki recording shuru! ({RECORDING_DURATION} seconds)...")
    sequence = []
    start_time = time.time()
    frames_with_hand = 0

    while time.time() - start_time < RECORDING_DURATION:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extract 63 features (21 landmarks x 3 coords)
                row = []
                for landmark in hand_landmarks.landmark:
                    row.extend([landmark.x, landmark.y, landmark.z])
                sequence.append(row)
                frames_with_hand += 1

        # Display UI
        elapsed = time.time() - start_time
        time_left = max(0, RECORDING_DURATION - int(elapsed))
        progress = elapsed / RECORDING_DURATION

        # Progress bar
        bar_width = frame.shape[1] - 40
        bar_x = 20
        bar_y = frame.shape[0] - 30
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + 20), (0, 255, 0), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (255, 255, 255), 1)

        # Status text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        status_color = (0, 255, 0) if results.multi_hand_landmarks else (0, 0, 255)
        hand_status = "Hand Detected" if results.multi_hand_landmarks else "NO HAND!"
        cv2.putText(frame, f"REC [{time_left}s] - {sign_name}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"{hand_status} | Frames: {frames_with_hand}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Red recording dot
        cv2.circle(frame, (frame.shape[1] - 30, 25), 10, (0, 0, 255), -1)

        cv2.imshow('Recording Motion Sign', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Recording cancelled by user.")
            cap.release()
            cv2.destroyAllWindows()
            hands.close()
            return

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    # Validate and save
    if len(sequence) < 10:
        print(f"\n[ERROR] Sirf {len(sequence)} frames mila! Kam se kam 10 frames chahiye.")
        print("-> Apna haath camera ke saamne dikhayein aur dobara try karein.")
        return

    # Normalize and save
    sequence = np.array(sequence)
    normalized = normalize_sequence(sequence)

    save_path = os.path.join(sign_dir, f"recording_{rec_num:03d}.npy")
    np.save(save_path, normalized)

    print(f"\n{'='*50}")
    print(f"  [SUCCESS] Motion sign '{sign_name}' recorded!")
    print(f"  Frames captured: {len(sequence)}")
    print(f"  Normalized to: {TARGET_FRAMES} frames")
    print(f"  Saved: {save_path}")
    print(f"  Total recordings for '{sign_name}': {rec_num}")
    print(f"{'='*50}")

    if rec_num < 3:
        print(f"\n[TIP] Better accuracy ke liye '{sign_name}' ko {3 - rec_num} aur baar record karein!")


# =========================================================================
# PREDICTION
# =========================================================================
def predict_motion_sign():
    """
    Live camera se motion sign capture karke predict karta hai.
    DTW se compare karke sabse closest matching sign batata hai.
    """
    ensure_motion_dir()

    # Check if any signs are recorded
    sign_dirs = [d for d in os.listdir(MOTION_DIR)
                 if os.path.isdir(os.path.join(MOTION_DIR, d))]

    if not sign_dirs:
        print("\n[ERROR] Koi motion sign record nahi hua hai!")
        print("-> Pehle Option 4 se kuch motion signs record karein.")
        return

    # Load all templates
    templates = {}  # {sign_name: [array1, array2, ...]}
    total_templates = 0
    for sign_name in sign_dirs:
        sign_dir = os.path.join(MOTION_DIR, sign_name)
        recordings = [f for f in os.listdir(sign_dir) if f.endswith('.npy')]
        if recordings:
            templates[sign_name] = []
            for rec_file in recordings:
                template = np.load(os.path.join(sign_dir, rec_file))
                templates[sign_name].append(template)
                total_templates += 1

    if not templates:
        print("\n[ERROR] Koi valid recording nahi mili!")
        return

    print(f"\n[INFO] {len(templates)} signs loaded ({total_templates} total templates)")
    print(f"-> Signs: {', '.join(templates.keys())}")

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera nahi khul raha!")
        return

    print(f"\n[INFO] Apna motion sign dikhayein ({PREDICTION_DURATION} seconds)...")
    print("-> Camera window par click karein aur 's' dabayein.")

    # Wait for start
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(frame, "Press 's' to START prediction", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Signs: {', '.join(templates.keys())}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow('Motion Sign Prediction', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    # Capture sequence
    sequence = []
    start_time = time.time()

    while time.time() - start_time < PREDICTION_DURATION:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                row = []
                for landmark in hand_landmarks.landmark:
                    row.extend([landmark.x, landmark.y, landmark.z])
                sequence.append(row)

        # Display
        elapsed = time.time() - start_time
        time_left = max(0, PREDICTION_DURATION - int(elapsed))
        progress = elapsed / PREDICTION_DURATION

        # Progress bar
        bar_width = frame.shape[1] - 40
        bar_x = 20
        bar_y = frame.shape[0] - 30
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + 20), (255, 165, 0), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (255, 255, 255), 1)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(frame, f"Observing... [{time_left}s] | Frames: {len(sequence)}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Motion Sign Prediction', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    if len(sequence) < 10:
        print(f"\n[WARNING] Sirf {len(sequence)} frames capture hue. Haath detect nahi hua.")
        print("-> Camera ke saamne haath dikhayein aur dobara try karein.")
        return

    # Normalize captured sequence
    captured = normalize_sequence(np.array(sequence))

    # Compare with all templates using DTW
    print("\n[INFO] AI comparing kar raha hai...")

    best_sign = None
    best_distance = float('inf')
    all_distances = {}

    for sign_name, sign_templates in templates.items():
        distances = []
        for template in sign_templates:
            dist = dtw_distance_fast(captured, template, window=15)
            distances.append(dist)
        avg_dist = np.mean(distances)
        min_dist = np.min(distances)
        all_distances[sign_name] = min_dist

        if min_dist < best_distance:
            best_distance = min_dist
            best_sign = sign_name

    # Sort by distance
    sorted_signs = sorted(all_distances.items(), key=lambda x: x[1])

    # Display results
    print(f"\n{'='*50}")
    print(f"  MOTION SIGN RESULTS")
    print(f"{'='*50}")

    for i, (sign, dist) in enumerate(sorted_signs):
        marker = " <<< BEST MATCH" if i == 0 else ""
        bar_len = max(1, int(30 * (1 - dist / (sorted_signs[-1][1] + 0.01))))
        bar = "█" * bar_len
        print(f"  {sign:15s} | Distance: {dist:.3f} | {bar}{marker}")

    print(f"{'='*50}")

    if best_distance > CONFIDENCE_THRESHOLD:
        print(f"\n[WARNING] Confidence kam hai (distance: {best_distance:.3f})")
        print(f"-> Best guess: '{best_sign}' but ye sahi nahi ho sakta.")
        print(f"-> Zyada recordings add karein ya sign clearly karein.")
    else:
        confidence_pct = max(0, min(100, (1 - best_distance / CONFIDENCE_THRESHOLD) * 100))
        print(f"\n  >>> PREDICTION: {best_sign.upper()} <<<")
        print(f"  >>> Confidence: {confidence_pct:.1f}% <<<")

    # Speak the result
    display_name = best_sign.replace('_', ' ').title()
    print(f"\n[INFO] Speaking: '{display_name}'")
    try:
        tts = gTTS(text=display_name, lang='en')
        audio_file = "motion_output.mp3"
        tts.save(audio_file)
        play_audio(audio_file)
    except Exception as e:
        print(f"[WARNING] Audio play nahi ho saka: {e}")

    return best_sign


# =========================================================================
# UTILITY FUNCTIONS
# =========================================================================
def list_recorded_signs():
    """Sabhi recorded static aur motion signs ki list dikhata hai."""
    ensure_motion_dir()

    print(f"\n{'='*50}")
    print(f"  RECORDED MOTION SIGNS")
    print(f"{'='*50}")

    sign_dirs = sorted([d for d in os.listdir(MOTION_DIR)
                        if os.path.isdir(os.path.join(MOTION_DIR, d))])

    if not sign_dirs:
        print("  Koi motion sign record nahi hua hai.")
        print("  -> Option 4 se motion signs record karein.")
    else:
        total = 0
        for sign_name in sign_dirs:
            sign_dir = os.path.join(MOTION_DIR, sign_name)
            recordings = [f for f in os.listdir(sign_dir) if f.endswith('.npy')]
            count = len(recordings)
            total += count
            status = "✓ Ready" if count >= 3 else f"⚠ Need {3 - count} more"
            print(f"  {sign_name:20s} | Recordings: {count:3d} | {status}")
        print(f"{'─'*50}")
        print(f"  Total: {len(sign_dirs)} signs, {total} recordings")

    print(f"{'='*50}")


def delete_motion_data():
    """Motion signs ka data delete karta hai."""
    if not os.path.exists(MOTION_DIR):
        print("\n[INFO] Koi motion data maujud nahi hai.")
        return

    sign_dirs = [d for d in os.listdir(MOTION_DIR)
                 if os.path.isdir(os.path.join(MOTION_DIR, d))]

    if not sign_dirs:
        print("\n[INFO] Koi motion sign data nahi hai.")
        return

    print(f"\n[WARNING] Ye sabhi motion signs delete ho jayenge:")
    for s in sign_dirs:
        count = len([f for f in os.listdir(os.path.join(MOTION_DIR, s)) if f.endswith('.npy')])
        print(f"  - {s} ({count} recordings)")

    confirm = input("\nKya aap sure hain? (y/n): ").strip().lower()
    if confirm == 'y':
        import shutil
        shutil.rmtree(MOTION_DIR)
        os.makedirs(MOTION_DIR)
        print("[SUCCESS] Sabhi motion sign data delete ho gaya!")
    else:
        print("[INFO] Delete cancel ho gaya.")


# =========================================================================
# BATCH RECORDING (Record multiple signs in one session)
# =========================================================================
def batch_record_signs():
    """Ek saath multiple signs record karne ka option."""
    signs_to_record = [
        "Hello", "Goodbye", "Please", "Thank You", "Yes", "No",
        "Sorry", "Help", "I Love You", "Stop"
    ]

    print(f"\n{'='*50}")
    print(f"  BATCH RECORDING - Suggested Motion Signs")
    print(f"{'='*50}")

    for i, sign in enumerate(signs_to_record, 1):
        print(f"  {i:2d}. {sign}")

    print(f"\n  0. Custom sign (apna naam daalein)")
    print(f"{'='*50}")

    choice = input("\nKonsa sign record karna hai? (number ya 0 for custom): ").strip()

    if choice == '0':
        record_motion_sign()
    elif choice.isdigit() and 1 <= int(choice) <= len(signs_to_record):
        sign_name = signs_to_record[int(choice) - 1]
        ensure_motion_dir()
        sign_dir = os.path.join(MOTION_DIR, sign_name.lower())
        if not os.path.exists(sign_dir):
            os.makedirs(sign_dir)

        existing = [f for f in os.listdir(sign_dir) if f.endswith('.npy')]
        rec_num = len(existing) + 1

        # Use the recording logic directly
        _record_sign_with_name(sign_name, sign_dir, rec_num)
    else:
        print("[ERROR] Galat option!")


def _record_sign_with_name(sign_name, sign_dir, rec_num):
    """Internal function to record a specific sign."""
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera nahi khul raha!")
        return

    print(f"\n[INFO] '{sign_name}' record karte hain (#{rec_num})")
    print(f"-> Camera window par click karein aur 's' dabayein.")

    # Wait for start
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(frame, f"Recording: '{sign_name}' (#{rec_num})", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Press 's' to START", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Get Ready - Motion Sign', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    cv2.destroyAllWindows()

    # Countdown
    countdown_start = time.time()
    while time.time() - countdown_start < 3:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        count_num = 3 - int(time.time() - countdown_start)
        text = str(max(count_num, 1))
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 4, 5)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
        cv2.imshow('Get Ready!', frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    # Record
    sequence = []
    start_time = time.time()
    frames_with_hand = 0

    while time.time() - start_time < RECORDING_DURATION:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                row = []
                for landmark in hand_landmarks.landmark:
                    row.extend([landmark.x, landmark.y, landmark.z])
                sequence.append(row)
                frames_with_hand += 1

        elapsed = time.time() - start_time
        time_left = max(0, RECORDING_DURATION - int(elapsed))
        progress = elapsed / RECORDING_DURATION

        bar_width = frame.shape[1] - 40
        bar_x = 20
        bar_y = frame.shape[0] - 30
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + 20), (0, 255, 0), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (255, 255, 255), 1)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        status_color = (0, 255, 0) if results.multi_hand_landmarks else (0, 0, 255)
        hand_status = "Hand Detected" if results.multi_hand_landmarks else "NO HAND!"
        cv2.putText(frame, f"REC [{time_left}s] - {sign_name}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"{hand_status} | Frames: {frames_with_hand}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.circle(frame, (frame.shape[1] - 30, 25), 10, (0, 0, 255), -1)

        cv2.imshow('Recording Motion Sign', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Recording cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            hands.close()
            return

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    if len(sequence) < 10:
        print(f"\n[ERROR] Sirf {len(sequence)} frames! Haath camera ke saamne rakhein.")
        return

    sequence = np.array(sequence)
    normalized = normalize_sequence(sequence)

    save_path = os.path.join(sign_dir, f"recording_{rec_num:03d}.npy")
    np.save(save_path, normalized)

    print(f"\n{'='*50}")
    print(f"  [SUCCESS] '{sign_name}' recorded! (#{rec_num})")
    print(f"  Frames: {len(sequence)} -> Normalized: {TARGET_FRAMES}")
    print(f"  Saved: {save_path}")
    print(f"{'='*50}")

    # Ask if user wants to record again
    again = input(f"\n'{sign_name}' ko ek aur baar record karein? (y/n): ").strip().lower()
    if again == 'y':
        _record_sign_with_name(sign_name, sign_dir, rec_num + 1)
