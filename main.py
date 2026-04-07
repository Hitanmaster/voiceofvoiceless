import cv2
import csv
import os
import time
import pandas as pd
import numpy as np
import pickle
import platform
import warnings
from gtts import gTTS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mediapipe as mp

# Import motion signs module
import motion_signs

warnings.filterwarnings("ignore")

DATASET_FILE = 'hand_landmarks.csv'
MODEL_FILE = 'sign_language_model.pkl'

# =========================================================================
# SMART INITIALIZATION
# =========================================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# =========================================================================
# HELPER FUNCTIONS
# =========================================================================
def setup_csv():
    """CSV file banata hai aur 63 coordinates ke headers set karta hai."""
    if not os.path.exists(DATASET_FILE) or os.path.getsize(DATASET_FILE) == 0:
        with open(DATASET_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            headers = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']]
            writer.writerow(headers)

def play_audio(filename):
    """OS ke hisaab se audio play karta hai bina extra library ke."""
    if platform.system() == "Windows":
        os.system(f"start {filename}")
    elif platform.system() == "Darwin": # macOS
        os.system(f"afplay {filename}")
    else: # Linux
        os.system(f"xdg-open {filename}")

# =========================================================================
# CORE FUNCTIONS (STATIC SIGNS - LOCAL)
# =========================================================================
def record_data():
    """Webcam on karke frames process karta hai aur CSV mein save karta hai."""
    sign_name = input("\nKaunsa sign record karna hai? (e.g., Hello): ")
    setup_csv()
    
    cap = cv2.VideoCapture(0)
    print("\n[INFO] Camera khul raha hai...")
    print("-> Camera window par click karein aur shuru karne ke liye 's' dabayein.")
    
    # Wait for user to get ready
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.putText(frame, "Press 's' to START recording", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Get Ready', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
            
    cv2.destroyWindow('Get Ready')
    
    print(f"\n[INFO] '{sign_name}' ki recording shuru! (5 seconds)...")
    start_time = time.time()
    count = 0

    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if not ret: break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                row = [sign_name]
                for landmark in hand_landmarks.landmark:
                    row.extend([landmark.x, landmark.y, landmark.z])

                with open(DATASET_FILE, mode='a', newline='') as f:
                    csv.writer(f).writerow(row)
                count += 1

        # Timer dikhane ke liye
        time_left = 5 - int(time.time() - start_time)
        cv2.putText(frame, f"Recording: {time_left}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Recording Data', frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    print(f"[SUCCESS] '{sign_name}' ke {count} frames successfully dataset mein add ho gaye!\n")

def train_model():
    """CSV data padh kar ML Model train karta hai."""
    if not os.path.exists(DATASET_FILE) or os.path.getsize(DATASET_FILE) == 0:
        print("\n[ERROR] Dataset nahi mila ya khali hai! Pehle kuch signs record karein (Option 1).")
        return

    print("\n[INFO] Data read kiya jaa raha hai...")

    try:
        df = pd.read_csv(DATASET_FILE)
        if 'label' not in df.columns:
            print("\n" + "!"*50)
            print(" [CRITICAL ERROR] CSV FILE CORRUPT HO GAYI HAI! ")
            print("!"*50)
            os.remove(DATASET_FILE)
            print("-> Corrupt file delete ho gayi. Kripya Option 1 se data wapas record karein.")
            return
    except Exception as e:
        print(f"\n[ERROR] Data read karne mein dikkat aayi: {e}")
        return

    if len(df['label'].unique()) < 2:
        print("\n[WARNING] Kam se kam 2 alag-alag signs record karein (e.g. 'Hello' aur 'Thanks').")
        return

    print("[INFO] Model train ho raha hai...")
    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"[SUCCESS] Model tayyar hai! Accuracy: {accuracy * 100:.2f}%")

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)

def live_prediction():
    """Local camera khol kar 3 seconds observe karta hai aur predict karta hai."""
    if not os.path.exists(MODEL_FILE):
        print("\n[ERROR] Trained model nahi mila! Pehle model train karein (Option 2).")
        return

    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)

    cap = cv2.VideoCapture(0)
    print("\n[INFO] AI aapko 3 seconds tak observe karega...")
    
    predictions_list = []
    start_time = time.time()

    while time.time() - start_time < 3:
        ret, frame = cap.read()
        if not ret: break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                row = []
                for landmark in hand_landmarks.landmark:
                    row.extend([landmark.x, landmark.y, landmark.z])

                pred = model.predict([row])[0]
                predictions_list.append(pred)
                
                cv2.putText(frame, f"Detecting: {pred}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        time_left = 3 - int(time.time() - start_time)
        cv2.putText(frame, f"Time left: {time_left}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Live Prediction', frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

    if predictions_list:
        final_prediction = max(set(predictions_list), key=predictions_list.count)
        print(f"\n" + "="*40)
        print(f"      AI KI PREDICTION: {final_prediction.upper()}      ")
        print("="*40)

        tts = gTTS(text=final_prediction, lang='en')
        audio_file = "output.mp3"
        tts.save(audio_file)
        play_audio(audio_file)
    else:
        print("\n[WARNING] Camera mein koi haath detect nahi hua. Kripya dobara koshish karein.")

def delete_dataset():
    """Manual override to delete the dataset."""
    if os.path.exists(DATASET_FILE):
        os.remove(DATASET_FILE)
        print(f"\n[SUCCESS] Purana dataset '{DATASET_FILE}' delete ho gaya hai!")
    else:
        print("\n[INFO] Koi dataset file maujud nahi hai. Aap fresh start kar sakte hain.")

# =========================================================================
# MAIN MENU
# =========================================================================
def main():
    while True:
        print("\n" + "="*55)
        print("   SIGN LANGUAGE TO VOICE - AI RECOGNITION SYSTEM")
        print("="*55)
        
        print("\n  ─── Static Signs (Letters/Poses) ───")
        print("  1. Record New Static Sign (5 Seconds)")
        print("  2. Train Static AI Model")
        print("  3. Predict Static Sign & Play Voice")
        
        print("\n  ─── Motion Signs (Hello, Goodbye, etc.) ───")
        print("  4. Record New Motion Sign (10 Seconds)")
        print("  5. Predict Motion Sign & Play Voice")
        print("  6. Quick Record (Pick from list)")
        
        print("\n  ─── Utilities ───")
        print("  7. View All Recorded Signs")
        print("  8. Reset/Delete Static Dataset")
        print("  9. Reset/Delete Motion Signs Data")
        print("  0. Exit")
        
        print("="*55)

        choice = input("\nEnter choice (0-9): ").strip()

        if choice == '1':
            record_data()
        elif choice == '2':
            train_model()
        elif choice == '3':
            live_prediction()
        elif choice == '4':
            motion_signs.record_motion_sign()
        elif choice == '5':
            motion_signs.predict_motion_sign()
        elif choice == '6':
            motion_signs.batch_record_signs()
        elif choice == '7':
            # Show both static and motion signs
            print("\n" + "─"*55)
            print("  STATIC SIGNS:")
            print("─"*55)
            if os.path.exists(DATASET_FILE) and os.path.getsize(DATASET_FILE) > 0:
                try:
                    df = pd.read_csv(DATASET_FILE)
                    if 'label' in df.columns:
                        for label, count in df['label'].value_counts().items():
                            print(f"    {label:20s} | {count} frames")
                    else:
                        print("    Dataset corrupt hai.")
                except Exception:
                    print("    Dataset read nahi ho saka.")
            else:
                print("    Koi static sign record nahi hua.")
            
            motion_signs.list_recorded_signs()
            
        elif choice == '8':
            delete_dataset()
        elif choice == '9':
            motion_signs.delete_motion_data()
        elif choice == '0':
            print("\nProgram band ho raha hai. Goodbye!")
            break
        else:
            print("\nGalat option! Kripya 0 se 9 ke beech chunein.")

if __name__ == "__main__":
    main()