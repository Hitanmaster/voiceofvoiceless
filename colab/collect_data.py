import cv2
import os
import time
import numpy as np
import mediapipe as mp

# =========================================================================
# CONFIGURATION
# =========================================================================
DATA_PATH = os.path.join('motion_signs') 
no_sequences = 30     # How many videos/sequences to collect per sign
sequence_length = 60  # Frames per sequence

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def extract_keypoints(results):
    """Extracts 63 landmarks (21 * 3) from hands and flattens them."""
    if results.multi_hand_landmarks:
        # For simplicity in this basic LSTM, we'll just grab the first hand detected
        # If you want two hands, you'd extract 126 features.
        hand_landmarks = results.multi_hand_landmarks[0]
        lh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
        return lh
    else:
        return np.zeros(21*3)

def collect_data():
    action = input("Enter the sign you want to collect data for (e.g., Hello, Thanks): ").strip()
    if not action: return
    
    action_path = os.path.join(DATA_PATH, action)
    os.makedirs(action_path, exist_ok=True)
    
    # Find next sequence number
    existing_seqs = [int(d) for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
    start_seq = max(existing_seqs) + 1 if existing_seqs else 0

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        for sequence in range(start_seq, start_seq + no_sequences):
            seq_dir = os.path.join(action_path, str(sequence))
            os.makedirs(seq_dir, exist_ok=True)
            
            # Wait screen before each sequence
            for i in range(3, 0, -1):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f'Get ready for "{action}" Video {sequence}/{start_seq + no_sequences - 1}', (15,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Starting in {i}...', (120,200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0, 255), 4, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
                cv2.waitKey(1000)
            
            # Record frames
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                
                # Make detections
                image, results = mediapipe_detection(frame, hands)
                draw_styled_landmarks(image, results)
                
                # Render text
                cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                
                cv2.imshow('OpenCV Feed', image)
                
                # Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(seq_dir, str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Data collection for {action} finished!")

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                  
    results = model.process(image)                 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

if __name__ == '__main__':
    collect_data()
