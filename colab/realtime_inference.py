import cv2
import numpy as np
import os
import mediapipe as mp

# Try importing tensorflow, but handle gracefully if not installed yet
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARNING] TensorFlow is not installed. Inference will not work until installed (pip install tensorflow).")

# =========================================================================
# CONFIGURATION
# =========================================================================
MODEL_PATH = 'action.h5'
DATA_PATH = os.path.join('motion_signs')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
        return lh
    else:
        return np.zeros(21*3)

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

def run_inference():
    if not TF_AVAILABLE:
        print("\n[ERROR] Please install tensorflow first: pip install tensorflow")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Model file '{MODEL_PATH}' not found!")
        print("Please train the model on Google Colab first and download 'action.h5' to this folder.")
        return

    # Automatically get the actions from the collected data folder
    if not os.path.exists(DATA_PATH):
        print(f"\n[ERROR] Data folder '{DATA_PATH}' not found. Cannot determine vocabulary.")
        return
        
    actions = np.array([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    
    print(f"Loading model '{MODEL_PATH}' for actions: {actions}...")
    model = load_model(MODEL_PATH)

    sequence = []
    sentence = []
    predictions = []
    threshold = 0.8 # Minimum confidence to show a prediction

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            image, results = mediapipe_detection(frame, hands)
            draw_styled_landmarks(image, results)
            
            # Predict Logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-60:] # Keep only last 60 frames
            
            if len(sequence) == 60:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                predictions.append(np.argmax(res))
                
                # Check if the prediction is stable over the last 10 frames
                if np.unique(predictions[-10:])[0] == np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]
            
            # Real-time text rendering
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Real-Time Sign Language Detection', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_inference()
