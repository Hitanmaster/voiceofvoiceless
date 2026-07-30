import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# This script is meant to be run in Google Colab or on a machine with a powerful GPU.
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

DATA_PATH = 'dataset/motion_signs' # Assuming you unzipped motion_signs.zip to 'dataset' folder

if not os.path.exists(DATA_PATH):
    print(f"ERROR: Cannot find {DATA_PATH}.")
    print("Please make sure you have uploaded and unzipped your motion_signs dataset.")
    exit()

actions = np.array([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
print("Signs detected for training:", actions)

label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    for sequence in os.listdir(action_path):
        seq_path = os.path.join(action_path, sequence)
        if os.path.isdir(seq_path):
            window = []
            # We expect exactly 60 frames (0.npy to 59.npy)
            for frame_num in range(60):
                res = np.load(os.path.join(seq_path, f"{frame_num}.npy"))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print("Training Data shape:", X_train.shape)

# Build LSTM Model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(60, 63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train Model
print("Starting training...")
history = model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))

# Save Model
model.save('action.h5')
print("\n--- TRAINING COMPLETE ---")
print("Model saved as action.h5. You can now download it and use it with realtime_inference.py!")
