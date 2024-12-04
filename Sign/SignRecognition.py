import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import mediapipe as mp

# Constants
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 26  # Assuming 26 letters in sign language alphabet

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def collect_data():                                                        # ----- Data collection -----
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    cap = cv2.VideoCapture(0)
    for i in range(NUM_CLASSES):
        if not os.path.exists(f'dataset/{chr(65+i)}'):
            os.makedirs(f'dataset/{chr(65+i)}')
        
        print(f"Collecting data for {chr(65+i)}. Press 'c' to capture (30 images). Press 'q' to quit.")
        count = 0
        while count < 30:
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)
            if key == ord('c'):
                cv2.imwrite(f'dataset/{chr(65+i)}/{count}.jpg', frame)
                count += 1
            elif key == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

def preprocess_data():
    data = []
    labels = []
    for i in range(NUM_CLASSES):
        path = f'dataset/{chr(65+i)}'
        for img_name in os.listdir(path):
            img = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(i)
    return np.array(data), np.array(labels)

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model():                                                                        # train model
    data, labels = preprocess_data()
    data = data.reshape(data.shape[0], IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255
    labels = to_categorical(labels, NUM_CLASSES)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    model = create_model()
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test))
    model.save('sign_language_model.h5')


def real_time_recognition():                                                                 # real time recognition
    model = tf.keras.models.load_model('sign_language_model.h5')
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract hand region
                h, w, _ = frame.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)

                # Add padding
                padding = 20
                hand_img = frame[max(0, y_min-padding):min(h, y_max+padding), 
                                 max(0, x_min-padding):min(w, x_max+padding)]

                if hand_img.size != 0:
                    hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                    hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
                    hand_img = hand_img.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255

                    prediction = model.predict(hand_img)
                    predicted_class = chr(65 + np.argmax(prediction))
                    cv2.putText(frame, f"Predicted: {predicted_class}", (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Sign Language Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Uncomment the following lines as needed
    #collect_data()
    #train_model()
    real_time_recognition()