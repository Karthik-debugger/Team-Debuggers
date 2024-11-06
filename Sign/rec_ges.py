import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('C:\Users\DELL\Downloads\main\sign_mnist_train.csv')

# Prepare dataset for training
X = data.drop('label', axis=1).values  # Features (pixel values)
y = data['label'].values  # Labels (gestures)

# Normalize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the model and scaler for later use
with open('gesture_model.pkl', 'wb') as f:
    pickle.dump((model, scaler), f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Load model and scaler
with open('gesture_model.pkl', 'rb') as f:
    model, scaler = pickle.load(f)

# Function to detect gestures
def recognize_gesture():
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        gesture_detected = "Unknown"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Flatten the landmarks
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                flattened_landmarks = np.array(landmarks).flatten()

                # Normalize the landmarks using the scaler
                normalized_landmarks = scaler.transform([flattened_landmarks])

                # Predict the gesture
                gesture_detected = model.predict(normalized_landmarks)[0]

                # Display the predicted gesture
                cv2.putText(img, str(gesture_detected), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Gesture Recognition", img)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_gesture()
