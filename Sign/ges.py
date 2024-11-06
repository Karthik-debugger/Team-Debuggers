import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# A function to calculate the distance between two points
def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) * 2 + (p1[1] - p2[1]) * 2)

# A function to calculate the angle between three points (used for finger angles)
def calculate_angle(p1, p2, p3):
    angle = math.degrees(math.atan2(p3[1] - p2[1], p3[0] - p2[0]) -
                         math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
    return angle if angle >= 0 else angle + 360

# A function to detect if fingers are up
def is_finger_up(landmarks, finger_tip, finger_mcp):
    return landmarks[finger_tip][1] < landmarks[finger_mcp][1]

# Function to detect gestures based on hand landmarks
def detect_gesture(landmarks):
    # "1" Gesture (only index finger up)
    if is_finger_up(landmarks, 8, 5) and not any(is_finger_up(landmarks, i, i-1) for i in [12, 16, 20]):
        return "1"  # Only index finger up
    # "2" Gesture (index and middle fingers up)
    elif is_finger_up(landmarks, 8, 5) and is_finger_up(landmarks, 12, 9) and not any(is_finger_up(landmarks, i, i-1) for i in [16, 20]):
        return "2"  # Index and middle fingers up
    # "Okay" Gesture (thumb and index finger forming a circle)
    elif calculate_distance(landmarks[4], landmarks[8]) < 0.05:
        return "Okay"  # Thumb and index finger close to each other
    # "Fist" Gesture (all fingers down)
    elif not any(is_finger_up(landmarks, i, i-1) for i in [8, 12, 16, 20]):
        return "Fist"  # All fingers down
    # "Open hand" Gesture (all fingers up)
    elif all(is_finger_up(landmarks, i, i-1) for i in [8, 12, 16, 20]):
        return "Open Hand"  # All fingers up
    # "Peace" Gesture (index and middle fingers up)
    elif is_finger_up(landmarks, 8, 5) and is_finger_up(landmarks, 12, 9) and not any(is_finger_up(landmarks, i, i-1) for i in [16, 20]):
        return "Peace"  # Index and middle fingers raised
    # "Rock" Gesture (pinky and index fingers up)
    elif is_finger_up(landmarks, 8, 5) and is_finger_up(landmarks, 20, 17) and not any(is_finger_up(landmarks, i, i-1) for i in [12, 16]):
        return "Rock"  # Pinky and index fingers raised
    # "Thumb Up" Gesture (thumb up)
    elif is_finger_up(landmarks, 4, 3) and not any(is_finger_up(landmarks, i, i-1) for i in [8, 12, 16, 20]):
        return "Thumb Up"  # Thumb raised
    # "Thumb Down" Gesture (thumb down)
    elif not is_finger_up(landmarks, 4, 3) and not any(is_finger_up(landmarks, i, i-1) for i in [8, 12, 16, 20]):
        return "Thumb Down"  # Thumb pointed down
    # "Spider Man" Gesture (index, middle, and ring fingers up)
    elif is_finger_up(landmarks, 8, 5) and is_finger_up(landmarks, 12, 9) and is_finger_up(landmarks, 16, 13) and not any(is_finger_up(landmarks, i, i-1) for i in [4, 20]):
        return "Spider Man"  # Index, middle, and ring fingers raised
    return "Unknown"  # Default to unknown gesture

# Function to process the webcam input and detect gestures
def recognize_gesture():
    # Initialize video capture
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        if not success:
            break

        # Convert the image to RGB (for MediaPipe processing)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        # Default gesture is "Unknown"
        gesture_detected = "Unknown"

        # If landmarks are detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw the hand landmarks on the image
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get landmarks as a list of (x, y) tuples
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

                # Detect the gesture
                gesture_detected = detect_gesture(landmarks)

                # Display the detected gesture on the screen
                cv2.putText(img, gesture_detected, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show the frame with the detected gesture
        cv2.imshow("Gesture Recognition", img)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Run the gesture recognition function
if __name__ == "__main__":
    recognize_gesture()