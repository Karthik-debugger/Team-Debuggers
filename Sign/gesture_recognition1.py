import cv2
import mediapipe as mp
import math

# List of predefined gestures
gestures = ["Hello", "Yes", "No", "1", "2", "3"]

# A function to calculate the distance between two points
def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Function to detect the gesture
def recognize_gesture():
    # Initialize MediaPipe hands solution
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        if not success:
            break
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        gesture_detected = "Unknown"  # Default to unknown gesture
        
        # If landmarks are detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the landmarks as a list of (x, y) tuples
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                
                # Gesture detection logic (based on hand landmarks)
                gesture_detected = detect_gesture(landmarks)
                
                # Display the detected gesture on the screen
                cv2.putText(img, gesture_detected, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show the frame
        cv2.imshow("Gesture Recognition", img)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Function to detect gesture based on hand landmarks
def detect_gesture(landmarks):
    # Example gesture detection based on certain landmark configurations
    if is_gesture_one(landmarks):
        return "1"
    elif is_gesture_two(landmarks):
        return "2"
    elif is_gesture_three(landmarks):
        return "3"
    elif is_gesture_yes(landmarks):
        return "Yes"
    elif is_gesture_no(landmarks):
        return "No"
    elif is_gesture_hello(landmarks):
        return "Hello"
    return "Unknown"

# Function to check if gesture is "1" (index finger raised)
def is_gesture_one(landmarks):
    return (landmarks[8][1] < landmarks[6][1] and  # Index finger extended
            landmarks[12][1] > landmarks[10][1] and # Middle finger curled
            landmarks[16][1] > landmarks[14][1] and # Ring finger curled
            landmarks[20][1] > landmarks[18][1])    # Little finger curled

# Function to check if gesture is "2" (index and middle finger raised)
def is_gesture_two(landmarks):
    return (landmarks[8][1] < landmarks[6][1] and  # Index finger extended
            landmarks[12][1] < landmarks[10][1] and # Middle finger extended
            landmarks[16][1] > landmarks[14][1] and # Ring finger curled
            landmarks[20][1] > landmarks[18][1])    # Little finger curled

# Function to check if gesture is "3" (index, middle, and ring fingers raised)
def is_gesture_three(landmarks):
    return (landmarks[8][1] < landmarks[6][1] and  # Index finger extended
            landmarks[12][1] < landmarks[10][1] and # Middle finger extended
            landmarks[16][1] < landmarks[14][1] and # Ring finger extended
            landmarks[20][1] > landmarks[18][1])    # Little finger curled

# Function to check if gesture is "Yes" (thumb up)
def is_gesture_yes(landmarks):
    return (landmarks[4][1] < landmarks[2][1] and   # Thumb up
            landmarks[8][1] > landmarks[6][1] and   # Index curled
            landmarks[12][1] > landmarks[10][1] and # Middle curled
            landmarks[16][1] > landmarks[14][1] and # Ring curled
            landmarks[20][1] > landmarks[18][1])    # Little curled

# Function to check if gesture is "No" (thumb down)
def is_gesture_no(landmarks):
    return (landmarks[4][1] > landmarks[2][1] and   # Thumb down
            landmarks[8][1] > landmarks[6][1] and   # Index curled
            landmarks[12][1] > landmarks[10][1] and # Middle curled
            landmarks[16][1] > landmarks[14][1] and # Ring curled
            landmarks[20][1] > landmarks[18][1])    # Little curled

# Function to check if gesture is "Hello" (open hand)
def is_gesture_hello(landmarks):
    return (landmarks[8][1] < landmarks[6][1] and  # Index finger extended
            landmarks[12][1] < landmarks[10][1] and # Middle finger extended
            landmarks[16][1] < landmarks[14][1] and # Ring finger extended
            landmarks[20][1] < landmarks[18][1] and # Little finger extended
            landmarks[4][0] < landmarks[3][0])      # Thumb spread out

if __name__ == "__main__":
    recognize_gesture()
