# GESTURE CONTROLLED INVERTING FILTER :) #
# this python script when run, opens a monitor window and 
# uses OpenCV and MediaPipe to track the users hand gestures. 
# when hands are open (or when all fingers are up), 
# the image captured on the monitor inverts each color pixel 
# allowing for live visual effects.
# PRESS 'Q' TO EXIT!!! (case insensitive)
import cv2
import mediapipe as mp
import numpy as np

# initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# open webcam (0 for default MacBook camera)
cap = cv2.VideoCapture(0)

# functions
# determine which fingers are up based on hand landmarks
def get_finger_states(landmarks):
    tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    base = [2, 6, 10, 14, 18]  # Lower joint of each finger
    fingers_up = []
    
    for tip, base_joint in zip(tips, base):
        if landmarks[tip].y < landmarks[base_joint].y:  # finger is up
            fingers_up.append(1)
        else:
            fingers_up.append(0)

    return fingers_up

# detect whether an open hand is shown
def detect_gesture(fingers_up):
    if (fingers_up == [1, 1, 1, 1, 1] or fingers_up == [0, 1, 1, 1, 1]):  
        return "open hand"
    
    return None  # no valid gesture detected


# main code
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    result = hands.process(rgb_frame)

    gesture = None  # Default: No gesture detected

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   landmark_drawing_spec=mp_draw.DrawingSpec(color=(255, 255, 255), thickness=3),
                                   connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3))

            # Convert landmarks to a usable format
            landmarks = [lm for lm in hand_landmarks.landmark]

            # Detect finger states
            fingers_up = get_finger_states(landmarks)

            # Detect gesture
            gesture = detect_gesture(fingers_up)

    # apply color inversion if open hand 
    if gesture == "open hand":
        frame = cv2.bitwise_not(frame)  # invert colors
    else:
        pass  # Keep original colors

    # display gesture on screen
    if gesture:
        cv2.putText(frame, gesture, (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                                                                #    size    color     thickness

    # show real-time feed
    cv2.imshow("gesture controlled filter", frame)
    

    # press 'q' to exit (case insensitive)
    if chr(cv2.waitKey(1) & 0xFF).lower() == 'q':
        break

cap.release()
cv2.destroyAllWindows()
