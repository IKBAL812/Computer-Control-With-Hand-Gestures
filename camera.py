import os
import cv2
from tensorflow.keras.models import load_model
import mediapipe as mp
import numpy as np

# Load the trained model
model = load_model('hand_sign_model.h5')
label_map = np.load('label_map.npy', allow_pickle=True).item()
# Reverse the label map
reverse_label_map = {idx: label for label, idx in label_map.items()}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to predict the hand sign
def predict_hand_sign(frame):
    img = cv2.resize(frame, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_label = reverse_label_map[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_label, confidence

# Initialize list to store latest 3 predictions
latest_predictions = []

# Variables to track button click
button_clicked = False # Quit button

# Function to draw the GUI
def draw_gui(frame):
    # Draw a rectangle for the Quit button
    cv2.rectangle(frame, (530, 410), (620, 460), (0, 0, 255), -1)
    cv2.putText(frame, "Quit", (545, 445), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global button_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the click is within the button region
        if 530 <= x <= 620 and 410 <= y <= 460:
            button_clicked = True

# Open the camera and start prediction
cap = cv2.VideoCapture(0)
cap.set(3, 720) # Camera width
cap.set(4, 720) # Camera height

cv2.namedWindow('Hand Sign Prediction')
cv2.setMouseCallback('Hand Sign Prediction', mouse_callback)

#Variables
gestureThreshold = 300 # for the line

while True:
    # Webcam
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret or button_clicked:
        break

    cv2.line(frame, (0, gestureThreshold), (720, gestureThreshold), (0, 0, 255), 2)

    # Detect and crop the hand
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks :
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calculate bounding box for the hand
            height, width, _ = frame.shape
            x_min = min([lm.x for lm in hand_landmarks.landmark])
            y_min = min([lm.y for lm in hand_landmarks.landmark])
            x_max = max([lm.x for lm in hand_landmarks.landmark])
            y_max = max([lm.y for lm in hand_landmarks.landmark])
            x_min, y_min, x_max, y_max = int(x_min * width), int(y_min * height), int(x_max * width), int(y_max * height)
            cropped_hand = frame[y_min:y_max, x_min:x_max]

            # Center points
            cx = (x_min + x_min) / 2
            cy = (y_min + y_max) / 2

            # If the sign is above the threshold line
            if cy <= gestureThreshold:

                if cropped_hand.size != 0:
                    hand_sign, confidence = predict_hand_sign(cropped_hand)

                # Only display the result if confidence is above 70%
                    if confidence >= 0.7:
                        latest_predictions.append((hand_sign, confidence))
                        if len(latest_predictions) > 3:
                            latest_predictions.pop(0)

    # Display the latest 3 predictions on the frame
    y_position = 30
    for i, (hand_sign, confidence) in enumerate(reversed(latest_predictions)):
        cv2.putText(frame, f"{hand_sign} ({confidence * 100:.2f}%)", (20, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        y_position += 30

    # Draw the GUI
    draw_gui(frame)

    cv2.imshow('Hand Sign Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()