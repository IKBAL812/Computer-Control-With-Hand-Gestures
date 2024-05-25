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
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
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


# Open the camera and start prediction
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Detect and crop the hand
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calculate bounding box for the hand
            height, width, _ = frame.shape
            x_min = min([lm.x for lm in hand_landmarks.landmark])
            y_min = min([lm.y for lm in hand_landmarks.landmark])
            x_max = max([lm.x for lm in hand_landmarks.landmark])
            y_max = max([lm.y for lm in hand_landmarks.landmark])
            x_min, y_min, x_max, y_max = int(x_min * width), int(y_min * height), int(x_max * width), int(
                y_max * height)
            cropped_hand = frame[y_min:y_max, x_min:x_max]

            if cropped_hand.size != 0:
                hand_sign, confidence = predict_hand_sign(cropped_hand)

                # Only display the result if confidence is above 90%
                if confidence >= 0.7:
                    cv2.putText(frame, f"{hand_sign} ({confidence * 100:.2f}%)", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Sign Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
