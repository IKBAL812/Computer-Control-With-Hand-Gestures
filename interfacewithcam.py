import os
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import mediapipe as mp
import numpy as np

class CameraApp:
    def __init__(self, root, image_path, bottom_image_path):
        self.root = root
        self.root.title("Control with Gestures App")
        self.root.geometry("1280x720")

        # Load the trained model
        self.model, self.label_map, self.reverse_label_map = self.load_model_and_labels()

        # Initialize MediaPipe Hands
        self.hands, self.mp_drawing = self.initialize_mediapipe()

        # Variables for predictions and presentation control
        self.latest_predictions = []
        self.buttonPressed = False
        self.buttonCounter = 0
        self.buttonDelay = 20

        # Initialize the video capture
        self.cap = self.initialize_camera()

        # Variables for presentation images
        self.imgNumber = 0
        self.gestureThreshold = 300
        self.folderPath = 'Presentation'
        self.pathImages = sorted(os.listdir(self.folderPath), key=len)

        # Set up the GUI
        self.setup_gui(image_path, bottom_image_path)

        # Start updating the camera feed
        self.update_camera()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Create a new window for presentation
        self.presentation_window = tk.Toplevel(self.root)
        self.presentation_window.title("Presentation")
        self.presentation_label = tk.Label(self.presentation_window)
        self.presentation_label.pack()
        self.update_presentation()

    def load_model_and_labels(self):
        model = load_model('hand_sign_model.h5')
        label_map = np.load('label_map.npy', allow_pickle=True).item()
        reverse_label_map = {idx: label for label, idx in label_map.items()}
        return model, label_map, reverse_label_map

    def initialize_mediapipe(self):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils
        return hands, mp_drawing

    def initialize_camera(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 720)  # Camera width
        cap.set(4, 720)  # Camera height
        return cap

    def setup_gui(self, image_path, bottom_image_path):
        # Left frame
        left_frame = tk.Frame(self.root, width=640, height=720)
        left_frame.pack(side="left", padx=10, pady=10)

        # Camera frame
        camera_frame = tk.Frame(left_frame, width=640, height=360)
        camera_frame.pack(side="top", padx=10, pady=10)
        self.camera_label = tk.Label(camera_frame)
        self.camera_label.pack()

        # Bottom image frame
        bottom_image_frame = tk.Frame(left_frame, width=640, height=360)
        bottom_image_frame.pack(side="top", padx=10, pady=10)
        bottom_image_label = tk.Label(bottom_image_frame)
        bottom_image_label.pack()
        bottom_image = self.load_image(bottom_image_path, (640, 360))
        bottom_image_label.config(image=bottom_image)
        bottom_image_label.image = bottom_image

        # Right frame
        image_frame = tk.Frame(self.root, width=640, height=720)
        image_frame.pack(side="right", padx=10, pady=10)
        image_label = tk.Label(image_frame)
        image_label.pack()
        image = self.load_image(image_path, (640, 720))
        image_label.config(image=image)
        image_label.image = image

    def load_image(self, path, size):
        image = Image.open(path)
        image = image.resize(size, Image.LANCZOS)
        return ImageTk.PhotoImage(image)

    def predict_hand_sign(self, frame):
        img = cv2.resize(frame, (128, 128))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        predictions = self.model.predict(img)
        predicted_label = self.reverse_label_map[np.argmax(predictions)]
        confidence = np.max(predictions)
        return predicted_label, confidence

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        return results

    def handle_hand_landmarks(self, frame, hand_landmarks):
        height, width, _ = frame.shape
        x_min = min([lm.x for lm in hand_landmarks.landmark])
        y_min = min([lm.y for lm in hand_landmarks.landmark])
        x_max = max([lm.x for lm in hand_landmarks.landmark])
        y_max = max([lm.y for lm in hand_landmarks.landmark])
        x_min, y_min, x_max, y_max = int(x_min * width), int(y_min * height), int(x_max * width), int(y_max * height)
        cropped_hand = frame[y_min:y_max, x_min:x_max]

        # Center points
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        # If the sign is above the threshold line
        if cy <= self.gestureThreshold:
            if cropped_hand.size != 0:
                hand_sign, confidence = self.predict_hand_sign(cropped_hand)
                if confidence >= 0.7:
                    self.latest_predictions.append((hand_sign, confidence))
                    if len(self.latest_predictions) > 3:
                        self.latest_predictions.pop(0)

                    # Presentation controls with the signs
                    self.control_presentation(hand_sign, confidence)

    def control_presentation(self, hand_sign, confidence):
        if hand_sign == "A" and confidence > 0.8:
            if self.imgNumber < len(self.pathImages) - 1:
                self.imgNumber += 1
                self.buttonPressed = True

        if hand_sign == "I" and confidence > 0.8:
            if self.imgNumber > 0:
                self.imgNumber -= 1
                self.buttonPressed = True

    def update_presentation(self):
        pathFullImage = os.path.join(self.folderPath, self.pathImages[self.imgNumber])
        imgCurrent = cv2.imread(pathFullImage)
        imgCurrent = cv2.resize(imgCurrent, (1080, 720))
        img = Image.fromarray(cv2.cvtColor(imgCurrent, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.presentation_label.imgtk = imgtk
        self.presentation_label.config(image=imgtk)
        self.root.after(100, self.update_presentation)

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            results = self.process_frame(frame)

            if results.multi_hand_landmarks and not self.buttonPressed:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                    self.handle_hand_landmarks(frame, hand_landmarks)

            if self.buttonPressed:
                self.buttonCounter += 1
                if self.buttonCounter > self.buttonDelay:
                    self.buttonCounter = 0
                    self.buttonPressed = False

            self.display_predictions(frame)
            self.draw_threshold_line(frame)
            self.update_image_label(self.camera_label, frame)

        self.root.after(10, self.update_camera)

    def display_predictions(self, frame):
        y_position = 30
        for hand_sign, confidence in reversed(self.latest_predictions):
            cv2.putText(frame, f"{hand_sign} ({confidence * 100:.2f}%)", (20, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            y_position += 30

    def draw_threshold_line(self, frame):
        cv2.line(frame, (0, self.gestureThreshold), (720, self.gestureThreshold), (0, 0, 255), 2)

    def update_image_label(self, label, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.config(image=imgtk)

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root, "Presentation/7.png", "Presentation/8.png")
    root.mainloop()
