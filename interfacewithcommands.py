import os
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import mediapipe as mp
import numpy as np

import ctypes
import pyautogui
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from ctypes.wintypes import HWND, UINT
import subprocess

class CameraApp:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("Control with Gestures App")
        self.root.geometry("1320x750")

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

        # Set up the GUI
        self.setup_gui(image_path)

        # Start updating the camera feed
        self.update_camera()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # Loads the trained model and label mappings.
    def load_model_and_labels(self):
        model = load_model('hand_sign_model.h5')
        label_map = np.load('label_map.npy', allow_pickle=True).item()
        reverse_label_map = {idx: label for label, idx in label_map.items()}
        return model, label_map, reverse_label_map

    # Initializes MediaPipe Hands solution and drawing utilities.
    def initialize_mediapipe(self):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils
        return hands, mp_drawing

    # Initializes the camera for capturing video.
    def initialize_camera(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 720)  # Camera width
        cap.set(4, 720)  # Camera height
        return cap

    # Sets up the GUI with camera and prediction frames.
    def setup_gui(self, image_path):

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

        # Divide bottom image frame into three sections
        self.prediction_frames = []
        for i in range(3):
            frame = tk.Frame(bottom_image_frame, width=213, height=360)
            text_label = tk.Label(frame, text="", fg="gray33", font=("Helvetica", 20), pady=5)
            text_label.pack()
            frame.pack(side="right", padx=20, pady=5)
            image_label = tk.Label(frame)
            image_label.pack()
            self.prediction_frames.append((image_label, text_label))

        # Right frame
        image_frame = tk.Frame(self.root, width=640, height=720)
        image_frame.pack(side="right", padx=10, pady=10)
        image_label = tk.Label(image_frame)
        image_label.pack()
        image = self.load_image(image_path, (640, 720))
        image_label.config(image=image)
        image_label.image = image

    # Loads and resizes an image from the given path.
    def load_image(self, path, size):
        image = Image.open(path)
        image = image.resize(size, Image.LANCZOS)
        return ImageTk.PhotoImage(image)

    # Predicts the hand sign from the given image frame using the trained model.
    def predict_hand_sign(self, frame):
        img = cv2.resize(frame, (128, 128))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        predictions = self.model.predict(img)
        predicted_label = self.reverse_label_map[np.argmax(predictions)]
        confidence = np.max(predictions)
        return predicted_label, confidence

    # Processes a frame to detect hand landmarks using MediaPipe.
    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        return results

    # Handles detected hand landmarks and crops the hand region for prediction.
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

                if confidence > 0.8:
                    match hand_sign:
                        case "MUTE":
                            self.mute()
                            self.latest_predictions.append((cropped_hand, "Mute", confidence))
                            if len(self.latest_predictions) > 3:
                                self.latest_predictions.pop(0)
                            self.display_latest_predictions()
                            self.buttonPressed = True

                        case "UNMUTE":
                            self.unmute()
                            self.latest_predictions.append((cropped_hand, "Unmute", confidence))
                            if len(self.latest_predictions) > 3:
                                self.latest_predictions.pop(0)
                            self.display_latest_predictions()
                            self.buttonPressed = True

                        case "BRIGHTNESS_DOWN":
                            self.brightness_down()
                            self.latest_predictions.append((cropped_hand, "Brightness Down", confidence))
                            if len(self.latest_predictions) > 3:
                                self.latest_predictions.pop(0)
                            self.display_latest_predictions()
                            self.buttonPressed = True

                        case "BRIGHTNESS_UP":
                            self.brightness_up()
                            self.latest_predictions.append((cropped_hand, "Brightness Up", confidence))
                            if len(self.latest_predictions) > 3:
                                self.latest_predictions.pop(0)
                            self.display_latest_predictions()
                            self.buttonPressed = True

                        case "DESKTOP":
                            self.desktop()
                            self.latest_predictions.append((cropped_hand, "Desktop", confidence))
                            if len(self.latest_predictions) > 3:
                                self.latest_predictions.pop(0)
                            self.display_latest_predictions()
                            self.buttonPressed = True

                        case "FULLSCREEN":
                            self.fullscreen()
                            self.latest_predictions.append((cropped_hand, "Fullscreen", confidence))
                            if len(self.latest_predictions) > 3:
                                self.latest_predictions.pop(0)
                            self.display_latest_predictions()
                            self.buttonPressed = True

                        case "NEXT_SLIDE":
                            self.next_slide()
                            self.latest_predictions.append((cropped_hand, "Next Slide", confidence))
                            if len(self.latest_predictions) > 3:
                                self.latest_predictions.pop(0)
                            self.display_latest_predictions()
                            self.buttonPressed = True

                        case "PREVIOUS_SLIDE":
                            self.previous_slide()
                            self.latest_predictions.append((cropped_hand, "Previous Slide", confidence))
                            if len(self.latest_predictions) > 3:
                                self.latest_predictions.pop(0)
                            self.display_latest_predictions()
                            self.buttonPressed = True

                        case "PAUSE_UNPAUSE":
                            self.pause_unpause()
                            self.latest_predictions.append((cropped_hand, "Pause Unpause", confidence))
                            if len(self.latest_predictions) > 3:
                                self.latest_predictions.pop(0)
                            self.display_latest_predictions()
                            self.buttonPressed = True

                        case "VOLUME_UP":
                            self.volume_up()
                            self.latest_predictions.append((cropped_hand, "Volume Up", confidence))
                            if len(self.latest_predictions) > 3:
                                self.latest_predictions.pop(0)
                            self.display_latest_predictions()
                            self.buttonPressed = True

                        case "VOLUME_DOWN":
                            self.volume_down()
                            self.latest_predictions.append((cropped_hand, "Volume Down", confidence))
                            if len(self.latest_predictions) > 3:
                                self.latest_predictions.pop(0)
                            self.display_latest_predictions()
                            self.buttonPressed = True


    # Displays the latest three predictions on the GUI.
    def display_latest_predictions(self):
        for i, (image_label, text_label) in enumerate(self.prediction_frames):
            if i < len(self.latest_predictions):
                cropped_hand, hand_sign, _ = self.latest_predictions[i]
                img = cv2.resize(cropped_hand, (150,150))
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)

                image_label.imgtk = imgtk
                image_label.config(image=imgtk)
                text_label.config(text=hand_sign)
            else:
                image_label.config(image="")
                text_label.config(text="")

    # Captures frames from the camera, processes them, and updates the GUI.
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

            self.draw_threshold_line(frame)
            self.update_image_label(self.camera_label, frame)

        self.root.after(10, self.update_camera)

    # Draws a threshold line on the camera frame.
    def draw_threshold_line(self, frame):
        cv2.line(frame, (0, self.gestureThreshold), (720, self.gestureThreshold), (0, 0, 255), 2)

    # Updates the given label with the provided frame image.
    def update_image_label(self, label, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.config(image=imgtk)

    # COMMAND FUNCTIONS
    def mute(self):
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume.SetMute(1, None)

    def unmute(self):
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume.SetMute(0, None)

    def volume_up(self):
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        current_volume = volume.GetMasterVolumeLevelScalar()
        volume.SetMasterVolumeLevelScalar(min(1.0, current_volume + 0.1), None)

    def volume_down(self):
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        current_volume = volume.GetMasterVolumeLevelScalar()
        volume.SetMasterVolumeLevelScalar(max(0.0, current_volume - 0.1), None)

    def brightness_up(self):
        current_brightness = int(subprocess.check_output(
            "powershell (Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness).CurrentBrightness"))
        new_brightness = min(current_brightness + 10, 100)
        subprocess.run(["powershell",
                        f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{new_brightness})"])

    def brightness_down(self):
        current_brightness = int(subprocess.check_output(
            "powershell (Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness).CurrentBrightness"))
        new_brightness = max(current_brightness - 10, 0)
        subprocess.run(["powershell",
                        f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{new_brightness})"])

    def desktop(self):
        pyautogui.hotkey('winleft', 'd')

    def pause_unpause(self):
        pyautogui.press("playpause")

    def next_slide(self):
        pyautogui.press("right")

    def previous_slide(self):
        pyautogui.press("left")

    def fullscreen(self):
        pyautogui.hotkey('win', 'up')

    # Handles the closing of the application.
    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root, "Image/gesture.png")
    root.mainloop()
