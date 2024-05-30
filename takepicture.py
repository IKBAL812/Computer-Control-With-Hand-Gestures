import tkinter as tk
from tkinter import Label, PhotoImage, Scrollbar, Canvas, Frame, Button
import cv2
import os
from PIL import Image, ImageTk
import time

# Set up directories
capture_folder = r'C:\PythonProject\HandImages\deneme'
if not os.path.exists(capture_folder):
    os.makedirs(capture_folder)


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Camera Application")
        self.geometry("1000x1000")  # Increased window size

        self.is_running = False  # To control the start/stop state

        # Box 1: Camera feed
        self.camera_frame = tk.Frame(self, width=400, height=300)
        self.camera_frame.grid(row=0, column=0, padx=10, pady=10)
        self.camera_label = Label(self.camera_frame)
        self.camera_label.pack()
        self.start_stop_button = Button(self.camera_frame, text="Start", command=self.toggle_camera)
        self.start_stop_button.pack()

        # Box 2: Picture, filename and countdown
        self.picture_frame = tk.Frame(self, width=400, height=300)
        self.picture_frame.grid(row=1, column=0, padx=10, pady=10)
        self.picture_label = Label(self.picture_frame)
        self.picture_label.pack()
        self.file_name_label = Label(self.picture_frame, text="Filename: Deneme")
        self.file_name_label.pack()
        self.counter_label = Label(self.picture_frame)
        self.counter_label.pack()

        # Box 3: Captured images list (using all remaining space)
        self.images_frame = tk.Frame(self, width=800, height=1200)
        self.images_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.images_canvas = Canvas(self.images_frame)
        self.scrollbar = Scrollbar(self.images_frame, orient="vertical", command=self.images_canvas.yview)
        self.images_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side=tk.RIGHT, fill="y")
        self.images_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollable_frame = Frame(self.images_canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.images_canvas.configure(
                scrollregion=self.images_canvas.bbox("all")
            )
        )
        self.images_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Load initial picture
        self.load_initial_picture()

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.counter = 2

        # Start the update loop
        self.update_camera()

        # Quit button
        self.quit_button = Button(self, text="Quit", command=self.on_closing)
        self.quit_button.grid(row=2, column=1, padx=10, pady=10, sticky="e")

    def load_initial_picture(self):
        image_path = r"C:\PythonProject\HandImages\5\image_1681396325.934546.jpg"  # Path to the predefined picture
        image = Image.open(image_path)
        image = image.resize((400, 300), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.picture_label.config(image=photo)
        self.picture_label.image = photo
        self.file_name_label.config(text=f"Filename: {image_path}")

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Mirror the frame horizontally
            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2_image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)

        if self.is_running:
            self.after(10, self.update_camera)

    def update_counter(self):
        if self.is_running:
            self.counter -= 1
            if self.counter == 0:
                self.take_picture()
                self.counter = 2

            self.counter_label.config(text=f"Next photo in: {self.counter} seconds")
            self.after(1000, self.update_counter)

    def take_picture(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Mirror the frame horizontally
            filename = f"{time.strftime('%Y%m%d-%H%M%S')}.png"
            filepath = os.path.join(capture_folder, filename)
            cv2.imwrite(filepath, frame)
            self.display_captured_image(filepath)

    def display_captured_image(self, filepath):
        image = Image.open(filepath)
        image = image.resize((200, 150), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)

        image_label = Label(self.scrollable_frame, image=photo)
        image_label.image = photo  # Keep a reference to avoid garbage collection
        image_label.pack()

        filename_label = Label(self.scrollable_frame, text=os.path.basename(filepath))
        filename_label.pack()

    def toggle_camera(self):
        if self.is_running:
            self.is_running = False
            self.start_stop_button.config(text="Start")
        else:
            self.is_running = True
            self.start_stop_button.config(text="Stop")
            self.update_camera()
            self.update_counter()

    def on_closing(self):
        self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = Application()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
