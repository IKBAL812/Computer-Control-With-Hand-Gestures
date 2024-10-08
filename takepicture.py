import tkinter as tk
from tkinter import Label, Scrollbar, Canvas, Frame, Button, Entry
import cv2
import os
from PIL import Image, ImageTk
import time

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Camera Application")
        self.geometry("1000x1000")  # Increased window size

        self.is_running = False  # To control the start/stop state
        self.picture_count = 0  # Counter for the number of pictures taken

        # Directory input
        self.dir_label = Label(self, text="Enter directory name:")
        self.dir_label.grid(row=0, column=0, padx=10, pady=10)
        self.dir_entry = Entry(self)
        self.dir_entry.grid(row=0, column=1, padx=10, pady=10)

        # Box 1: Camera feed
        self.camera_frame = tk.Frame(self, width=400, height=300)
        self.camera_frame.grid(row=1, column=0, padx=10, pady=10)
        self.camera_label = Label(self.camera_frame)
        self.camera_label.pack()
        self.start_stop_button = Button(self.camera_frame, text="Start", command=self.toggle_camera)
        self.start_stop_button.pack()

        # Box 2: Picture counter and countdown
        self.picture_frame = tk.Frame(self, width=400, height=300)
        self.picture_frame.grid(row=2, column=0, padx=10, pady=10)
        self.file_name_label = Label(self.picture_frame, text="Filename: ")
        self.file_name_label.pack()
        self.counter_label = Label(self.picture_frame)
        self.counter_label.pack()
        self.picture_count_label = Label(self.picture_frame, text=f"Pictures taken: {self.picture_count}")
        self.picture_count_label.pack()

        # Box 3: Captured images list (using all remaining space)
        self.images_frame = tk.Frame(self, width=800, height=1200)
        self.images_frame.grid(row=1, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")
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

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.counter = 1

        # Quit button
        self.quit_button = Button(self, text="Quit", command=self.on_closing)
        self.quit_button.grid(row=0, column=2, padx=10, pady=10)

        self.image_list = []  # List to keep track of image labels

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
                self.counter = 1

            self.counter_label.config(text=f"Next photo in: {self.counter} seconds")
            self.after(1000, self.update_counter)

    def take_picture(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Mirror the frame horizontally
            directory = self.dir_entry.get() or 'default'
            capture_folder = os.path.join(r'C:\PythonProject\HandImages', directory)
            if not os.path.exists(capture_folder):
                os.makedirs(capture_folder)
            filename = f"{time.strftime('%Y%m%d-%H%M%S')}.png"
            filepath = os.path.join(capture_folder, filename)
            cv2.imwrite(filepath, frame)
            self.display_captured_image(filepath)

            # Update picture count
            self.picture_count += 1
            self.picture_count_label.config(text=f"Pictures taken: {self.picture_count}")

    def display_captured_image(self, filepath):
        image = Image.open(filepath)
        image = image.resize((200, 150), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)

        # Insert image and label at the top of the frame
        image_label = Label(self.scrollable_frame, image=photo)
        image_label.image = photo  # Keep a reference to avoid garbage collection
        filename_label = Label(self.scrollable_frame, text=os.path.basename(filepath))

        # Add new image and filename label at the top of the list
        for widget in self.image_list:
            widget.grid_forget()
        self.image_list.insert(0, image_label)
        self.image_list.insert(1, filename_label)

        for i, widget in enumerate(self.image_list):
            widget.grid(row=i, column=0, pady=5)

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
