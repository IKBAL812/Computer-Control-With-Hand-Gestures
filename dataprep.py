import os
import cv2
import numpy as np
import random

from tensorflow.keras.utils import to_categorical
import mediapipe as mp
from matplotlib import pyplot as plt

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils


def detect_and_draw_hand(image):
    results = hands.process(image)
    if results and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            x_min = min([lm.x for lm in hand_landmarks.landmark])
            y_min = min([lm.y for lm in hand_landmarks.landmark])
            x_max = max([lm.x for lm in hand_landmarks.landmark])
            y_max = max([lm.y for lm in hand_landmarks.landmark])
            x_min, y_min, x_max, y_max = int(x_min * image.shape[1]), int(y_min * image.shape[0]), int(x_max * image.shape[1]), int(y_max * image.shape[0])
            cropped_hand = image[y_min:y_max, x_min:x_max]
        return cropped_hand
    return None

# Load images and labels with hand detection and drawing


def load_images_from_folder(base_folder):
    i = 1
    images = []
    labels = []
    class_names = os.listdir(base_folder)

    for class_name in class_names:
        class_folder = os.path.join(base_folder, class_name)
        if os.path.isdir(class_folder):
            for filename in os.listdir(class_folder):
                img_path = os.path.join(class_folder, filename)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img is not None:
                    img_with_landmarks = detect_and_draw_hand(img)
                    if img_with_landmarks is not None:
                        try:
                            img_with_landmarks = cv2.resize(img_with_landmarks, (128, 128))  # Resize image to 128x128
                        except cv2.error as e:
                            img = cv2.resize(img, (128, 128))  # Resize image to 128x128
                            images.append(img)
                            labels.append(class_name)
                            print("Image ", i, " done. (In folder ", class_name, " )")
                            i = i + 1
                            continue
                        images.append(img_with_landmarks)
                        labels.append(class_name)
                        print("Image ", i, " done. (In folder ", class_name, " )")
                        i = i+1
                    else:
                        img = cv2.resize(img, (128, 128))  # Resize image to 128x128
                        images.append(img)
                        labels.append(class_name)
                        print("Image ", i, " done. (In folder ", class_name, " )")
                        i = i + 1
                else:
                    print(f"Warning: Could not read image {img_path}")

    return np.array(images), np.array(labels)


images, labels = load_images_from_folder(r'C:\PythonProject\HandImages')
label_map = {label: idx for idx, label in enumerate(np.unique(labels))}
labels = np.array([label_map[label] for label in labels])
print("Mediapipe drawing complete, moving on to data augment.")


def augment_images(images_tmp):
    augmented_images_tmp = []
    i = 1
    option = random.randint(1, 3)
    for img in images_tmp:
        if option == 1:
            # Blurred image
            blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
            blurred_img = blurred_img.astype('float32') / 255.0
            augmented_images_tmp.append(blurred_img)
            print("Image", i, "blurred and added")
            i = i+1
            option = random.randint(1, 3)
            continue

        elif option == 2:
            # Sharpened image
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened_img = cv2.filter2D(img, -1, kernel)
            sharpened_img = sharpened_img.astype('float32') / 255.0
            augmented_images_tmp.append(sharpened_img)
            print("Image", i, "sharpened and added")
            i = i+1
            option = random.randint(1, 3)
            continue

        elif option == 3:
            # Original image
            img = img.astype('float32') / 255.0
            augmented_images_tmp.append(img)
            print("Image", i, "normally added")
            i = i+1
            option = random.randint(1, 3)
            continue

    return np.array(augmented_images_tmp)


augmented_images = augment_images(images)


print("Data augment complete, moving on to data shuffling")

# Shuffle the data
shuffled_indices = np.random.permutation(len(augmented_images))
augmented_images = augmented_images[shuffled_indices]
augmented_labels = labels[shuffled_indices]

print("Shuffling complete, moving on to label converting")

# Convert labels to categorical
augmented_labels = to_categorical(augmented_labels, num_classes=len(label_map))

print("Process complete, downloading arrays...")

np.save('augmented_images.npy', augmented_images)
np.save('augmented_labels.npy', augmented_labels)
np.save('label_map.npy', label_map)

print("Data prep complete!")