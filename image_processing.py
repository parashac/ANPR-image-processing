import os
import cv2
import numpy as np

dataset_dir = "D:/ANPR project/Dataset/character_ocr"

def preprocess_character_image(image_path):
    """
    Preprocess a character image: Resize, Grayscale, Gaussian Blur, Otsu Threshold, and Erosion.

    Parameters:
        image_path (str): Path to the character image.

    Returns:
        preprocessed image (numpy array)
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 64))
    image = cv2.GaussianBlur(image, (5, 5), 0)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    return image

for char_folder in os.listdir(dataset_dir):
    char_folder_path = os.path.join(dataset_dir, char_folder)

    if os.path.isdir(char_folder_path):
        for filename in os.listdir(char_folder_path):
            image_path = os.path.join(char_folder_path, filename)

            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                processed_image = preprocess_character_image(image_path)

                cv2.imwrite(image_path, processed_image)

print("Done")
