import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import img_to_array

cnn_model = load_model("D:/ANPR project/code/python/checkpoints/model_epoch_15_.keras")

class_names = {i: str(i) for i in range(10)}  # Digits 0-9
class_names.update({10: 'ba', 11: 'baa', 12: 'bhe', 13: 'c', 14: 'cha', 15: 'di',
                    16: 'ga', 17: 'ha', 18: 'ja', 19: 'jha', 20: 'ka', 21: 'kha',
                    22: 'ko', 23: 'lu', 24: 'ma', 25: 'me', 26: 'naa', 27: 'nya',
                    28: 'pa', 29: 'pra', 30: 'se', 31: 'su', 32: 'ta', 33: 'ya'})


def preprocess_character(image):
    """Prepares a character image for CNN prediction."""
    crop = cv2.resize(image, (64, 64))
    crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)  # Convert to RGB
    crop = crop.astype("float32") / 255.0  # Normalize
    crop = img_to_array(crop)
    crop = np.expand_dims(crop, axis=0)
    return crop


def recognize_character(character_image):
    """Predicts a character using the trained CNN model."""
    processed_img = preprocess_character(character_image)
    prob = cnn_model.predict(processed_img)[0]
    idx = np.argmax(prob)
    return class_names.get(idx, "?")


def preprocess_plate(image):
    """Applies preprocessing to extract characters."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    eroded = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)

    return gray, blurred, binary, eroded


def extract_characters(eroded_image):
    """Finds character contours and extracts bounding boxes."""
    contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    # Filter valid bounding boxes based on size constraints
    valid_boxes = [(x, y, w, h) for (x, y, w, h) in bounding_boxes if 12 < w < 200 and 17 < h < 200]

    return valid_boxes


def sort_characters(valid_boxes):
    """Sorts detected characters based on their row position."""
    if not valid_boxes:
        return [], True  # No valid characters found

    # Sort bounding boxes by y-coordinate first (row-wise sorting)
    valid_boxes.sort(key=lambda b: b[1])

    # Determine row separation using median height
    heights = [y for _, y, _, _ in valid_boxes]
    median_height = np.median(heights)

    # Split into rows
    top_row = [box for box in valid_boxes if box[1] < median_height]
    bottom_row = [box for box in valid_boxes if box[1] >= median_height]

    # Check if it's a single-row plate
    is_single_row = len(bottom_row) == 0 or abs(top_row[-1][1] - bottom_row[0][1]) < 15

    if is_single_row:
        sorted_boxes = sorted(valid_boxes, key=lambda b: b[0])  # Sort left-to-right
        return [sorted_boxes], is_single_row
    else:
        top_row.sort(key=lambda b: b[0])  # Sort top row left-to-right
        bottom_row.sort(key=lambda b: b[0])  # Sort bottom row left-to-right
        return [top_row, bottom_row], is_single_row


def recognize_plate(image_path, debug=False):
    """Main function to recognize a license plate from an image."""
    image = cv2.imread(image_path)
    original_image = image.copy()

    # Preprocess the image
    gray, blurred, binary, eroded = preprocess_plate(image)

    # Extract character bounding boxes
    valid_boxes = extract_characters(eroded)

    if not valid_boxes:
        print("No characters detected!")
        return ""

    # Sort characters into rows
    sorted_rows, is_single_row = sort_characters(valid_boxes)

    # Recognize characters
    plate_text = ""
    character_images = []

    for row in sorted_rows:
        for x, y, w, h in row:
            char_crop = eroded[y:y + h, x:x + w]
            char_crop = cv2.bitwise_not(char_crop)
            resized_char = cv2.resize(char_crop, (64, 64))
            character_images.append(resized_char)

            recognized_char = recognize_character(resized_char)
            plate_text += recognized_char + " " if is_single_row else recognized_char

            # Draw bounding boxes on the original image
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(original_image, recognized_char, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        plate_text += " " if not is_single_row else ""  # Space for row separation

    # Debugging visualization
    if debug:
        plt.figure(figsize=(15, 8))
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original")
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(gray, cmap='gray')
        plt.title("Grayscale")
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.imshow(blurred, cmap='gray')
        plt.title("Gaussian Blur")
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.imshow(binary, cmap='gray')
        plt.title("Binary")
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(eroded, cmap='gray')
        plt.title("Eroded")
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title("Final with Bounding Boxes")
        plt.axis('off')

        plt.show()

        # Show extracted characters
        if character_images:
            plt.figure(figsize=(12, 4))
            for i, char_img in enumerate(character_images):
                plt.subplot(1, len(character_images), i + 1)
                plt.imshow(char_img, cmap='gray')
                plt.title(recognize_character(char_img))
                plt.axis('off')
            plt.suptitle("Extracted Characters and Predictions")
            plt.show()

    print("Recognized License Plate:", plate_text.strip())
    return plate_text.strip()


# Example usage with debugging enabled
license_plate_text = recognize_plate("D:/ANPR project/code/python/image/img_6.png", debug=True)
