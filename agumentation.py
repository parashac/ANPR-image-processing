import os
import cv2
import random
import numpy as np
import xml.etree.ElementTree as ET
from glob import glob
import shutil

# Define paths
image_dir = "D:/ANPR project/Dataset/last_option/train/images"
label_dir = "D:/ANPR project/Dataset/last_option/train/labels"
aug_image_dir = "D:/ANPR project/Dataset/last_option/train/aug/images"
aug_label_dir = "D:/ANPR project/Dataset/last_option/train/aug/labels"

os.makedirs(aug_image_dir, exist_ok=True)
os.makedirs(aug_label_dir, exist_ok=True)

# Get all image paths
image_paths = glob(os.path.join(image_dir, "*.jpg"))

# Select 50% of the dataset randomly
selected_images = random.sample(image_paths, len(image_paths) // 2)

# Define augmentation distribution
num_images = len(selected_images)
num_gray = int(num_images * 0.75)  # 75% grayscale
num_blur = int(num_images * 0.10)  # 10% blur
num_noise = num_images - (num_gray + num_blur)  # 15% noise

# Shuffle selected images
random.shuffle(selected_images)

# Split into augmentation categories
gray_images = selected_images[:num_gray]
blur_images = selected_images[num_gray:num_gray + num_blur]
noise_images = selected_images[num_gray + num_blur:]

def apply_augmentations(image, aug_type):
    if aug_type == "grayscale":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    elif aug_type == "blur":
        image = cv2.GaussianBlur(image, (3, 3), 0.3)

    elif aug_type == "noise":
        noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)

    return image

def update_xml_label(xml_path, new_filename):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename_tag = root.find("filename")
    if filename_tag is not None:
        filename_tag.text = new_filename

    return tree

# Process images for each augmentation type
for img_path in selected_images:
    img_name = os.path.basename(img_path)
    xml_path = os.path.join(label_dir, img_name.replace(".jpg", ".xml"))

    if not os.path.exists(xml_path):
        print(f"Label missing for {img_name}, skipping...")
        continue

    # Read image
    image = cv2.imread(img_path)

    # Determine augmentation type
    if img_path in gray_images:
        aug_type = "grayscale"
    elif img_path in blur_images:
        aug_type = "blur"
    else:
        aug_type = "noise"

    # Apply augmentation
    aug_image = apply_augmentations(image, aug_type)

    # Generate new filename
    new_img_name = f"aug_{img_name}"
    new_img_path = os.path.join(aug_image_dir, new_img_name)

    # Save augmented image
    cv2.imwrite(new_img_path, aug_image)

    # Update and save XML annotation
    new_xml_path = os.path.join(aug_label_dir, new_img_name.replace(".jpg", ".xml"))
    updated_tree = update_xml_label(xml_path, new_img_name)
    updated_tree.write(new_xml_path)

    print(f"Saved: {new_img_name} ({aug_type}) with annotation")

