import cv2
import numpy as np

print("dct_transform.py LOADED")

def extract_dct_features(image_path, size=224):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Image not found or invalid path")

    image = cv2.resize(image, (size, size))
    image = image.astype(np.float32) / 255.0

    dct_image = cv2.dct(image)
    dct_image = np.log(np.abs(dct_image) + 1)

    return dct_image

