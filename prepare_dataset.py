import os
import shutil
import random

# ==============================
# Kaggle dataset root path
# ==============================
KAGGLE_DATASET_PATH = r"C:\Users\KEERTHANA\Downloads\archive"

# ==============================
# Target SpectraNet dataset paths
# ==============================
TARGET_REAL = "dataset/real"
TARGET_FAKE = "dataset/fake"

# ==============================
# Number of images per class
# ==============================
MAX_IMAGES = 721   # change later if needed

os.makedirs(TARGET_REAL, exist_ok=True)
os.makedirs(TARGET_FAKE, exist_ok=True)

# ==============================
# Recursive image collection
# ==============================
def collect_images(folder):
    image_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                image_files.append(os.path.join(root, file))
    return image_files

# ==============================
# Copy images safely
# ==============================
def copy_images(src_folder, dst_folder, max_images):
    images = collect_images(src_folder)

    if len(images) == 0:
        print(f"❌ No images found in: {src_folder}")
        return

    random.shuffle(images)
    images = images[:max_images]

    for img_path in images:
        filename = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(dst_folder, filename))

    print(f"✅ Copied {len(images)} images to {dst_folder}")

# ==============================
# SOURCE FOLDERS (DOUBLE LEVEL)
# ==============================
REAL_SOURCE = os.path.join(KAGGLE_DATASET_PATH, "RealArt", "RealArt")
FAKE_SOURCE = os.path.join(KAGGLE_DATASET_PATH, "AiArtData", "AiArtData")

# ==============================
# Run dataset preparation
# ==============================
copy_images(REAL_SOURCE, TARGET_REAL, MAX_IMAGES)
copy_images(FAKE_SOURCE, TARGET_FAKE, MAX_IMAGES)

print("🎉 Dataset preparation completed successfully.")
