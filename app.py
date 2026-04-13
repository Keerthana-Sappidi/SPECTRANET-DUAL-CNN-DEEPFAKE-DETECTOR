import streamlit as st
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import gdown

from dct.dct_transform import extract_dct_features
from models.spatial_cnn import SpatialCNN
from models.spectral_cnn import SpectralCNN
from models.fusion_model import SpectraNet


# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="SpectraNet", layout="centered")
st.title("🧠 SpectraNet – Deepfake Image Detector")


# -------------------------------
# Device
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# Model Download (IMPORTANT 🔥)
# -------------------------------
MODEL_PATH = "saved_models/spectranet.pth"

if not os.path.exists(MODEL_PATH):
    os.makedirs("saved_models", exist_ok=True)
    url = "https://drive.google.com/uc?id=1Tl2ss5TZAMsowPQVyG-pIcWHixPiduLr"
    with st.spinner("Downloading model... please wait ⏳"):
        gdown.download(url, MODEL_PATH, quiet=False)


# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    spatial = SpatialCNN().to(DEVICE)
    spectral = SpectralCNN().to(DEVICE)
    model = SpectraNet(spatial, spectral).to(DEVICE)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model.eval()
    return model


model = load_model()


# -------------------------------
# Image Transform
# -------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload an image (JPG / PNG)", type=["jpg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to OpenCV format
    img_np = np.array(image)
    img_np = cv2.resize(img_np, (224, 224))

    # RGB preprocessing
    rgb_tensor = transform(img_np).unsqueeze(0).to(DEVICE)

    # DCT preprocessing
    temp_path = "temp.jpg"
    image.save(temp_path)
    dct_features = extract_dct_features(temp_path)
    dct_tensor = torch.tensor(dct_features).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Prediction
    with torch.no_grad():
        output = model(rgb_tensor, dct_tensor)
        confidence = output.item()

    if confidence > 0.65:
        st.error(f"⚠️ AI-Generated Image\nConfidence: {confidence:.2f}")
    else:
        st.success(f"✅ Real Image\nConfidence: {1 - confidence:.2f}")