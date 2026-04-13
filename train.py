import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from dct.dct_transform import extract_dct_features
from models.spatial_cnn import SpatialCNN
from models.spectral_cnn import SpectralCNN
from models.fusion_model import SpectraNet


# -------------------------------
# Custom Dataset
# -------------------------------
class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform

        for img in os.listdir(real_dir):
            self.data.append(os.path.join(real_dir, img))
            self.labels.append(0)

        for img in os.listdir(fake_dir):
            self.data.append(os.path.join(fake_dir, img))
            self.labels.append(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        # RGB image
        rgb_img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, (224, 224))

        if self.transform:
            rgb_img = self.transform(rgb_img)

        # DCT image
        dct_img = extract_dct_features(img_path)
        dct_img = torch.tensor(dct_img).unsqueeze(0)

        return rgb_img, dct_img, torch.tensor(label, dtype=torch.float32)
    # -------------------------------
# Training Configuration
# -------------------------------
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# Image Transformations
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
# Main Training Function
# -------------------------------
def train_model():
    # Dataset paths
    real_dir = "dataset/real"
    fake_dir = "dataset/fake"

    dataset = DeepfakeDataset(
        real_dir=real_dir,
        fake_dir=fake_dir,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Initialize models
    spatial_model = SpatialCNN().to(DEVICE)
    spectral_model = SpectralCNN().to(DEVICE)

    model = SpectraNet(
        spatial_model=spatial_model,
        spectral_model=spectral_model
    ).to(DEVICE)

    # Loss and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for rgb_img, dct_img, label in dataloader:
            rgb_img = rgb_img.to(DEVICE)
            dct_img = dct_img.to(DEVICE)
            label = label.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()

            output = model(rgb_img, dct_img)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}], "
            f"Loss: {epoch_loss / len(dataloader):.4f}"
        )

    # Save trained model
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/spectranet.pth")

    print("Model training completed and saved successfully.")


# -------------------------------
# Run Training
# -------------------------------
if __name__ == "__main__":
    train_model()

