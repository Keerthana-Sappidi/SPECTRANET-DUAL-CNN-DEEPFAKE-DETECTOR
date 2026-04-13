import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from dct.dct_transform import extract_dct_features
from models.spatial_cnn import SpatialCNN
from models.spectral_cnn import SpectralCNN
from models.fusion_model import SpectraNet


# -------------------------------
# Dataset for Evaluation
# -------------------------------
class EvalDataset(Dataset):
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

        rgb_img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, (224, 224))

        if self.transform:
            rgb_img = self.transform(rgb_img)

        dct_img = extract_dct_features(img_path)
        dct_img = torch.tensor(dct_img).unsqueeze(0)

        return rgb_img, dct_img, label


# -------------------------------
# Evaluation Function
# -------------------------------
def evaluate_model():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = EvalDataset(
        real_dir="dataset/real",
        fake_dir="dataset/fake",
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    spatial_model = SpatialCNN().to(DEVICE)
    spectral_model = SpectralCNN().to(DEVICE)

    model = SpectraNet(spatial_model, spectral_model).to(DEVICE)
    model.load_state_dict(torch.load("saved_models/spectranet.pth", map_location=DEVICE))
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for rgb_img, dct_img, label in dataloader:
            rgb_img = rgb_img.to(DEVICE)
            dct_img = dct_img.to(DEVICE)

            output = model(rgb_img, dct_img)
            prediction = (output.item() > 0.5)

            y_true.append(label)
            y_pred.append(int(prediction))

    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1-Score :", f1_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


# -------------------------------
# Run Evaluation
# -------------------------------
if __name__ == "__main__":
    evaluate_model()
