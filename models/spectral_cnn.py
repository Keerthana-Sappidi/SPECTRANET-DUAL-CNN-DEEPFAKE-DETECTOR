import torch.nn as nn
from torchvision import models

print("spectral_cnn.py LOADED")

class SpectralCNN(nn.Module):
    """
    Spectral CNN using ResNet-18 to extract DCT-based frequency features
    """

    def __init__(self):
        super(SpectralCNN, self).__init__()

        # Load pretrained ResNet-18
        self.backbone = models.resnet18(pretrained=True)

        # Modify first conv layer to accept 1-channel input
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Remove classification layer
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)
