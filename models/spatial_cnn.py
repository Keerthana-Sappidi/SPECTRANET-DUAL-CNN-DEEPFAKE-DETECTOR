import torch.nn as nn
from torchvision import models

print("spatial_cnn.py LOADED")

class SpatialCNN(nn.Module):
    """
    Spatial CNN using ResNet-50 to extract RGB features
    """

    def __init__(self):
        super(SpatialCNN, self).__init__()

        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)
