import torch
import torch.nn as nn

print("fusion_model.py LOADED")

class SpectraNet(nn.Module):
    """
    Dual-stream fusion model combining spatial and spectral features
    """

    def __init__(self, spatial_model, spectral_model):
        super(SpectraNet, self).__init__()

        self.spatial_model = spatial_model
        self.spectral_model = spectral_model

        # Fusion + classification layers
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_input, dct_input):
        spatial_features = self.spatial_model(rgb_input)
        spectral_features = self.spectral_model(dct_input)

        fused_features = torch.cat(
            (spatial_features, spectral_features), dim=1
        )

        output = self.classifier(fused_features)
        return output
