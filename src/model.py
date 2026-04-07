"""
Face Mask Detection Model
Backbone: MobileNetV2 (pretrained on ImageNet)
Fine-tuned for binary classification: with_mask / without_mask
"""

import torch
import torch.nn as nn
from torchvision import models


class MaskDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(MaskDetector, self).__init__()

        # Load pretrained MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=pretrained)

        # Replace the classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def load_model(weights_path, device):
    """Load trained model from .pth file."""
    model = MaskDetector(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model
