from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


def build_resnet_classifier(architecture: str = "resnet50", num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    arch = architecture.lower()
    if arch == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
    elif arch == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
    else:
        raise ValueError("Unsupported architecture. Use 'resnet50' or 'resnet18'.")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model



def build_resnet18_classifier(num_classes: int = 2, pretrained: bool = True):
    return build_resnet_classifier("resnet18", num_classes=num_classes, pretrained=pretrained)
