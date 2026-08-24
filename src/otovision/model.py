from __future__ import annotations

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_model(num_classes: int, pretrained: bool = True, dropout: float = 0.25):
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    return model
