from typing import Tuple
import torch
import torch.nn as nn


class SmallCNNBackbone(nn.Module):
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(256, feature_dim), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class RCNNDetector(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int = 512):
        super().__init__()
        self.backbone = SmallCNNBackbone(feature_dim)
        self.classifier = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(256, num_classes))
        self.bbox_regressor = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(256, 4))

    def forward(self, roi_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(roi_images)
        return self.classifier(features), self.bbox_regressor(features)
