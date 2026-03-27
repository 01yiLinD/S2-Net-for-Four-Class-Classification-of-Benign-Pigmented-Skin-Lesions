import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNetMoleClassifier(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(ResNetMoleClassifier, self).__init__()

        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


def create_resnet_model(num_classes=4, pretrained=True):
    return ResNetMoleClassifier(num_classes, pretrained)