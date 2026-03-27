import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNetFeatureExtractor(nn.Module):
    
    def __init__(self, pretrained=True):
        super(ResNetFeatureExtractor).__init__()

        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ClassifierHead(nn.Module):

    def __init__(self, in_features=2048, out_features=2):
        super(ClassifierHead, self).__init__()
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, out_features)
        )

    def forward(self, x):
        return self.head(x)


class ResNetHierarchicalClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetHierarchicalClassifier, self).__init__()
        
        self.feature_extractor = ResNetFeatureExtractor(pretrained=pretrained)
        in_features = 2048

        self.coarse_head = ClassifierHead(in_features, out_features=2)
        
        self.fine_head_A = ClassifierHead(in_features, out_features=2)

        self.fine_head_B = ClassifierHead(in_features, out_features=2)

    def forward(self, x, head_type='coarse'):
        
        features = self.feature_extractor(x)
        
        if head_type == 'coarse':
            return self.coarse_head(features)
        elif head_type == 'fine_A':
            return self.fine_head_A(features)
        elif head_type == 'fine_B':
            return self.fine_head_B(features)
        else:
            return self.coarse_head(features)


def create_resnet_harr_model(model_type, **kwargs):
    return ResNetHierarchicalClassifier(pretrained=kwargs.get('pretrained', True))