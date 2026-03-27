import torch
import torch.nn as nn

from utils.MixStyle import MixStyle
from torchvision.models import resnet50, ResNet50_Weights


class MixStyleResNet50(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, mix="crossdomain", pretrained=True):
        super(MixStyleResNet50, self).__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = backbone.fc
        
        self.mixstyle = MixStyle(p=p, alpha=alpha, mix=mix)

    def forward(self, x):
        x = self.stem(x)
        
        x = self.layer1(x)
        x = self.mixstyle(x)
        
        x = self.layer2(x)
        x = self.mixstyle(x)
        
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

def create_resnet_mixstyle_model(num_classes=4, p=0.5, alpha=0.1, mix="crossdomain", pretrained=True):
    print("mix pattern:", mix)
    model = MixStyleResNet50(p=p, alpha=alpha, mix=mix, pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model