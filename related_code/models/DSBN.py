import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class DSBN2d(nn.Module):

    def __init__(self, num_features):
        super(DSBN2d, self).__init__()

        self.bn_source = nn.BatchNorm2d(num_features)
        self.bn_target = nn.BatchNorm2d(num_features)

        self.domain_flag = "source"

    
    def forward(self, x):
        if self.domain_flag == "source":
            return self.bn_source(x)
        elif self.domain_flag == "target":
            return self.bn_target(x)
        else:
            raise ValueError("Unknown domain_flag: {self.domain_flag}")


def convert_model_to_dsbn(model):
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_features = child.num_features
            dsbn = DSBN2d(num_features)

            dsbn.bn_source.weight.data.copy_(child.weight.data)
            dsbn.bn_source.bias.data.copy_(child.bias.data)
            dsbn.bn_source.running_mean.data.copy_(child.running_mean.data)
            dsbn.bn_source.running_var.data.copy_(child.running_var.data)

            dsbn.bn_target.weight.data.copy_(child.weight.data)
            dsbn.bn_target.bias.data.copy_(child.bias.data)
            dsbn.bn_target.running_mean.data.copy_(child.running_mean.data)
            dsbn.bn_target.running_var.data.copy_(child.running_var.data)

            setattr(model, name, dsbn)
        else:
            convert_model_to_dsbn(child)
    return model


class DomainDiscriminator(nn.Module):

    def __init__(self, in_feature_dim, num_classes=4, hidden_dim=1024):
        super(DomainDiscriminator, self).__init__()
        self.input_dim = in_feature_dim + num_classes

        self.layer1 = nn.Linear(self.input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.layer3 = nn.Linear(hidden_dim, 1)

    
    def forward(self, feature, softmax_output):
        op_out = torch.cat((feature, softmax_output), dim=1)
        
        x = self.layer1(op_out)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.layer3(x)
        return x
    

class SkinMoleSDA(nn.Module):

    def __init__(self, num_classes=4, backbone="resnet50"):
        super(SkinMoleSDA, self).__init__()

        print(f"Creating {backbone} with DSBN ...")
        
        if backbone == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.feature_dim = 2048
        else:
            raise NotImplementedError("Not ResNet50")
        
        del self.backbone.fc
        self.backbone.fc = nn.Identity()

        self.backbone = convert_model_to_dsbn(self.backbone)

        self.bottleneck = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Linear(512, num_classes)
        self.domain_discriminator = DomainDiscriminator(in_feature_dim=512, num_classes=num_classes)


    def set_domain(self, domain_flag):
        for m in self.modules():
            if isinstance(m, DSBN2d):
                m.domain_flag = domain_flag

    
    def extract_features(self, x, domain_flag):
        self.set_domain(domain_flag)

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        f = torch.flatten(x, 1) # [B, 2048]

        f_bottle = self.bottleneck(f) # [B, 512]
        return f_bottle
    

    def forward(self, x, domain_flag):
        features = self.extract_features(x, domain_flag)
        logits = self.classifier(features)
        return logits, features
    

if __name__ == "__main__":
    net = SkinMoleSDA(num_classes=4)
    img = torch.randn(2, 3, 224, 224)
    
    # Source
    logits_s, feat_s = net(img, domain_flag="source")
    print(f"Source Logits: {logits_s.shape}, Source Features: {feat_s.shape}")
    
    # Target
    logits_t, feat_t = net(img, domain_flag="target")
    print(f"Target Logits: {logits_t.shape}, Target Features: {feat_t.shape}")
    
    softmax_fake = torch.softmax(logits_s, dim=1)
    d_out = net.domain_discriminator(feat_s, softmax_fake)
    print(f"Discriminator Output: {d_out.shape}")