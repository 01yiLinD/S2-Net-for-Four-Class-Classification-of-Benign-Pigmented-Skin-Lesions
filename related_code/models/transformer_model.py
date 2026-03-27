import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class ViTMoleClassifier(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(ViTMoleClassifier, self).__init__()

        LOCAL_MODEL_PATH = 'vit'

        # use pretrained Vision Transformer
        if pretrained:
            self.vit = ViTModel.from_pretrained(LOCAL_MODEL_PATH, attn_implementation="eager")
        else:
            config = ViTConfig.from_pretrained(LOCAL_MODEL_PATH)
            if hasattr(config, "attn_implementation"):
                config.attn_implementation = "eager"
            else:
                setattr(config, "attn_implementation", "eager")
            self.vit = ViTModel(config)

        self.classifier = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)


def create_transformer_model(num_classes=4, pretrained=True):
    return ViTMoleClassifier(num_classes, pretrained)