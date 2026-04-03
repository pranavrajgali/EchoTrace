"""
core/model.py — EchoTrace ResNet50 dual-input architecture
No warm-start from old weights. ImageNet init only.
Freeze: layers 1-3 frozen, layer 4 + fc head trainable.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class EchoTraceResNet(nn.Module):
    """
    Dual-input architecture:
      - x       : (B, 3, 224, 224) 3-channel feature image
      - scalars : (B, 8)           forensic scalar vector
    ResNet50 backbone → 2048-dim embedding
    Concat scalars    → 2056-dim
    FC head           → 2056 → 512 → 1 (logit)
    """
    def __init__(self, num_scalars=8):
        super().__init__()

        # Backbone: ResNet50 with ImageNet weights
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Replace classification head with identity — we extract 2048-dim embeddings
        self.resnet.fc = nn.Identity()

        # Forensic classifier head
        self.fc = nn.Sequential(
            nn.Linear(2048 + num_scalars, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 1),   # raw logit — use BCEWithLogitsLoss
        )

        # Apply freeze strategy immediately on init
        self._apply_freeze()

    def _apply_freeze(self):
        """Freeze layers 1-3, keep layer 4 and fc head trainable."""
        # Step 1: freeze entire backbone
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Step 2: selectively unfreeze layer 4 only
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        # Step 3: fc head is always trainable (new weights, always trains)
        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, x, scalars):
        """
        x       : Tensor (B, 3, 224, 224)
        scalars : Tensor (B, 8)
        returns : Tensor (B, 1) — raw logit
        """
        features = self.resnet(x)                          # (B, 2048)
        features = torch.flatten(features, 1)              # safety flatten
        combined = torch.cat([features, scalars], dim=1)   # (B, 2056)
        return self.fc(combined)                           # (B, 1)


def build_model(device="cpu"):
    """
    Factory function. Returns EchoTraceResNet on the target device.
    Freeze strategy is applied inside __init__ — nothing to do here.
    """
    model = EchoTraceResNet(num_scalars=8)
    return model.to(device)


def get_loss():
    """Binary cross-entropy with logits. Numerically stable."""
    return nn.BCEWithLogitsLoss()


def get_optimizer(model):
    """
    Differential learning rates:
      layer4 backbone : 1e-5  (fine-tuning pre-trained features)
      fc head         : 1e-4  (training new classification head)
    """
    return torch.optim.Adam([
        {"params": model.resnet.layer4.parameters(), "lr": 1e-5},
        {"params": model.fc.parameters(),            "lr": 1e-4},
    ])