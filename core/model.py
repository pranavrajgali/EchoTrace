# model.py

import torch
import torch.nn as nn
import torchvision.models as models


def build_model(device):
    """
    Builds ResNet50 model for binary classification
    using transfer learning (feature extraction mode).
    """

    # Load pretrained ResNet50
    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V1
    )

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 1)  # Binary output
    )

    model = model.to(device)

    return model


def get_loss():
    return nn.BCEWithLogitsLoss()


def get_optimizer(model, lr=1e-4):
    return torch.optim.Adam(
        model.fc.parameters(),  # Only train classifier head
        lr=lr
    )