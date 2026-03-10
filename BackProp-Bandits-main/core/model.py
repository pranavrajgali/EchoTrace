import torch
import torch.nn as nn
import torchvision.models as models

def build_model(device):
    # Load pretrained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # 1. Freeze initial layers (Layer 1-3)
    for param in model.parameters():
        param.requires_grad = False

    # 2. UNFREEZE Layer 4 for Fine-Tuning
    # This allows the model to adapt its high-level vision to spectrograms
    for param in model.layer4.parameters():
        param.requires_grad = True

    # 3. Replace classifier head
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512), # Increased capacity
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 1) 
    )

    return model.to(device)

def get_loss():
    return nn.BCEWithLogitsLoss()

def get_optimizer(model):
    # Differential Learning Rates: Backbone learns slower than the Head
    return torch.optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 1e-5},
        {'params': model.fc.parameters(), 'lr': 1e-4}
    ])