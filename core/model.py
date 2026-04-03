import torch
import torch.nn as nn
import torchvision.models as models

class EchoTraceResNet(nn.Module):
    """
    EchoTrace Dual-Input Architecture:
    - Backbone: ResNet50 (Frozen Stages 1-2, Fine-tuned Stage 3-4).
    - Inputs: 3-channel Feature Image + 8-dim Forensic Scalar Vector.
    - Head: 2056-dim Classifier (2048 ResNet embedding + 8 scalars).
    """
    def __init__(self, num_scalars=8):
        super(EchoTraceResNet, self).__init__()
        
        # Load ResNet50 with ImageNet pre-trained weights
        self.resnet = models.resnet50(weights='IMAGENET1K_V1')
        
        # Replace the standard 1000-class ImageNet FC layer with Identity
        # This allows us to extract the raw 2048-dim feature vector from the backbone
        self.resnet.fc = nn.Identity() 

        # Define the Forensic Classifier Head
        # Input: 2048 (from ResNet) + 8 (Scalar features) = 2056
        self.fc = nn.Sequential(
            nn.Linear(2048 + num_scalars, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)    # Output logit for Binary Cross Entropy
        )

    def forward(self, x, scalars):
        """
        Forward pass for dual-input feature fusion.
        x: Tensor of shape (Batch, 3, 224, 224) - The 3-channel feature image.
        scalars: Tensor of shape (Batch, 8) - Forensic scalar features.
        """
        spectral_features = self.resnet(x)
        spectral_features = torch.flatten(spectral_features, 1)
        combined_features = torch.cat((spectral_features, scalars), dim=1)
        return self.fc(combined_features)


def build_model(device="cpu"):
    """Factory function for EchoTrace model initialization with explicit layer freezing."""
    model = EchoTraceResNet()
    
    # Freeze layers 1 and 2 (early features from ImageNet)
    for param in model.resnet.layer1.parameters():
        param.requires_grad = False
    for param in model.resnet.layer2.parameters():
        param.requires_grad = False
    
    return model.to(device)


def get_loss():
    """Returns the Binary Cross Entropy with Logits loss criterion."""
    return nn.BCEWithLogitsLoss()


def get_optimizer(model):
    """Returns the Adam optimizer with differential learning rates for fine-tuning."""
    return torch.optim.Adam([
        {'params': model.resnet.layer3.parameters(), 'lr': 1e-6},
        {'params': model.resnet.layer4.parameters(), 'lr': 1e-5},
        {'params': model.fc.parameters(), 'lr': 1e-4}
    ])