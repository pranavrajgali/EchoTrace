import torch
import torch.nn as nn
import torchvision.models as models

class EchoTraceResNet(nn.Module):
    """
    EchoTrace Dual-Input Architecture:
    - Backbone: ResNet50 (Frozen Stages 1-3, Fine-tuned Stage 4).
    - Inputs: 3-channel Feature Image + 8-dim Forensic Scalar Vector.
    - Head: 2056-dim Classifier (2048 ResNet embedding + 8 scalars).
    """
    def __init__(self, num_scalars=8):
        super(EchoTraceResNet, self).__init__()
        
        # Load ResNet50 backbone
        # We initialize with None because weights will be handled via warm_start_new_pipeline
        self.resnet = models.resnet50(weights=None)
        
        # Replace the standard 1000-class ImageNet FC layer with Identity
        # This allows us to extract the raw 2048-dim feature vector from the backbone
        self.resnet.fc = nn.Identity() 

        # Define the Forensic Classifier Head
        # Input: 2048 (from ResNet) + 8 (Scalar features) = 2056
        self.fc = nn.Sequential(
            nn.Linear(2048 + num_scalars, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 512), # Added intermediate layer for better feature mapping
            nn.ReLU(),
            nn.Linear(512, 1)    # Output logit for Binary Cross Entropy
        )

    def forward(self, x, scalars):
        """
        Forward pass for dual-input feature fusion.
        x: Tensor of shape (Batch, 3, 224, 224) - The 3-channel feature image.
        scalars: Tensor of shape (Batch, 8) - Forensic scalar features.
        """
        # 1. Extract 2048-dim latent features from the ResNet backbone
        # We ensure the tensor is flattened to (Batch, 2048)
        spectral_features = self.resnet(x)
        spectral_features = torch.flatten(spectral_features, 1)
        
        # 2. Concatenate spectral features with forensic scalar features
        # Resulting shape: (Batch, 2056)
        combined_features = torch.cat((spectral_features, scalars), dim=1)
        
        # 3. Pass through the classifier head to get the final logit
        return self.fc(combined_features)

def warm_start_new_pipeline(model, checkpoint_path, device="cpu"):
    """
    Hardened weight surgery:
    1. Handles both 'conv1' and 'resnet.conv1' for 3-channel averaging.
    2. Maps weights from legacy 1-channel ResNet50.
    """
    print(f"[*] Commencing weight surgery from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Handle the conv1 averaging first (before or after prefix)
    for key in ['conv1.weight', 'resnet.conv1.weight']:
        if key in checkpoint:
            w = checkpoint[key]
            if w.shape[1] == 1:
                print(f"[!] Averaging {key} for 3-channel input.")
                checkpoint[key] = w.repeat(1, 3, 1, 1) / 3.0
    
    # Prefix mapping loop
    new_state_dict = {}
    for k, v in checkpoint.items():
        # Map raw resnet keys to our wrapper
        if not k.startswith('resnet.') and not k.startswith('fc.'):
            new_state_dict[f"resnet.{k}"] = v
        else:
            new_state_dict[k] = v

    # Load compatible weights (Backbone Stage 1-4)
    # strict=False allows us to load the backbone while ignoring the missing FC head weights
    msg = model.load_state_dict(new_state_dict, strict=False)
    
    print(f"[+] Loaded backbone weights successfully.")
    print(f"[!] Note: Missing keys (expected for new head): {len(msg.missing_keys)}")
    
    return model

def build_model(device="cpu"):
    """Factory function for EchoTrace model initialization."""
    model = EchoTraceResNet()
    return model.to(device)

def get_loss():
    """Returns the Binary Cross Entropy with Logits loss criterion."""
    return nn.BCEWithLogitsLoss()

def get_optimizer(model, lr_backbone=1e-6, lr_head=1e-4):
    """Returns the AdamW optimizer with differential learning rates for better fine-tuning."""
    return torch.optim.AdamW([
        {'params': model.resnet.layer3.parameters(), 'lr': lr_backbone},
        {'params': model.resnet.layer4.parameters(), 'lr': lr_backbone},
        {'params': model.fc.parameters(), 'lr': lr_head}
    ])