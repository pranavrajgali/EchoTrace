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

def build_model(device):
    """Factory function to initialize the model on the specified device."""
    model = EchoTraceResNet(num_scalars=8)
    return model.to(device)

def warm_start_new_pipeline(model, checkpoint_path, device):
    """
    Performs weight surgery to adapt old model weights to the new architecture.
    
    1. Averages weights for 'conv1' to handle 3 distinct feature channels.
    2. Strips old 'fc' weights to avoid dimension mismatch errors (2048 vs 2056).
    3. Maps backbone keys to the new class structure.
    """
    print(f"[*] Commencing warm start from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Check if we need to modify the first convolution layer
    # Old models used 3 identical mel channels; new models use 3 different feature types
    if 'conv1.weight' in checkpoint:
        old_conv1_weight = checkpoint['conv1.weight']
        # Average the old RGB weights into a single-channel representation, 
        # then expand back to 3 channels to provide a balanced initialization.
        avg_weight = old_conv1_weight.mean(dim=1, keepdim=True)
        checkpoint['conv1.weight'] = avg_weight.expand(-1, 3, -1, -1)
    
    # Remove all keys belonging to the old classifier head
    # The new head has a different input dimension (2056) and layer structure
    keys_to_remove = [k for k in checkpoint.keys() if k.startswith('fc.')]
    for k in keys_to_remove:
        del checkpoint[k]
    
    # Remap backbone keys if the checkpoint was saved without the 'resnet.' prefix
    new_state_dict = {}
    for k, v in checkpoint.items():
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

def get_criterion():
    """Returns the loss function: Binary Cross Entropy with Logits."""
    return nn.BCEWithLogitsLoss()