# src/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class ColorL1Loss(nn.Module):
    """L1 loss for AB color channels."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: {reduction}")
        
        self.reduction = reduction
        self.l1_loss = nn.L1Loss(reduction=reduction)
    
    def forward(self, pred_ab: torch.Tensor, target_ab: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 loss between predicted and target AB channels.
        
        Args:
            pred_ab: Predicted AB channels (B, 2, H, W)
            target_ab: Target AB channels (B, 2, H, W)
            
        Returns:
            L1 loss value
        """
        try:
            # Validate inputs
            if pred_ab.shape != target_ab.shape:
                raise ValueError(f"Shape mismatch: pred {pred_ab.shape} vs target {target_ab.shape}")
            
            if pred_ab.size(1) != 2:
                raise ValueError(f"Expected 2 channels for AB, got {pred_ab.size(1)}")
            
            loss = self.l1_loss(pred_ab, target_ab)
            
            return loss
            
        except Exception as e:
            logger.error(f"Error computing ColorL1Loss: {e}")
            raise

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features."""
    
    def __init__(self, layer_weights: dict = None):
        super().__init__()
        
        # Load pretrained VGG16
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        self.features = vgg.features
        
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Default layer weights
        if layer_weights is None:
            layer_weights = {
                '4': 1.0,   # relu1_2
                '9': 1.0,   # relu2_2
                '16': 1.0,  # relu3_3
                '23': 1.0,  # relu4_3
            }
        
        self.layer_weights = layer_weights
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between predicted and target RGB images.
        
        Args:
            pred_rgb: Predicted RGB images (B, 3, H, W)
            target_rgb: Target RGB images (B, 3, H, W)
            
        Returns:
            Perceptual loss value
        """
        try:
            # Validate inputs
            if pred_rgb.shape != target_rgb.shape:
                raise ValueError(f"Shape mismatch: pred {pred_rgb.shape} vs target {target_rgb.shape}")
            
            if pred_rgb.size(1) != 3:
                raise ValueError(f"Expected 3 channels for RGB, got {pred_rgb.size(1)}")
            
            # Normalize for VGG (ImageNet stats)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred_rgb.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred_rgb.device)
            
            pred_norm = (pred_rgb - mean) / std
            target_norm = (target_rgb - mean) / std
            
            # Extract features
            pred_features = self._extract_features(pred_norm)
            target_features = self._extract_features(target_norm)
            
            # Compute weighted loss
            total_loss = 0.0
            for layer_id, weight in self.layer_weights.items():
                if layer_id in pred_features:
                    loss = self.mse_loss(pred_features[layer_id], target_features[layer_id])
                    total_loss += weight * loss
            
            return total_loss
            
        except Exception as e:
            logger.error(f"Error computing PerceptualLoss: {e}")
            raise
    
    def _extract_features(self, x: torch.Tensor) -> dict:
        """Extract VGG features from specified layers."""
        features = {}
        
        for layer_id, layer in enumerate(self.features):
            x = layer(x)
            layer_name = str(layer_id)
            
            if layer_name in self.layer_weights:
                features[layer_name] = x
        
        return features

class CombinedLoss(nn.Module):
    """Combined loss with L1 and perceptual components."""
    
    def __init__(self, l1_weight: float = 1.0, perceptual_weight: float = 0.1):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        
        self.l1_loss = ColorL1Loss()
        self.perceptual_loss = PerceptualLoss() if perceptual_weight > 0 else None
    
    def forward(self, pred_ab: torch.Tensor, target_ab: torch.Tensor, 
                pred_rgb: torch.Tensor = None, target_rgb: torch.Tensor = None) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred_ab: Predicted AB channels
            target_ab: Target AB channels
            pred_rgb: Predicted RGB images (optional, for perceptual loss)
            target_rgb: Target RGB images (optional, for perceptual loss)
            
        Returns:
            Combined loss value
        """
        try:
            # L1 loss on AB channels
            l1_loss = self.l1_loss(pred_ab, target_ab)
            total_loss = self.l1_weight * l1_loss
            
            # Add perceptual loss if enabled and RGB images provided
            if (self.perceptual_weight > 0 and self.perceptual_loss is not None 
                and pred_rgb is not None and target_rgb is not None):
                perceptual_loss = self.perceptual_loss(pred_rgb, target_rgb)
                total_loss += self.perceptual_weight * perceptual_loss
            
            return total_loss
            
        except Exception as e:
            logger.error(f"Error computing CombinedLoss: {e}")
            raise