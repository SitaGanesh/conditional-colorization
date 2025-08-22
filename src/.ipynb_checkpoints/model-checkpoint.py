# src/model.py
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class DoubleConv(nn.Module):
    """Double convolution block with batch normalization and ReLU activation."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channel counts must be positive")
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class UNet(nn.Module):
    """
    U-Net architecture for image colorization.
    
    Input: 4 channels (L + hint_mask + hint_ab(2))
    Output: 2 channels (AB)
    """
    
    def __init__(self, in_channels: int = 4, out_channels: int = 2, base: int = 64):
        super().__init__()
        
        # Validate parameters
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if base <= 0:
            raise ValueError(f"base must be positive, got {base}")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base = base
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(base * 8, base * 16)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)  # base*8 + base*8 from skip connection
        
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)  # base*4 + base*4 from skip connection
        
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)  # base*2 + base*2 from skip connection
        
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)  # base + base from skip connection
        
        # Final output layer
        self.final = nn.Conv2d(base, out_channels, kernel_size=1)
        
        logger.info(f"Initialized UNet with in_channels={in_channels}, out_channels={out_channels}, base={base}")
    
    def _pad_to_match(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pad tensor x to match the spatial dimensions of target tensor."""
        diff_h = target.size(2) - x.size(2)
        diff_w = target.size(3) - x.size(3)
        
        if diff_h > 0 or diff_w > 0:
            # Pad with zeros if needed
            pad_h = diff_h // 2
            pad_w = diff_w // 2
            x = nn.functional.pad(x, (pad_w, diff_w - pad_w, pad_h, diff_h - pad_h))
        elif diff_h < 0 or diff_w < 0:
            # Crop if needed
            start_h = (-diff_h) // 2
            start_w = (-diff_w) // 2
            x = x[:, :, start_h:start_h + target.size(2), start_w:start_w + target.size(3)]
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor (B, in_channels, H, W)
            
        Returns:
            Output tensor (B, out_channels, H, W)
        """
        try:
            # Validate input
            if x.dim() != 4:
                raise ValueError(f"Expected 4D input tensor, got {x.dim()}D")
            
            if x.size(1) != self.in_channels:
                raise ValueError(f"Expected {self.in_channels} input channels, got {x.size(1)}")
            
            # Encoder path with skip connections
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            e3 = self.enc3(self.pool2(e2))
            e4 = self.enc4(self.pool3(e3))
            
            # Bottleneck
            bottleneck = self.bottleneck(self.pool4(e4))
            
            # Decoder path with skip connections
            d4 = self.up4(bottleneck)
            d4 = self._pad_to_match(d4, e4)
            d4 = torch.cat([d4, e4], dim=1)
            d4 = self.dec4(d4)
            
            d3 = self.up3(d4)
            d3 = self._pad_to_match(d3, e3)
            d3 = torch.cat([d3, e3], dim=1)
            d3 = self.dec3(d3)
            
            d2 = self.up2(d3)
            d2 = self._pad_to_match(d2, e2)
            d2 = torch.cat([d2, e2], dim=1)
            d2 = self.dec2(d2)
            
            d1 = self.up1(d2)
            d1 = self._pad_to_match(d1, e1)
            d1 = torch.cat([d1, e1], dim=1)
            d1 = self.dec1(d1)
            
            # Final output
            output = self.final(d1)
            
            return output
            
        except Exception as e:
            logger.error(f"Error in UNet forward pass: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'base_channels': self.base
        }