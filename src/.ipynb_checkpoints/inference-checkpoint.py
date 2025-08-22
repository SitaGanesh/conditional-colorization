# src/inference.py
import argparse
import numpy as np
from PIL import Image
import torch
import cv2
import os
import logging
from typing import Optional, Tuple

from src.model import UNet
from src.utils import rgb_to_lab_norm, lab_norm_to_rgb_uint8, hex_to_ab, build_model_input, ensure_dir

logger = logging.getLogger(__name__)

class ColorizeInference:
    """Inference class for image colorization."""
    
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        """
        Initialize inference model.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
        """
        self.device = torch.device(device)
        self.model, self.size = self._load_model(checkpoint_path)
        
        logger.info(f"Inference model loaded. Image size: {self.size}, Device: {self.device}")
    
    def _load_model(self, checkpoint_path: str) -> Tuple[UNet, int]:
        """Load model from checkpoint."""
        try:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            # Load checkpoint
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Get model parameters
            base = checkpoint.get("base", 64)
            size = checkpoint.get("size", 256)
            
            # Create model
            model = UNet(in_channels=4, out_channels=2, base=base).to(self.device)
            
            # Load state dict
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            
            logger.info(f"Model loaded successfully. Epoch: {checkpoint.get('epoch', 'unknown')}")
            
            return model, size
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _load_and_preprocess_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess input image."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Load image
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB").resize((self.size, self.size), Image.BICUBIC)
            
            rgb_array = np.array(img_rgb, dtype=np.uint8)
            
            # Convert to LAB
            L, AB = rgb_to_lab_norm(rgb_array)
            
            return L, rgb_array
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
    
    def _load_mask(self, mask_path: str) -> np.ndarray:
        """Load and preprocess mask image."""
        try:
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            
            # Load mask
            with Image.open(mask_path) as mask_img:
                mask_resized = mask_img.convert("L").resize((self.size, self.size), Image.BICUBIC)
            
            # Convert to float array [0, 1]
            mask_array = np.array(mask_resized, dtype=np.float32) / 255.0
            
            # Add channel dimension
            mask_array = mask_array[..., np.newaxis]  # (H,W,1)
            
            return mask_array
            
        except Exception as e:
            logger.error(f"Error loading mask {mask_path}: {e}")
            raise
    
    def _create_hints(self, hint_mask: np.ndarray, color_hex: Optional[str] = None) -> np.ndarray:
        """Create color hints from mask and color."""
        try:
            hint_ab = np.zeros((self.size, self.size, 2), dtype=np.float32)
            
            if color_hex is not None:
                # Convert hex to AB
                ab_color = hex_to_ab(color_hex)
                
                # Apply color to masked regions
                hint_ab = np.tile(ab_color.reshape(1, 1, 2), (self.size, self.size, 1)).astype(np.float32)
                hint_ab = hint_ab * hint_mask
            
            return hint_ab
            
        except Exception as e:
            logger.error(f"Error creating hints: {e}")
            raise
    
    def colorize(self, image_path: str, output_path: str, 
                 mask_path: Optional[str] = None, color_hex: Optional[str] = None) -> str:
        """
        Colorize an image with optional mask and color hints.
        
        Args:
            image_path: Path to input image
            output_path: Path to save colorized image
            mask_path: Optional path to mask image
            color_hex: Optional hex color for hints
            
        Returns:
            Path to saved colorized image
        """
        try:
            logger.info(f"Colorizing image: {image_path}")
            
            # Load and preprocess image
            L, original_rgb = self._load_and_preprocess_image(image_path)
            
            # Process mask and hints
            if mask_path and os.path.exists(mask_path):
                hint_mask = self._load_mask(mask_path)
                hint_ab = self._create_hints(hint_mask, color_hex)
                logger.info(f"Using mask: {mask_path}, color: {color_hex}")
            else:
                hint_mask = np.zeros((self.size, self.size, 1), dtype=np.float32)
                hint_ab = np.zeros((self.size, self.size, 2), dtype=np.float32)
                logger.info("No mask provided, using automatic colorization")
            
            # Build model input
            model_input = build_model_input(L, hint_mask, hint_ab)
            model_input = model_input.unsqueeze(0).to(self.device)  # Add batch dimension
            
            # Run inference
            # (in ColorizeInference.colorize, after model inference)
            with torch.no_grad():
                pred_ab = self.model(model_input)
                # Fix: always select the 0th example (batch=1 shape)
                pred_ab = pred_ab.cpu().numpy().transpose(1, 2, 0)

            
            # Convert to RGB
            rgb_colorized = lab_norm_to_rgb_uint8(L, pred_ab)
            
            # Save result
            ensure_dir(os.path.dirname(output_path))
            colorized_img = Image.fromarray(rgb_colorized)
            colorized_img.save(output_path)
            
            logger.info(f"Colorized image saved: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error during colorization: {e}")
            raise

    def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> Optional[str]:
        """Find the latest checkpoint in the checkpoint directory."""
        try:
            if not os.path.exists(checkpoint_dir):
                return None
            
            checkpoints = [
                f for f in os.listdir(checkpoint_dir)
                if f.startswith("ckpt_epoch_1") and f.endswith(".pth")
            ]
            
            if not checkpoints:
                return None
            
            # FIX: Proper epoch number extraction
            def extract_epoch(filename):
                try:
                    # Extract number from "ckpt_epoch_X.pth"
                    epoch_part = filename.replace("ckpt_epoch_", "").replace(".pth", "")
                    return int(epoch_part)
                except ValueError:
                    return 0
            
            # Sort by epoch number
            latest_checkpoint = max(checkpoints, key=extract_epoch)
            
            return os.path.join(checkpoint_dir, latest_checkpoint)
            
        except Exception as e:
            logger.error(f"Error finding latest checkpoint: {e}")
            return None
def main():
    parser = argparse.ArgumentParser(description='Run image colorization inference')
    
    parser.add_argument('--img', required=True, help='Path to input image')
    parser.add_argument('--ckpt', default=None, help='Path to model checkpoint')
    parser.add_argument('--mask', default=None, help='Path to mask image (white = apply color)')
    parser.add_argument('--color', default=None, help='Hex color like #00aaff')
    parser.add_argument('--output', default='colorized_output.png', help='Output image path')
    parser.add_argument('--device', default=None, help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Find checkpoint if not provided
    if args.ckpt is None:
        args.ckpt = find_latest_checkpoint()
        if args.ckpt is None:
            raise ValueError("No checkpoint found. Please train the model first or provide --ckpt")
    
    logger.info(f"Using checkpoint: {args.ckpt}")
    
    # Create inference object and run colorization
    colorizer = ColorizeInference(args.ckpt, device)
    
    try:
        result_path = colorizer.colorize(
            image_path=args.img,
            output_path=args.output,
            mask_path=args.mask,
            color_hex=args.color
        )
        
        print(f"Colorization complete! Result saved to: {result_path}")
        
    except Exception as e:
        logger.error(f"Colorization failed: {e}")
        raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()