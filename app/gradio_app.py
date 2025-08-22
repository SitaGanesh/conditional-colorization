# app/gradio_app.py

# --- Path bootstrap ---
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ----------------------

import gradio as gr
import numpy as np
from PIL import Image
import torch
import logging
from typing import Optional

from src.model import UNet
from src.utils import rgb_to_lab_norm, lab_norm_to_rgb_uint8, hex_to_ab, build_model_input

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradioColorizer:
    """Gradio interface for image colorization."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.size = 256
        self.base = 64
        
        self._load_model()
        
        logger.info(f"Gradio colorizer initialized. Device: {self.device}")
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint."""
        checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
        
        if not os.path.isdir(checkpoint_dir):
            return None
        
        try:
            checkpoints = [
                f for f in os.listdir(checkpoint_dir)
                if f.startswith("ckpt_epoch_") and f.endswith(".pth")
            ]
            
            if not checkpoints:
                return None
            
            # Sort by epoch number
            latest_checkpoint = sorted(
                checkpoints,
                key=lambda x: int(x.split("_")[-1].split("."))
            )[-1]
            
            return os.path.join(checkpoint_dir, latest_checkpoint)
            
        except Exception as e:
            logger.error(f"Error finding checkpoint: {e}")
            return None
    
    def _load_model(self):
        """Load the colorization model."""
        try:
            checkpoint_path = self._find_latest_checkpoint()
            
            if checkpoint_path is None:
                logger.warning("No checkpoint found. Model will not be available.")
                return
            
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Get parameters
            self.base = checkpoint.get("base", self.base)
            self.size = checkpoint.get("size", self.size)
            
            # Create and load model
            self.model = UNet(in_channels=4, out_channels=2, base=self.base).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            
            logger.info(f"Model loaded successfully. Size: {self.size}, Base: {self.base}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def predict(self, image_pil: Image.Image, mask_pil: Optional[Image.Image] = None, 
                color_hex: str = "#00aaff") -> Image.Image:
        """
        Predict colorization for an image.
        
        Args:
            image_pil: Input PIL image
            mask_pil: Optional mask PIL image
            color_hex: Hex color string
            
        Returns:
            Colorized PIL image
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Please train the model first.")
            
            if image_pil is None:
                raise ValueError("No input image provided")
            
            # Preprocess image
            img_rgb = image_pil.convert("RGB").resize((self.size, self.size), Image.BICUBIC)
            rgb_array = np.array(img_rgb, dtype=np.uint8)
            
            # Convert to LAB
            L, _ = rgb_to_lab_norm(rgb_array)
            
            # Process mask and hints
            if mask_pil is not None:
                # Process mask
                mask_resized = mask_pil.convert("L").resize((self.size, self.size), Image.BICUBIC)
                mask_array = np.array(mask_resized, dtype=np.float32) / 255.0
                hint_mask = mask_array[..., np.newaxis]  # (H,W,1)
                
                # Create color hints
                ab_color = hex_to_ab(color_hex)
                hint_ab = (np.tile(ab_color.reshape(1, 1, 2), (self.size, self.size, 1)).astype(np.float32)
                          * hint_mask)
            else:
                # No hints
                hint_mask = np.zeros((self.size, self.size, 1), dtype=np.float32)
                hint_ab = np.zeros((self.size, self.size, 2), dtype=np.float32)
            
            # Build model input
            model_input = build_model_input(L, hint_mask, hint_ab)
            model_input = model_input.unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                pred_ab = self.model(model_input)
                pred_ab = pred_ab.cpu().numpy().transpose(1, 2, 0)  # (H,W,2)
            
            # Convert to RGB
            rgb_colorized = lab_norm_to_rgb_uint8(L, pred_ab)
            
            return Image.fromarray(rgb_colorized)
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            # Return original image in case of error
            if image_pil is not None:
                return image_pil.convert("RGB")
            else:
                # Return a blank image
                return Image.new("RGB", (256, 256), (128, 128, 128))

# Create the colorizer instance
colorizer = GradioColorizer()

def create_interface():
    """Create the Gradio interface."""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .gr-button {
        background: linear-gradient(45deg, #007acc, #0099ff);
        border: none;
        color: white;
        font-weight: bold;
    }
    .gr-button:hover {
        background: linear-gradient(45deg, #005c99, #007acc);
    }
    """
    
    with gr.Blocks(title="🎨 Conditional Image Colorizer", css=css) as demo:
        
        # Header
        gr.Markdown(
            """
            # 🎨 Conditional Image Colorizer
            
            Upload a grayscale or color image, optionally paint a mask over regions you want to colorize, 
            pick a color, and let the AI colorize your image!
            
            **Instructions:**
            1. Upload an input image (grayscale or color)
            2. Optionally upload a mask image (white areas will be colorized with the chosen color)
            3. Choose a color using the color picker
            4. Click "Colorize" to generate the result
            """
        )
        
        # Main interface
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input")
                input_image = gr.Image(
                    label="📷 Input Image", 
                    type="pil",
                    height=300
                )
                
                mask_image = gr.Image(
                    label="🎭 Mask (Optional)", 
                    type="pil",
                    height=300
                )
                
                color_picker = gr.ColorPicker(
                    label="🌈 Color Choice", 
                    value="#00aaff",
                    info="Choose the color to apply to masked regions"
                )
                
                colorize_btn = gr.Button(
                    "🎨 Colorize Image", 
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column():
                gr.Markdown("### Output")
                output_image = gr.Image(
                    label="✨ Colorized Result",
                    height=400
                )
        
        # Example section
        gr.Markdown("### 📝 Tips and Examples")
        
        with gr.Row():
            gr.Markdown(
                """
                **Tips for best results:**
                - Use grayscale images for more dramatic colorization
                - Paint white regions in the mask where you want specific colors
                - Try different colors to see various artistic interpretations
                - The AI will automatically colorize areas without masks
                """
            )
        
        # Model status
        model_status = "✅ Model loaded and ready!" if colorizer.model is not None else "❌ Model not found. Please train the model first."
        gr.Markdown(f"**Model Status:** {model_status}")
        
        # Event handler
        colorize_btn.click(
            fn=colorizer.predict,
            inputs=[input_image, mask_image, color_picker],
            outputs=output_image,
            api_name="colorize"
        )
        
        # Examples (if you have sample images)
        try:
            example_dir = os.path.join(PROJECT_ROOT, "examples")
            if os.path.exists(example_dir):
                example_images = [
                    os.path.join(example_dir, f)
                    for f in os.listdir(example_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ][:3]  # Limit to 3 examples
                
                if example_images:
                    gr.Examples(
                        examples=[[img, None, "#ff6b6b"] for img in example_images],
                        inputs=[input_image, mask_image, color_picker],
                        outputs=output_image,
                        fn=colorizer.predict,
                        cache_examples=False
                    )
        except Exception as e:
            logger.warning(f"Could not load examples: {e}")
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    
    # Launch with custom settings
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True,
        quiet=False
    )