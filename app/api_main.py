# app/api_main.py

# --- Path bootstrap ---
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ----------------------

from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
import numpy as np
from PIL import Image
import torch
import logging

from src.model import UNet
from src.utils import rgb_to_lab_norm, lab_norm_to_rgb_uint8, hex_to_ab, build_model_input

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="🎨 Conditional Image Colorizer API",
    description="API for AI-powered image colorization with conditional color hints",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
model_size = 256
model_base = 64

def find_latest_checkpoint() -> Optional[str]:
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
        
        latest_checkpoint = sorted(
            checkpoints,
            key=lambda x: int(x.split("_")[-1].split("."))
        )[-1]
        
        return os.path.join(checkpoint_dir, latest_checkpoint)
        
    except Exception as e:
        logger.error(f"Error finding checkpoint: {e}")
        return None

def load_model():
    """Load the colorization model."""
    global model, model_size, model_base
    
    try:
        checkpoint_path = find_latest_checkpoint()
        
        if checkpoint_path is None:
            logger.warning("No checkpoint found. Model will not be available.")
            return False
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get parameters
        model_base = checkpoint.get("base", model_base)
        model_size = checkpoint.get("size", model_size)
        
        # Create and load model
        model = UNet(in_channels=4, out_channels=2, base=model_base).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        logger.info(f"Model loaded successfully. Size: {model_size}, Base: {model_base}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

# Load model on startup
model_loaded = load_model()

@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    if model_loaded:
        logger.info("✅ API server started successfully with model loaded")
    else:
        logger.warning("⚠️ API server started but no model available")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "🎨 Conditional Image Colorizer API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "device": str(device),
        "endpoints": {
            "health": "/health",
            "colorize": "/colorize (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "model_size": model_size,
        "model_base": model_base
    }

@app.post("/colorize")
async def colorize_image(
    image: UploadFile = File(..., description="Input image file"),
    color_hex: str = Form("#00aaff", description="Hex color for hints (e.g., #ff6b6b)"),
    mask: Optional[UploadFile] = File(None, description="Optional mask image (white = apply color)")
):
    """
    Colorize an image with optional mask and color hints.
    
    - **image**: Input image file (JPEG, PNG, etc.)
    - **color_hex**: Hex color code for colorization hints
    - **mask**: Optional mask image where white areas will be colorized with the specified color
    
    Returns the colorized image as PNG.
    """
    try:
        # Check if model is loaded
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not available. Please train the model first."
            )
        
        # Validate image file
        if not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid image file. Please upload an image."
            )
        
        # Read and preprocess input image
        image_bytes = await image.read()
        
        try:
            input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            input_image = input_image.resize((model_size, model_size), Image.BICUBIC)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error processing input image: {str(e)}"
            )
        
        # Convert to numpy and LAB
        rgb_array = np.array(input_image, dtype=np.uint8)
        L, _ = rgb_to_lab_norm(rgb_array)
        
        # Process mask if provided
        if mask is not None:
            try:
                mask_bytes = await mask.read()
                mask_image = Image.open(io.BytesIO(mask_bytes)).convert("L")
                mask_image = mask_image.resize((model_size, model_size), Image.BICUBIC)
                
                mask_array = np.array(mask_image, dtype=np.float32) / 255.0
                hint_mask = mask_array[..., np.newaxis]  # (H,W,1)
                
                # Create color hints
                ab_color = hex_to_ab(color_hex)
                hint_ab = (np.tile(ab_color.reshape(1, 1, 2), (model_size, model_size, 1)).astype(np.float32)
                          * hint_mask)
                
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error processing mask image: {str(e)}"
                )
        else:
            # No mask provided
            hint_mask = np.zeros((model_size, model_size, 1), dtype=np.float32)
            hint_ab = np.zeros((model_size, model_size, 2), dtype=np.float32)
        
        # Build model input
        try:
            model_input = build_model_input(L, hint_mask, hint_ab)
            model_input = model_input.unsqueeze(0).to(device)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error building model input: {str(e)}"
            )
        
        # Run inference
        try:
            with torch.no_grad():
                pred_ab = model(model_input)
                pred_ab = pred_ab.cpu().numpy().transpose(1, 2, 0)  # (H,W,2)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error during model inference: {str(e)}"
            )
        
        # Convert to RGB
        try:
            rgb_colorized = lab_norm_to_rgb_uint8(L, pred_ab)
            output_image = Image.fromarray(rgb_colorized)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error converting to RGB: {str(e)}"
            )
        
        # Convert to bytes
        try:
            img_buffer = io.BytesIO()
            output_image.save(img_buffer, format="PNG")
            img_buffer.seek(0)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error saving output image: {str(e)}"
            )
        
        logger.info(f"Successfully colorized image with color {color_hex}")
        
        return Response(
            content=img_buffer.getvalue(),
            media_type="image/png",
            headers={
                "Content-Disposition": "inline; filename=colorized_image.png"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in colorize endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available"
        )
    
    try:
        model_info = model.get_model_info()
        model_info.update({
            "device": str(device),
            "input_size": model_size
        })
        return model_info
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model info: {str(e)}"
        )

@app.post("/reload_model")
async def reload_model():
    """Reload the model from the latest checkpoint."""
    try:
        global model_loaded
        model_loaded = load_model()
        
        if model_loaded:
            return {"message": "Model reloaded successfully", "status": "success"}
        else:
            return {"message": "Failed to reload model", "status": "error"}
            
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reloading model: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "app.api_main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )