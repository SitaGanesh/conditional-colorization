# src/train.py
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from tqdm import tqdm
import logging
import time
from typing import Optional

from src.dataset import ColorizeDataset
from src.model import UNet
from src.losses import ColorL1Loss, CombinedLoss
from src.utils import lab_norm_to_rgb_uint8, save_rgb, ensure_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Trainer:
    """Training manager for the colorization model."""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = config['device']
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Create directories
        ensure_dir(config['checkpoint_dir'])
        ensure_dir(config['sample_dir'])
        
        # Initialize dataset and dataloader
        self._setup_data()
        
        # Initialize model
        self._setup_model()
        
        # Initialize optimizer and scheduler
        self._setup_training()
    
    def _setup_data(self):
        """Setup datasets and dataloaders."""
        try:
            # Training dataset
            self.train_dataset = ColorizeDataset(
                self.config['data_root'],
                size=self.config['size'],
                max_hints=self.config['max_hints']
            )
            
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config['num_workers'],
                pin_memory=self.config['pin_memory'],
                drop_last=True
            )
            
            # Validation dataset (optional)
            self.val_loader = None
            if self.config['val_root'] and os.path.isdir(self.config['val_root']):
                self.val_dataset = ColorizeDataset(
                    self.config['val_root'],
                    size=self.config['size'],
                    max_hints=self.config['max_hints']
                )
                
                self.val_loader = DataLoader(
                    self.val_dataset,
                    batch_size=self.config['batch_size'],
                    shuffle=False,
                    num_workers=self.config['num_workers'],
                    pin_memory=self.config['pin_memory']
                )
                
                logger.info(f"Validation dataset: {len(self.val_dataset)} images")
            
            logger.info(f"Training dataset: {len(self.train_dataset)} images")
            
        except Exception as e:
            logger.error(f"Error setting up data: {e}")
            raise
    
    def _setup_model(self):
        """Setup model."""
        try:
            self.model = UNet(
                in_channels=4,
                out_channels=2,
                base=self.config['base']
            ).to(self.device)
            
            # Print model info
            model_info = self.model.get_model_info()
            logger.info(f"Model initialized: {model_info}")
            
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            raise
    
    def _setup_training(self):
        """Setup optimizer, scheduler, and loss function."""
        try:
            # Optimizer
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
            
            # Learning rate scheduler
            if self.config['scheduler'] == 'plateau':
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=0.5,
                    patience=self.config['scheduler_patience'],
                    verbose=True
                )
            elif self.config['scheduler'] == 'step':
                self.scheduler = StepLR(
                    self.optimizer,
                    step_size=self.config['step_size'],
                    gamma=0.5
                )
            else:
                self.scheduler = None
            
            # Loss function
            self.criterion = ColorL1Loss()
            
            logger.info(f"Training setup complete. Optimizer: Adam, LR: {self.config['learning_rate']}")
            
        except Exception as e:
            logger.error(f"Error setting up training: {e}")
            raise
    
    def save_checkpoint(self, epoch: int, val_loss: Optional[float] = None, is_best: bool = False):
        """Save model checkpoint."""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'size': self.config['size'],
                'base': self.config['base'],
                'val_loss': val_loss,
                'config': self.config
            }
            
            # Save regular checkpoint
            ckpt_path = os.path.join(self.config['checkpoint_dir'], f'ckpt_epoch_{epoch}.pth')
            torch.save(checkpoint, ckpt_path)
            
            # Save best model
            if is_best:
                best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
                torch.save(checkpoint, best_path)
                logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
            
            logger.info(f"Checkpoint saved: {ckpt_path}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            raise
    
    def save_sample_grid(self, inputs: torch.Tensor, predictions: torch.Tensor, targets: torch.Tensor, epoch: int, prefix: str = 'train'):
        """
        Save sample images for visual inspection.
        """
        try:
            L = inputs[:, :1, ...].detach().cpu().numpy().transpose(0, 2, 3, 1)  # (B,H,W,1)
            pred_ab = predictions.detach().cpu().numpy().transpose(0, 2, 3, 1)
            gt_ab = targets.detach().cpu().numpy().transpose(0, 2, 3, 1)
            samples = []
            num_samples = min(4, L.shape[0])
            for i in range(num_samples):
                rgb_pred = lab_norm_to_rgb_uint8(L[i], pred_ab[i])
                rgb_gt   = lab_norm_to_rgb_uint8(L[i], gt_ab[i])
                comparison = np.concatenate([rgb_pred, rgb_gt], axis=1)
                samples.append(comparison)
            if samples:
                grid = np.concatenate(samples, axis=0)
                output_path = os.path.join(self.config['sample_dir'], f'{prefix}_samples_epoch_{epoch}.png')
                save_rgb(output_path, grid)
                logger.info(f"Sample grid saved: {output_path}")
        except Exception as e:
            logger.error(f"Error saving sample grid: {e}")

    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['epochs']}")
        
        for batch_idx, (inputs, targets, _) in enumerate(pbar):
            try:
                # Move to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(inputs)
                loss = self.criterion(predictions, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config['grad_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                
                self.optimizer.step()
                
                # Update metrics
                batch_loss = loss.item()
                total_loss += batch_loss
                
                # Update progress bar
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                # Save sample grid from last batch
                if batch_idx == num_batches - 1:
                    self.save_sample_grid(inputs, predictions, targets, epoch, 'train')
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        return total_loss / num_batches
    
    def validate_epoch(self, epoch: int) -> float:
        """Validate for one epoch."""
        if self.val_loader is None:
            return float('inf')
        
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, _) in enumerate(self.val_loader):
                try:
                    # Move to device
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Forward pass
                    predictions = self.model(inputs)
                    loss = self.criterion(predictions, targets)
                    
                    total_loss += loss.item()
                    
                    # Save sample grid from first batch
                    if batch_idx == 0:
                        self.save_sample_grid(inputs, predictions, targets, epoch, 'val')
                
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        return total_loss / num_batches
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(1, self.config['epochs'] + 1):
            try:
                # Train epoch
                train_loss = self.train_epoch(epoch)
                
                # Validate epoch
                val_loss = self.validate_epoch(epoch)
                
                # Log metrics
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Save checkpoint
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                self.save_checkpoint(epoch, val_loss, is_best)
                
                # Early stopping
                if (self.config['early_stopping_patience'] > 0 and 
                    self.patience_counter >= self.config['early_stopping_patience']):
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
                
            except Exception as e:
                logger.error(f"Error in epoch {epoch}: {e}")
                continue
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Train image colorization model')
    
    # Data arguments
    parser.add_argument('--data', type=str, default='data/train', help='Training data directory')
    parser.add_argument('--val', type=str, default='data/val', help='Validation data directory')
    
    # Model arguments
    parser.add_argument('--size', type=int, default=256, help='Image size')
    parser.add_argument('--base', type=int, default=64, help='Base number of channels')
    parser.add_argument('--max_hints', type=int, default=20, help='Maximum number of color hints')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    
    # Scheduler arguments
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'step', 'none'])
    parser.add_argument('--scheduler_patience', type=int, default=5, help='Scheduler patience')
    parser.add_argument('--step_size', type=int, default=30, help='Step size for StepLR')
    
    # Early stopping
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Early stopping patience')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory for data loader')
    
    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--sample_dir', type=str, default='samples', help='Sample output directory')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create config
    config = {
        'data_root': args.data,
        'val_root': args.val,
        'size': args.size,
        'base': args.base,
        'max_hints': args.max_hints,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'scheduler': args.scheduler,
        'scheduler_patience': args.scheduler_patience,
        'step_size': args.step_size,
        'early_stopping_patience': args.early_stopping_patience,
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory,
        'checkpoint_dir': args.checkpoint_dir,
        'sample_dir': args.sample_dir,
        'device': device
    }
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()