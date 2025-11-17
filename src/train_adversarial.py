"""
Adversarial Training for Feature Inversion (Run 2)

Combines MSE loss with adversarial loss from PatchGAN discriminator.
Goal: Produce sharper, more realistic reconstructions than MSE alone.

FIXES APPLIED:
1. Increased MSE weight to 100.0 (was 1.0) - makes pixel accuracy dominant
2. Added warmup period (5 epochs MSE-only before adversarial kicks in)
3. Added gradient clipping for generator stability
4. Reduced discriminator updates (every 2 batches instead of every batch)
5. Better logging to track MSE vs Adversarial contributions

Training strategy:
1. Generator (decoder) tries to reconstruct images that:
   - Match original pixels (MSE loss) - PRIMARY objective
   - Look realistic enough to fool discriminator (adversarial loss) - SECONDARY
2. Discriminator tries to distinguish real images from reconstructions
3. Warmup: First 5 epochs train with MSE only to establish baseline

Loss: Total = 100 × MSE + 0.01 × Adversarial (after warmup)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

from models import FeatureExtractor, AttentionDecoder
from discriminator import PatchGANDiscriminator
from dataset import get_dataloaders

def select_device_safe(architecture, layer_name):
    """
    Select device with memory-aware fallback.
    
    VGG16 block1 (112x112 features) causes MPS OOM on Apple Silicon.
    Auto-fallback to CPU for large feature maps on MPS.
    
    Args:
        architecture: Model architecture
        layer_name: Layer name
        
    Returns:
        device: Safe device for training
    """
    # Large feature maps that cause MPS issues
    large_features = [
        ('vgg16', 'block1'),  # 64x112x112
    ]
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        # Check if this config causes MPS memory issues
        if (architecture, layer_name) in large_features:
            print(f"[WARNING] {architecture} {layer_name} has large features - using CPU instead of MPS")
            return torch.device('cpu')
        return torch.device('mps')
    else:
        return torch.device('cpu')

        
class AdversarialTrainer:
    """
    Train decoder with adversarial loss for realistic reconstructions.
    
    FIXED VERSION with:
    - Higher MSE weight (100.0) for pixel accuracy priority
    - Warmup period (5 epochs MSE-only)
    - Gradient clipping for stability
    - Less frequent discriminator updates
    """
    
    def __init__(self, encoder, decoder, discriminator, device, 
                 mse_weight=100.0, adv_weight=0.01, lr=0.0001, 
                 warmup_epochs=5, grad_clip=1.0, d_update_freq=2):
        """
        Initialize adversarial trainer with improved hyperparameters.
        
        Args:
            encoder: Feature extractor (frozen, pre-trained)
            decoder: Decoder network to train (generator)
            discriminator: PatchGAN discriminator
            device: Training device (cuda/mps/cpu)
            mse_weight: Weight for MSE loss (default 100.0 - INCREASED)
                       Higher = prioritize pixel accuracy
            adv_weight: Weight for adversarial loss (default 0.01)
                       Lower relative contribution due to higher MSE weight
            lr: Learning rate (default 0.0001 - REDUCED for stability)
            warmup_epochs: Epochs to train MSE-only before adding adversarial (default 5)
            grad_clip: Gradient clipping max norm (default 1.0)
            d_update_freq: Update discriminator every N batches (default 2)
        """
        # Move models to device and set training modes
        self.encoder = encoder.to(device).eval()  # Frozen, always in eval mode
        self.decoder = decoder.to(device)          # Generator, will be trained
        self.discriminator = discriminator.to(device)  # Will be trained
        self.device = device
        
        # Loss weights - MSE is now PRIMARY objective
        # With mse_weight=100, adv_weight=0.01:
        # - MSE contributes: 100 × 0.04 = 4.0
        # - Adversarial contributes: 0.01 × 7.0 = 0.07
        # MSE is ~57x more important than adversarial
        self.mse_weight = mse_weight
        self.adv_weight = adv_weight
        self.warmup_epochs = warmup_epochs
        self.grad_clip = grad_clip
        self.d_update_freq = d_update_freq
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        
        # Separate optimizers - LOWER learning rate for stability
        self.optimizer_G = optim.Adam(
            decoder.parameters(), 
            lr=lr,  # 0.0001 instead of 0.001
            betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            discriminator.parameters(), 
            lr=lr,  # Match generator LR
            betas=(0.5, 0.999)
        )
        
        # Learning rate schedulers
        self.scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_G, 
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        self.scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_D, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'epoch': [],
            'train_loss_G': [],
            'train_mse': [],
            'train_adv': [],
            'train_mse_weighted': [],      # NEW: Track weighted contributions
            'train_adv_weighted': [],      # NEW: Track weighted contributions
            'train_loss_D': [],
            'train_D_real': [],
            'train_D_fake': [],
            'val_loss_G': [],
            'val_mse': [],
            'val_adv': [],
            'learning_rate_G': [],
            'learning_rate_D': [],
            'adv_weight_used': []          # NEW: Track actual adversarial weight used
        }
        
        self.best_val_loss = float('inf')
        self.current_epoch = 0
    
    def get_adv_weight(self):
        """
        Get adversarial weight for current epoch (warmup strategy).
        
        Returns 0.0 during warmup, then ramps to full weight.
        
        Returns:
            Current adversarial weight
        """
        if self.current_epoch < self.warmup_epochs:
            return 0.0  # MSE-only during warmup
        else:
            return self.adv_weight  # Full adversarial after warmup
    
    def train_discriminator(self, real_images, fake_images):
        """
        Train discriminator to distinguish real from reconstructed images.
        
        Args:
            real_images: Original images [B, 3, 224, 224]
            fake_images: Reconstructed images [B, 3, 224, 224]
            
        Returns:
            loss_D: Discriminator loss
            D_real_logit: Mean logit on real images
            D_fake_logit: Mean logit on fake images
        """
        self.optimizer_D.zero_grad()
        
        # Real images: label = 0.9 (label smoothing)
        real_labels = torch.ones(real_images.size(0), 1, 26, 26).to(self.device) * 0.9
        D_real = self.discriminator(real_images)
        loss_real = self.adversarial_loss(D_real, real_labels)
        
        # Fake images: label = 0
        fake_labels = torch.zeros(fake_images.size(0), 1, 26, 26).to(self.device)
        D_fake = self.discriminator(fake_images.detach())
        loss_fake = self.adversarial_loss(D_fake, fake_labels)
        
        # Average loss
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        self.optimizer_D.step()
        
        # Return mean logits (before sigmoid)
        return loss_D.item(), D_real.mean().item(), D_fake.mean().item()
    
    def train_generator(self, real_images, fake_images, adv_weight):
        """
        Train generator with MSE + adversarial loss.
        
        Args:
            real_images: Original images
            fake_images: Reconstructed images
            adv_weight: Current adversarial weight (0.0 during warmup)
            
        Returns:
            loss_G: Total generator loss
            mse: MSE component (unweighted)
            adv: Adversarial component (unweighted)
            mse_weighted: MSE contribution to total loss
            adv_weighted: Adversarial contribution to total loss
        """
        self.optimizer_G.zero_grad()
        
        # MSE loss: pixel accuracy
        mse = self.mse_loss(fake_images, real_images)
        
        # Adversarial loss (only if past warmup)
        if adv_weight > 0:
            real_labels = torch.ones(fake_images.size(0), 1, 26, 26).to(self.device)
            D_fake = self.discriminator(fake_images)
            adv = self.adversarial_loss(D_fake, real_labels)
        else:
            adv = torch.tensor(0.0).to(self.device)
        
        # Weighted contributions
        mse_weighted = self.mse_weight * mse
        adv_weighted = adv_weight * adv
        
        # Combined loss
        loss_G = mse_weighted + adv_weighted
        loss_G.backward()
        
        # Gradient clipping for stability
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=self.grad_clip)
        
        self.optimizer_G.step()
        
        return loss_G.item(), mse.item(), adv.item(), mse_weighted.item(), adv_weighted.item()
    
    def train_epoch(self, train_loader):
        """
        Train one epoch with improved strategy.
        
        Updates:
        - Discriminator updated every d_update_freq batches (not every batch)
        - Tracks weighted loss components
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of average metrics for epoch
        """
        self.decoder.train()
        self.discriminator.train()
        
        # Get current adversarial weight (0.0 during warmup)
        adv_weight = self.get_adv_weight()
        
        # Accumulators
        losses_G, losses_D = [], []
        mse_losses, adv_losses = [], []
        mse_weighted_losses, adv_weighted_losses = [], []
        D_real_scores, D_fake_scores = [], []
        
        for batch_idx, images in enumerate(tqdm(train_loader, desc='Training')):
            images = images.to(self.device)
            
            # Forward pass
            with torch.no_grad():
                features = self.encoder(images)
            
            reconstructed = self.decoder(features)
            
            # Update discriminator (every d_update_freq batches)
            if adv_weight > 0 and batch_idx % self.d_update_freq == 0:
                loss_D, D_real, D_fake = self.train_discriminator(images, reconstructed)
                losses_D.append(loss_D)
                D_real_scores.append(D_real)
                D_fake_scores.append(D_fake)
            
            # Update generator (every batch)
            loss_G, mse, adv, mse_w, adv_w = self.train_generator(images, reconstructed, adv_weight)
            
            losses_G.append(loss_G)
            mse_losses.append(mse)
            adv_losses.append(adv)
            mse_weighted_losses.append(mse_w)
            adv_weighted_losses.append(adv_w)
        
        # Return epoch averages
        result = {
            'loss_G': sum(losses_G) / len(losses_G),
            'mse': sum(mse_losses) / len(mse_losses),
            'adv': sum(adv_losses) / len(adv_losses) if adv_losses else 0.0,
            'mse_weighted': sum(mse_weighted_losses) / len(mse_weighted_losses),
            'adv_weighted': sum(adv_weighted_losses) / len(adv_weighted_losses),
        }
        
        # Add discriminator metrics if available
        if losses_D:
            result.update({
                'loss_D': sum(losses_D) / len(losses_D),
                'D_real': sum(D_real_scores) / len(D_real_scores),
                'D_fake': sum(D_fake_scores) / len(D_fake_scores)
            })
        else:
            result.update({'loss_D': 0.0, 'D_real': 0.0, 'D_fake': 0.0})
        
        return result
    
    @torch.no_grad()
    def validate(self, val_loader):
        """
        Validate generator on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.decoder.eval()
        self.discriminator.eval()
        
        adv_weight = self.get_adv_weight()
        
        losses_G, mse_losses, adv_losses = [], [], []
        
        for images in val_loader:
            images = images.to(self.device)
            
            # Forward pass
            features = self.encoder(images)
            reconstructed = self.decoder(features)
            
            # MSE
            mse = self.mse_loss(reconstructed, images)
            
            # Adversarial (if past warmup)
            if adv_weight > 0:
                real_labels = torch.ones(images.size(0), 1, 26, 26).to(self.device)
                D_fake = self.discriminator(reconstructed)
                adv = self.adversarial_loss(D_fake, real_labels)
            else:
                adv = torch.tensor(0.0).to(self.device)
            
            loss_G = self.mse_weight * mse + adv_weight * adv
            
            losses_G.append(loss_G.item())
            mse_losses.append(mse.item())
            adv_losses.append(adv.item())
        
        return {
            'loss_G': sum(losses_G) / len(losses_G),
            'mse': sum(mse_losses) / len(mse_losses),
            'adv': sum(adv_losses) / len(adv_losses)
        }
    
    def save_checkpoint(self, epoch, val_loss, save_path, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'decoder_state_dict': self.decoder.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
            'config': {
                'mse_weight': self.mse_weight,
                'adv_weight': self.adv_weight,
                'warmup_epochs': self.warmup_epochs,
                'grad_clip': self.grad_clip,
                'd_update_freq': self.d_update_freq
            }
        }
        
        torch.save(checkpoint, save_path)
        
        if is_best:
            print(f"[SAVED] Best model at epoch {epoch} (val_loss: {val_loss:.6f})")
    
    def train(self, train_loader, val_loader, epochs, save_dir):
        """
        Full training loop with improved logging.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history dictionary
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            adv_weight = self.get_adv_weight()
            
            print(f"\nEpoch {epoch}/{epochs}")
            if epoch <= self.warmup_epochs:
                print(f"[WARMUP] Training with MSE only (adversarial weight = 0.0)")
            else:
                print(f"[FULL TRAINING] MSE weight = {self.mse_weight}, Adv weight = {adv_weight}")
            print("-" * 70)
            
            # Train one epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate schedulers
            self.scheduler_G.step(val_metrics['loss_G'])
            if adv_weight > 0:
                self.scheduler_D.step(train_metrics['loss_D'])
            
            # Get current learning rates
            lr_G = self.optimizer_G.param_groups[0]['lr']
            lr_D = self.optimizer_D.param_groups[0]['lr']
            
            # Print epoch summary with weighted contributions
            print(f"Train - Loss_G: {train_metrics['loss_G']:.6f}")
            print(f"        MSE: {train_metrics['mse']:.6f} (weighted: {train_metrics['mse_weighted']:.6f})")
            print(f"        Adv: {train_metrics['adv']:.6f} (weighted: {train_metrics['adv_weighted']:.6f})")
            if adv_weight > 0:
                print(f"Train - Loss_D: {train_metrics['loss_D']:.6f} "
                      f"(D_real: {train_metrics['D_real']:.3f}, D_fake: {train_metrics['D_fake']:.3f})")
            print(f"Val   - Loss_G: {val_metrics['loss_G']:.6f} "
                  f"(MSE: {val_metrics['mse']:.6f}, Adv: {val_metrics['adv']:.6f})")
            print(f"LR    - G: {lr_G:.6f}, D: {lr_D:.6f}")
            
            # Update history
            self.history['epoch'].append(epoch)
            self.history['train_loss_G'].append(train_metrics['loss_G'])
            self.history['train_mse'].append(train_metrics['mse'])
            self.history['train_adv'].append(train_metrics['adv'])
            self.history['train_mse_weighted'].append(train_metrics['mse_weighted'])
            self.history['train_adv_weighted'].append(train_metrics['adv_weighted'])
            self.history['train_loss_D'].append(train_metrics['loss_D'])
            self.history['train_D_real'].append(train_metrics['D_real'])
            self.history['train_D_fake'].append(train_metrics['D_fake'])
            self.history['val_loss_G'].append(val_metrics['loss_G'])
            self.history['val_mse'].append(val_metrics['mse'])
            self.history['val_adv'].append(val_metrics['adv'])
            self.history['learning_rate_G'].append(lr_G)
            self.history['learning_rate_D'].append(lr_D)
            self.history['adv_weight_used'].append(adv_weight)
            
            # Save checkpoints
            is_best = val_metrics['loss_G'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss_G']
            
            # Save every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(
                    epoch, val_metrics['loss_G'], 
                    save_dir / f'adversarial_epoch{epoch}.pth'
                )
            
            # Save best model
            if is_best:
                self.save_checkpoint(
                    epoch, val_metrics['loss_G'],
                    save_dir / 'adversarial_best.pth', 
                    is_best=True
                )
        
        # Save final checkpoint
        self.save_checkpoint(
            epochs, val_metrics['loss_G'],
            save_dir / f'adversarial_epoch{epochs}.pth'
        )
        
        # Save training history
        metrics_dir = save_dir.parent / 'metrics_adversarial'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.history)
        df.to_csv(metrics_dir / 'training_history.csv', index=False)
        print(f"\n[SAVED] Training history: {metrics_dir / 'training_history.csv'}")
        
        return self.history