"""
Adversarial Training for Feature Inversion (Run 2)

Combines MSE loss with adversarial loss from PatchGAN discriminator.
Goal: Produce sharper, more realistic reconstructions than MSE alone.

Training strategy:
1. Generator (decoder) tries to reconstruct images that:
   - Match original pixels (MSE loss)
   - Look realistic enough to fool discriminator (adversarial loss)
2. Discriminator tries to distinguish real images from reconstructions

Loss: Total = MSE + λ_adv × Adversarial
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
    
    This class manages the alternating optimization of:
    - Generator (decoder): creates reconstructions
    - Discriminator: judges if reconstructions look realistic
    
    The competition between these two networks pushes the decoder
    to create sharper, more photo-realistic reconstructions.
    """
    
    def __init__(self, encoder, decoder, discriminator, device, 
                 mse_weight=1.0, adv_weight=0.01, lr=0.001):
        """
        Initialize adversarial trainer.
        
        Sets up two competing networks:
        - Generator (decoder): reconstructs images from features
        - Discriminator: distinguishes real from reconstructed images
        
        Args:
            encoder: Feature extractor (frozen, pre-trained)
            decoder: Decoder network to train (generator)
            discriminator: PatchGAN discriminator
            device: Training device (cuda/mps/cpu)
            mse_weight: Weight for MSE loss (default 1.0)
                       Higher = prioritize pixel accuracy
            adv_weight: Weight for adversarial loss (default 0.01)
                       Higher = prioritize realism over accuracy
            lr: Learning rate for both networks
        """
        # Move models to device and set training modes
        self.encoder = encoder.to(device).eval()  # Frozen, always in eval mode
        self.decoder = decoder.to(device)          # Generator, will be trained
        self.discriminator = discriminator.to(device)  # Will be trained
        self.device = device
        
        # Loss weights control MSE vs adversarial trade-off
        # mse_weight=1.0, adv_weight=0.01 means:
        # - MSE is primary objective (pixel accuracy)
        # - Adversarial is secondary (realism)
        self.mse_weight = mse_weight
        self.adv_weight = adv_weight
        
        # Loss functions
        # MSE: measures pixel-level reconstruction error
        self.mse_loss = nn.MSELoss()
        
        # BCEWithLogitsLoss: binary cross-entropy for real/fake classification
        # Combines sigmoid and BCE for numerical stability
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        
        # Separate optimizers for generator and discriminator
        # Adam with beta1=0.5 is standard for GANs (helps stabilize training)
        self.optimizer_G = optim.Adam(
            decoder.parameters(), 
            lr=lr, 
            betas=(0.5, 0.999)  # Lower beta1 for GAN stability
        )
        self.optimizer_D = optim.Adam(
            discriminator.parameters(), 
            lr=lr, 
            betas=(0.5, 0.999)
        )
        
        # Learning rate schedulers reduce LR when validation loss plateaus
        self.scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_G, 
            mode='min',      # Minimize validation loss
            factor=0.5,      # Reduce LR by half
            patience=5       # Wait 5 epochs before reducing
        )
        self.scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_D, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        # Training history tracks all metrics across epochs
        self.history = {
            'epoch': [],
            'train_loss_G': [],     # Total generator loss
            'train_mse': [],        # MSE component
            'train_adv': [],        # Adversarial component
            'train_loss_D': [],     # Discriminator loss
            'train_D_real': [],     # Discriminator score on real images
            'train_D_fake': [],     # Discriminator score on fake images
            'val_loss_G': [],       # Validation generator loss
            'val_mse': [],          # Validation MSE
            'val_adv': [],          # Validation adversarial
            'learning_rate_G': [],  # Generator LR
            'learning_rate_D': []   # Discriminator LR
        }
        
        # Track best validation loss for checkpoint saving
        self.best_val_loss = float('inf')
    
    def train_discriminator(self, real_images, fake_images):
        """
        Train discriminator to distinguish real from reconstructed images.
        
        Discriminator learns:
        - High scores (~1) for real images
        - Low scores (~0) for reconstructed images
        
        Args:
            real_images: Original images [B, 3, 224, 224]
            fake_images: Reconstructed images [B, 3, 224, 224]
            
        Returns:
            loss_D: Discriminator loss
            D_real: Mean score on real images
            D_fake: Mean score on fake images
        """
        self.optimizer_D.zero_grad()
        
        # Real images: label = 0.9 (label smoothing for stability)
        real_labels = torch.ones(real_images.size(0), 1, 26, 26).to(self.device) * 0.9
        D_real = self.discriminator(real_images)
        loss_real = self.adversarial_loss(D_real, real_labels)
        
        # Fake images: label = 0
        # .detach() prevents gradients flowing to generator
        fake_labels = torch.zeros(fake_images.size(0), 1, 26, 26).to(self.device)
        D_fake = self.discriminator(fake_images.detach())
        loss_fake = self.adversarial_loss(D_fake, fake_labels)
        
        # Average loss from real and fake
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        self.optimizer_D.step()
        
        return loss_D.item(), D_real.mean().item(), D_fake.mean().item()
    
    def train_generator(self, real_images, fake_images):
        """
        Train generator with MSE + adversarial loss.
        
        Generator objectives:
        1. MSE: Match pixels accurately
        2. Adversarial: Fool discriminator
        
        Args:
            real_images: Original images
            fake_images: Reconstructed images
            
        Returns:
            loss_G: Total generator loss
            mse: MSE component
            adv: Adversarial component
        """
        self.optimizer_G.zero_grad()
        
        # MSE loss: pixel accuracy
        mse = self.mse_loss(fake_images, real_images)
        
        # Adversarial loss: fool discriminator
        # Generator wants discriminator to output high scores
        real_labels = torch.ones(fake_images.size(0), 1, 26, 26).to(self.device)
        D_fake = self.discriminator(fake_images)
        adv = self.adversarial_loss(D_fake, real_labels)
        
        # Combined loss with weights
        loss_G = self.mse_weight * mse + self.adv_weight * adv
        loss_G.backward()
        self.optimizer_G.step()
        
        return loss_G.item(), mse.item(), adv.item()
    
    def train_epoch(self, train_loader):
        """
        Train one epoch with alternating updates.
        
        For each batch:
        1. Update discriminator
        2. Update generator
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of average metrics for epoch
        """
        self.decoder.train()
        self.discriminator.train()
        
        # Accumulators for batch metrics
        losses_G, losses_D = [], []
        mse_losses, adv_losses = [], []
        D_real_scores, D_fake_scores = [], []
        
        for images in tqdm(train_loader, desc='Training'):
            images = images.to(self.device)
            
            # Forward pass: encode and decode
            with torch.no_grad():
                features = self.encoder(images)
            
            reconstructed = self.decoder(features)
            
            # Update discriminator (learn to distinguish real from fake)
            loss_D, D_real, D_fake = self.train_discriminator(images, reconstructed)
            
            # Update generator (learn to reconstruct and fool discriminator)
            loss_G, mse, adv = self.train_generator(images, reconstructed)
            
            # Accumulate metrics
            losses_G.append(loss_G)
            losses_D.append(loss_D)
            mse_losses.append(mse)
            adv_losses.append(adv)
            D_real_scores.append(D_real)
            D_fake_scores.append(D_fake)
        
        # Return epoch averages
        return {
            'loss_G': sum(losses_G) / len(losses_G),
            'loss_D': sum(losses_D) / len(losses_D),
            'mse': sum(mse_losses) / len(mse_losses),
            'adv': sum(adv_losses) / len(adv_losses),
            'D_real': sum(D_real_scores) / len(D_real_scores),
            'D_fake': sum(D_fake_scores) / len(D_fake_scores)
        }
    
    @torch.no_grad()
    def validate(self, val_loader):
        """
        Validate generator on validation set.
        
        Only evaluates generator, not discriminator.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.decoder.eval()
        self.discriminator.eval()
        
        losses_G, mse_losses, adv_losses = [], [], []
        
        for images in val_loader:
            images = images.to(self.device)
            
            # Forward pass
            features = self.encoder(images)
            reconstructed = self.decoder(features)
            
            # Calculate losses
            mse = self.mse_loss(reconstructed, images)
            
            real_labels = torch.ones(images.size(0), 1, 26, 26).to(self.device)
            D_fake = self.discriminator(reconstructed)
            adv = self.adversarial_loss(D_fake, real_labels)
            
            loss_G = self.mse_weight * mse + self.adv_weight * adv
            
            losses_G.append(loss_G.item())
            mse_losses.append(mse.item())
            adv_losses.append(adv.item())
        
        return {
            'loss_G': sum(losses_G) / len(losses_G),
            'mse': sum(mse_losses) / len(mse_losses),
            'adv': sum(adv_losses) / len(adv_losses)
        }
    
    def save_checkpoint(self, epoch, val_loss, save_path, is_best=False):
        """
        Save model checkpoint.
        
        Saves both generator and discriminator states.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            save_path: Path to save checkpoint
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'decoder_state_dict': self.decoder.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }
        
        torch.save(checkpoint, save_path)
        
        if is_best:
            print(f"[SAVED] Best model at epoch {epoch} (val_loss: {val_loss:.6f})")
    
    def train(self, train_loader, val_loader, epochs, save_dir):
        """
        Full training loop.
        
        Trains for specified epochs, saves checkpoints, tracks metrics.
        
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
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 70)
            
            # Train one epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate schedulers
            self.scheduler_G.step(val_metrics['loss_G'])
            self.scheduler_D.step(train_metrics['loss_D'])
            
            # Get current learning rates
            lr_G = self.optimizer_G.param_groups[0]['lr']
            lr_D = self.optimizer_D.param_groups[0]['lr']
            
            # Print epoch summary
            print(f"Train - Loss_G: {train_metrics['loss_G']:.6f} "
                  f"(MSE: {train_metrics['mse']:.6f}, Adv: {train_metrics['adv']:.6f})")
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
            self.history['train_loss_D'].append(train_metrics['loss_D'])
            self.history['train_D_real'].append(train_metrics['D_real'])
            self.history['train_D_fake'].append(train_metrics['D_fake'])
            self.history['val_loss_G'].append(val_metrics['loss_G'])
            self.history['val_mse'].append(val_metrics['mse'])
            self.history['val_adv'].append(val_metrics['adv'])
            self.history['learning_rate_G'].append(lr_G)
            self.history['learning_rate_D'].append(lr_D)
            
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
        
        # Save training history to CSV
        metrics_dir = save_dir.parent / 'metrics_adversarial'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.history)
        df.to_csv(metrics_dir / 'training_history.csv', index=False)
        print(f"\n[SAVED] Training history: {metrics_dir / 'training_history.csv'}")
        
        return self.history