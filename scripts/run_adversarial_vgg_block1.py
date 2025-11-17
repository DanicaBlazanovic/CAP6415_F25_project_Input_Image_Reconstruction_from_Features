"""
Run 2: VGG16 Block1 with Adversarial Loss (GAN-based Reconstruction)

This script implements adversarial training for image reconstruction from VGG16 
block1 features, combining MSE loss (pixel accuracy) with adversarial loss 
(perceptual realism) to achieve sharper, more photorealistic reconstructions.

ADVERSARIAL TRAINING APPROACH:
==============================

The training uses a GAN framework with two competing networks:
- Generator (Decoder): Reconstructs images from features
- Discriminator (PatchGAN): Judges whether images are real or reconstructed

This competition encourages the decoder to create reconstructions that are both
pixel-accurate (MSE) and perceptually realistic (adversarial).

LOSS FUNCTION:
=============

Total Loss = λ_MSE × MSE + λ_adv × Adversarial

Where:
- λ_MSE = 100.0 (makes pixel accuracy the PRIMARY objective)
- λ_adv = 0.01 (adds subtle perceptual enhancement)

With typical values:
- MSE ~0.04 → weighted contribution = 100 × 0.04 = 4.0
- Adversarial ~7.0 → weighted contribution = 0.01 × 7.0 = 0.07
- MSE dominates at 57:1 ratio, ensuring pixel fidelity

TRAINING STRATEGY:
==================

1. Warmup Period (Epochs 1-5):
   - Train with MSE only (adversarial weight = 0.0)
   - Establishes solid baseline reconstruction
   - Prevents early training instability

2. Full Training (Epochs 6-30):
   - Combine MSE + adversarial loss
   - Alternating updates: discriminator every 2 batches, generator every batch
   - Gradient clipping (max_norm=1.0) prevents exploding gradients

3. Optimization:
   - Learning rate: 0.0001 (stable for GAN training)
   - Optimizer: Adam with beta1=0.5, beta2=0.999
   - Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)

EXPECTED RESULTS:
=================

Based on GAN reconstruction literature and our experimental setup:
- MSE: ~0.03-0.04 (competitive with baseline MSE ~0.04)
- PSNR: ~14-15 dB (competitive with baseline 14.45 dB)
- LPIPS: <0.4 (improved from baseline 0.398 via adversarial sharpening)
- Visual quality: Sharper textures, enhanced high-frequency details

USAGE:
======

    # Full training (30 epochs, recommended)
    python scripts/run_adversarial_vgg_block1.py --epochs 30 --batch-size 1
    
    # Quick test (10 images, 2 epochs)
    python scripts/run_adversarial_vgg_block1.py --limit 10 --epochs 2
    
    # Custom hyperparameters
    python scripts/run_adversarial_vgg_block1.py --mse-weight 100.0 --adv-weight 0.01

REQUIREMENTS:
=============

- Hardware: Google Colab Pro A100 (40GB) for VGG16 block1 (112×112 features)
- Platform: Cross-platform compatible (Windows, Mac, Linux)
- Python: 3.10+ with PyTorch 2.0+
- Time: ~2-3 hours for 30 epochs on A100

REFERENCES:
===========

This implementation is based on:
- SRGAN (Ledig et al., 2017): Adversarial loss for super-resolution
- Pix2Pix (Isola et al., 2017): PatchGAN discriminator architecture
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / 'src'))

import argparse
import time
from datetime import datetime

import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import FeatureExtractor, AttentionDecoder
from discriminator import PatchGANDiscriminator
from train_adversarial import AdversarialTrainer, select_device_safe
from dataset import get_dataloaders


def save_model_info(config, save_path):
    """
    Save comprehensive experiment configuration to text file.
    
    This function documents all hyperparameters, model architectures, and
    training settings for reproducibility. The saved file serves as a 
    permanent record of experiment configuration.
    
    WHAT THIS SAVES:
    ================
    - Model architecture details (VGG16 block1, feature dimensions)
    - Decoder and discriminator parameter counts
    - Loss configuration (MSE weight, adversarial weight, warmup strategy)
    - Training hyperparameters (epochs, learning rate, optimizers, schedulers)
    - Hardware used (device: CUDA/MPS/CPU)
    
    WHY THIS MATTERS:
    =================
    - Enables exact reproduction of experiments months/years later
    - Documents what hyperparameters were actually used (not just defaults)
    - Helps compare different experimental runs
    - Critical for academic papers and technical reports
    
    Args:
        config (dict): Dictionary containing all experiment parameters
            Required keys:
            - architecture: Model name (e.g., 'vgg16')
            - layer: Layer name (e.g., 'block1')
            - feat_channels, feat_h, feat_w: Feature tensor dimensions
            - decoder_params, discriminator_params: Model sizes
            - num_blocks: Transformer blocks in decoder
            - mse_weight, adv_weight: Loss weights
            - warmup_epochs: MSE-only warmup duration
            - grad_clip: Gradient clipping threshold
            - d_update_freq: Discriminator update frequency
            - epochs, lr: Training hyperparameters
            - device: Hardware used
            
        save_path (str or Path): File path to save configuration
            Example: 'results/vgg16/model_info_vgg16_block1_adversarial_fixed.txt'
    
    Returns:
        None (saves file to disk)
    
    Example output file:
        Run 2: Adversarial Loss Experiment
        ==================================================
        Architecture: vgg16
        Layer: block1
        Feature shape: 64 x 112 x 112
        Decoder parameters: 233,667
        
    """
    
    # Convert to Path object for cross-platform compatibility
    # pathlib handles Windows backslashes and Unix forward slashes automatically
    save_path = Path(save_path)
    
    # Create parent directories if they don't exist
    # parents=True: create all intermediate directories
    # exist_ok=True: don't raise error if directory already exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write configuration to text file
    # Using context manager ('with') ensures file is properly closed even if error occurs
    with open(save_path, 'w') as f:
        # Header section
        f.write("Run 2: Adversarial Loss Experiment (FIXED VERSION)\n")
        f.write("="*50 + "\n")
        
        # Model architecture details
        # These define what features we're inverting
        f.write(f"Architecture: {config['architecture']}\n")
        f.write(f"Layer: {config['layer']}\n")
        f.write(f"Feature shape: {config['feat_channels']} x {config['feat_h']} x {config['feat_w']}\n")
        
        # Decoder (generator) architecture
        f.write(f"Decoder type: AttentionDecoder\n")
        f.write(f"Decoder parameters: {config['decoder_params']:,}\n")  # :, adds thousand separators
        
        # Discriminator architecture
        f.write(f"Discriminator: PatchGAN\n")
        f.write(f"Discriminator parameters: {config['discriminator_params']:,}\n")
        
        # Decoder internal structure
        f.write(f"Transformer blocks: {config['num_blocks']}\n")
        
        # Loss configuration - CRITICAL for understanding training behavior
        f.write(f"\nLoss Configuration (FIXED):\n")
        f.write(f"  - MSE weight: {config['mse_weight']} (INCREASED for pixel accuracy)\n")
        f.write(f"  - Adversarial weight: {config['adv_weight']}\n")
        f.write(f"  - Warmup epochs: {config['warmup_epochs']} (MSE-only)\n")
        f.write(f"  - Gradient clipping: {config['grad_clip']}\n")
        f.write(f"  - Discriminator update freq: every {config['d_update_freq']} batches\n")
        
        # Training hyperparameters
        f.write(f"\nTraining Configuration:\n")
        f.write(f"  - Epochs: {config['epochs']}\n")
        f.write(f"  - Learning rate: {config['lr']} (REDUCED for stability)\n")
        f.write(f"  - Optimizer: Adam (beta1=0.5, beta2=0.999)\n")
        f.write(f"  - Scheduler: ReduceLROnPlateau\n")
        f.write(f"  - Device: {config['device']}\n")
    
    # Print confirmation message
    print(f"[SAVED] Model info: {save_path}")


def plot_training_curves(history, save_path):
    """
    Plot comprehensive training curves showing all loss components and training dynamics.
    
    This function creates a 2x3 grid of subplots visualizing the complete
    training process for adversarial training. It helps diagnose training
    issues and understand the balance between different objectives.
    
    WHAT THIS PLOTS:
    ================
    
    Subplot 1 (top-left): Generator Total Loss
    - Train vs validation generator loss over epochs
    - Shows overall training progress
    - Vertical line marks end of warmup period
    
    Subplot 2 (top-center): MSE Component (Unweighted)
    - Raw MSE values before applying weight
    - Should decrease and stabilize around 0.03-0.04
    - Primary metric for pixel accuracy
    
    Subplot 3 (top-right): Weighted Loss Contributions
    - Shows actual contribution of MSE vs adversarial to total loss
    - MSE (blue) should dominate (4.0 vs 0.07 = 57:1 ratio)
    - Helps verify loss balancing is correct
    
    Subplot 4 (bottom-left): Adversarial Component (Unweighted)
    - Raw adversarial loss before applying weight
    - Can be high (~3-9) but weighted contribution is small
    - Shows if generator is learning to fool discriminator
    
    Subplot 5 (bottom-center): Discriminator Loss
    - Discriminator's classification loss
    - Should be moderate (~0.3-0.7) - not too low or high
    - Too low = discriminator too strong, too high = too weak
    
    Subplot 6 (bottom-right): Discriminator Scores (Logits)
    - D_real (blue): discriminator score on real images (should be positive)
    - D_fake (red): discriminator score on reconstructed images (should be negative)
    - Gap between them shows discriminator strength
    - Horizontal line at y=0 shows decision boundary
    
    WHY THIS MATTERS:
    =================
    - Diagnose training failures (MSE not decreasing, discriminator too strong)
    - Verify fixes are working (warmup period, loss balancing)
    - Track GAN training dynamics (generator vs discriminator balance)
    - Document experiment for papers/reports
    
    Args:
        history (dict): Training history from AdversarialTrainer.train()
            Required keys:
            - epoch: List of epoch numbers [1, 2, 3, ...]
            - train_loss_G, val_loss_G: Generator losses
            - train_mse, val_mse: MSE components
            - train_mse_weighted, train_adv_weighted: Weighted contributions
            - train_adv, val_adv: Adversarial components
            - train_loss_D: Discriminator loss
            - train_D_real, train_D_fake: Discriminator scores
            - adv_weight_used: Adversarial weight per epoch (0 during warmup)
            
        save_path (str or Path): File path to save figure
            Example: 'results/vgg16/figures_adversarial_fixed/training_curves.png'
    
    Returns:
        None (saves figure to disk)
    
    Example:
        >>> history = trainer.train(...)
        >>> plot_training_curves(history, 'training_curves.png')
        [SAVED] Training curves: training_curves.png
    """
    # Convert to Path for cross-platform compatibility
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure with 2 rows, 3 columns of subplots
    # figsize=(18, 10): wide enough to show details clearly
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Extract epoch numbers for x-axis
    epochs = history['epoch']
    
    # Find warmup end epoch (when adversarial weight becomes non-zero)
    # This marks the transition from MSE-only to MSE+adversarial training
    # adv_weight_used is 0.0 during warmup, then 0.01 after
    warmup = history['adv_weight_used'].index(next(x for x in history['adv_weight_used'] if x > 0)) if any(x > 0 for x in history['adv_weight_used']) else len(epochs)
    
    # ==================================================================
    # SUBPLOT 1: Generator Total Loss (Train vs Validation)
    # ==================================================================
    # Shows combined MSE + adversarial loss over training
    # Validation loss helps detect overfitting
    axes[0, 0].plot(epochs, history['train_loss_G'], label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss_G'], label='Val', linewidth=2)
    
    # Mark warmup period end with vertical red dashed line
    if warmup > 0:
        axes[0, 0].axvline(x=warmup, color='red', linestyle='--', label='Warmup End')
    
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Total Generator Loss', fontsize=12)
    axes[0, 0].set_title('Generator Loss (MSE + Adversarial)', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)  # Light grid for readability
    
    # ==================================================================
    # SUBPLOT 2: MSE Component (Unweighted)
    # ==================================================================
    # Raw MSE before multiplying by weight (100.0)
    # This is the actual pixel-level reconstruction error
    # Should decrease from ~0.04 to ~0.03 if training is working
    axes[0, 1].plot(epochs, history['train_mse'], label='Train MSE', linewidth=2, color='orange')
    axes[0, 1].plot(epochs, history['val_mse'], label='Val MSE', linewidth=2, color='red')
    
    if warmup > 0:
        axes[0, 1].axvline(x=warmup, color='red', linestyle='--', label='Warmup End')
    
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('MSE Loss (unweighted)', fontsize=12)
    axes[0, 1].set_title('MSE Component (Pixel Accuracy)', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # ==================================================================
    # SUBPLOT 3: Weighted Loss Contributions
    # ==================================================================
    # Shows actual contribution to total loss after applying weights
    # MSE weighted = 100 × MSE (should be ~4.0 if MSE ~0.04)
    # Adv weighted = 0.01 × Adv (should be ~0.07 if Adv ~7.0)
    # MSE should dominate (blue line >> green line)
    axes[0, 2].plot(epochs, history['train_mse_weighted'], label='MSE (weighted)', linewidth=2, color='blue')
    axes[0, 2].plot(epochs, history['train_adv_weighted'], label='Adv (weighted)', linewidth=2, color='green')
    
    if warmup > 0:
        axes[0, 2].axvline(x=warmup, color='red', linestyle='--', label='Warmup End')
    
    axes[0, 2].set_xlabel('Epoch', fontsize=12)
    axes[0, 2].set_ylabel('Weighted Loss Contribution', fontsize=12)
    axes[0, 2].set_title('Loss Components (Weighted)', fontsize=14, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # ==================================================================
    # SUBPLOT 4: Adversarial Component (Unweighted)
    # ==================================================================
    # Raw adversarial loss before multiplying by weight (0.01)
    # Can be large (~3-9) - this is normal for GANs
    # What matters is the weighted contribution (subplot 3)
    # Should be zero during warmup, then jump up after epoch 5
    axes[1, 0].plot(epochs, history['train_adv'], label='Train Adv', linewidth=2, color='green')
    axes[1, 0].plot(epochs, history['val_adv'], label='Val Adv', linewidth=2, color='lime')
    
    if warmup > 0:
        axes[1, 0].axvline(x=warmup, color='red', linestyle='--', label='Warmup End')
    
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Adversarial Loss (unweighted)', fontsize=12)
    axes[1, 0].set_title('Adversarial Component (Realism)', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # ==================================================================
    # SUBPLOT 5: Discriminator Loss
    # ==================================================================
    # How well discriminator is learning to classify real vs fake
    # Should be moderate (~0.3-0.7):
    # - Too low (<0.2) = discriminator too strong, generator can't fool it
    # - Too high (>1.0) = discriminator too weak, not providing useful gradient
    # Should be zero during warmup (no discriminator training)
    axes[1, 1].plot(epochs, history['train_loss_D'], label='D Loss', linewidth=2, color='purple')
    
    if warmup > 0:
        axes[1, 1].axvline(x=warmup, color='red', linestyle='--', label='Warmup End')
    
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Discriminator Loss', fontsize=12)
    axes[1, 1].set_title('Discriminator Training', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # ==================================================================
    # SUBPLOT 6: Discriminator Scores (Raw Logits)
    # ==================================================================
    # D_real: discriminator output on real images (should be positive ~+1 to +3)
    # D_fake: discriminator output on reconstructed images (should be negative ~-3 to -7)
    # Gap between them shows how well discriminator distinguishes real from fake
    # Horizontal line at y=0 is decision boundary:
    #   - Above 0: discriminator thinks image is real
    #   - Below 0: discriminator thinks image is fake
    axes[1, 2].plot(epochs, history['train_D_real'], label='D(real) logit', linewidth=2, color='blue')
    axes[1, 2].plot(epochs, history['train_D_fake'], label='D(fake) logit', linewidth=2, color='red')
    axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.3)  # Decision boundary
    
    if warmup > 0:
        axes[1, 2].axvline(x=warmup, color='red', linestyle='--', label='Warmup End')
    
    axes[1, 2].set_xlabel('Epoch', fontsize=12)
    axes[1, 2].set_ylabel('Discriminator Logits', fontsize=12)
    axes[1, 2].set_title('Discriminator Scores (Raw Logits)', fontsize=14, fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Overall figure title
    fig.suptitle('Run 2: VGG16 Block1 - Adversarial Training', 
                 fontsize=16, fontweight='bold')
    
    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Save figure as high-resolution PNG
    # dpi=200: high quality for papers/presentations
    # bbox_inches='tight': remove extra whitespace
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    # Close figure to free memory
    plt.close()
    
    print(f"[SAVED] Training curves: {save_path}")


def main():
    """Main experiment runner with fixed hyperparameters."""
    parser = argparse.ArgumentParser(
        description='Run 2: Train VGG16 Block1 decoder with adversarial loss (FIXED)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size (use 1 for large feature maps)')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate (REDUCED for stability)')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of data loading workers')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit training images (for quick testing)')
    
    # Loss weights (FIXED)
    parser.add_argument('--mse-weight', type=float, default=100.0,
                       help='Weight for MSE loss (INCREASED)')
    parser.add_argument('--adv-weight', type=float, default=0.01,
                       help='Weight for adversarial loss')
    
    # New parameters
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Epochs for MSE-only warmup')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                       help='Gradient clipping max norm')
    parser.add_argument('--d-update-freq', type=int, default=2,
                       help='Update discriminator every N batches')
    
    args = parser.parse_args()
    
    # Print experiment header
    print("\n" + "="*70)
    print("RUN 2: ADVERSARIAL LOSS TRAINING - VGG16 BLOCK1 (FIXED VERSION)")
    print("="*70)
    
    # Configuration
    architecture = 'vgg16'
    layer_name = 'block1'
    num_blocks = 4
    
    device = select_device_safe(architecture, layer_name)
    print(f"\nDevice: {device}")
    print(f"System: {sys.platform}")
    
    # Print configuration
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION - Run 2: Adversarial Loss (FIXED)")
    print("="*70)
    print(f"Architecture: {architecture}")
    print(f"Layer: {layer_name}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr} (REDUCED for stability)")
    print(f"MSE Weight: {args.mse_weight} (INCREASED - pixel accuracy priority)")
    print(f"Adversarial Weight: {args.adv_weight}")
    print(f"Warmup Epochs: {args.warmup_epochs} (MSE-only)")
    print(f"Gradient Clipping: {args.grad_clip}")
    print(f"Discriminator Update Freq: every {args.d_update_freq} batches")
    print(f"Optimizer: Adam (beta1=0.5, beta2=0.999)")
    print(f"Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
    
    # Create results directories
    checkpoint_dir = Path('results') / architecture / 'checkpoints_adversarial_fixed'
    metrics_dir = Path('results') / architecture / 'metrics_adversarial_fixed'
    figures_dir = Path('results') / architecture / 'figures_adversarial_fixed'
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Checkpoint Dir: {checkpoint_dir}")
    print("="*70 + "\n")
    
    # Load data
    print("[INFO] Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}\n")
    
    # Create encoder
    print("[INFO] Creating encoder...")
    encoder = FeatureExtractor(architecture=architecture, layer_name=layer_name)
    encoder.to(device)
    encoder.eval()
    
    # Get feature dimensions
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        dummy_features = encoder(dummy_input)
        feat_channels, feat_h, feat_w = dummy_features.shape[1:]
    
    print(f"Feature shape: {feat_channels} x {feat_h} x {feat_w}")
    
    # Create decoder
    print("[INFO] Creating decoder (generator)...")
    decoder = AttentionDecoder(
        input_channels=feat_channels,
        input_size=feat_h,
        output_size=224,
        num_blocks=num_blocks
    )
    
    decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Decoder parameters: {decoder_params:,}")
    
    # Create discriminator
    print("[INFO] Creating discriminator (PatchGAN)...")
    discriminator = PatchGANDiscriminator(in_channels=3)
    
    discriminator_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Discriminator parameters: {discriminator_params:,}\n")
    
    # Save model configuration
    config = {
        'architecture': architecture,
        'layer': layer_name,
        'feat_channels': feat_channels,
        'feat_h': feat_h,
        'feat_w': feat_w,
        'num_blocks': num_blocks,
        'decoder_params': decoder_params,
        'discriminator_params': discriminator_params,
        'epochs': args.epochs,
        'lr': args.lr,
        'mse_weight': args.mse_weight,
        'adv_weight': args.adv_weight,
        'warmup_epochs': args.warmup_epochs,
        'grad_clip': args.grad_clip,
        'd_update_freq': args.d_update_freq,
        'device': str(device)
    }
    
    model_info_path = Path('results') / architecture / f'model_info_{architecture}_{layer_name}_adversarial_fixed.txt'
    save_model_info(config, model_info_path)
    
    # Create trainer with fixed hyperparameters
    print("[INFO] Creating adversarial trainer (FIXED VERSION)...")
    trainer = AdversarialTrainer(
        encoder=encoder,
        decoder=decoder,
        discriminator=discriminator,
        device=device,
        mse_weight=args.mse_weight,
        adv_weight=args.adv_weight,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        grad_clip=args.grad_clip,
        d_update_freq=args.d_update_freq
    )
    
    # Start training
    print("\n" + "="*70)
    print("STARTING TRAINING (FIXED VERSION)")
    print("="*70)
    
    start_time = time.time()
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=checkpoint_dir
    )
    
    # Calculate training time
    total_time = (time.time() - start_time) / 60
    
    # Plot training curves
    curves_path = figures_dir / f'{architecture}_{layer_name}_adversarial_fixed_training_curves.png'
    plot_training_curves(history, curves_path)
    
    # Print final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE - Run 2: Adversarial Loss (FIXED)")
    print("="*70)
    print(f"Total Time: {total_time:.1f} minutes")
    print(f"Best Val Loss: {trainer.best_val_loss:.6f}")
    print(f"Final MSE (train): {history['train_mse'][-1]:.6f}")
    print(f"Final MSE (val): {history['val_mse'][-1]:.6f}")
    print(f"Checkpoints saved in: {checkpoint_dir}/")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()