"""
Run 2: VGG16 Block1 with Adversarial Loss (GAN)

Experiment runner for adversarial training with PatchGAN discriminator.
Combines MSE loss with adversarial loss for sharper reconstructions.

Usage:
    python scripts/run_adversarial_vgg_block1.py --epochs 30 --batch-size 1
    python scripts/run_adversarial_vgg_block1.py --limit 10 --epochs 2  # Quick test

Cross-platform compatible: Windows, Mac, Linux
"""

import sys
from pathlib import Path

# Add src directory to Python path for imports
# pathlib ensures cross-platform compatibility (works on Windows/Mac/Linux)
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / 'src'))

import argparse
import time
from datetime import datetime

import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for headless environments (Colab, servers)
import matplotlib.pyplot as plt

from models import FeatureExtractor, AttentionDecoder
from discriminator import PatchGANDiscriminator
from train_adversarial import AdversarialTrainer, select_device_safe
from dataset import get_dataloaders


def save_model_info(config, save_path):
    """
    Save experiment configuration to text file.
    
    Documents all hyperparameters and model architecture details
    for reproducibility and reference.
    
    Args:
        config: Dictionary with experiment configuration
        save_path: Path to save info file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("Run 2: Adversarial Loss Experiment\n")
        f.write("="*50 + "\n")
        f.write(f"Architecture: {config['architecture']}\n")
        f.write(f"Layer: {config['layer']}\n")
        f.write(f"Feature shape: {config['feat_channels']} x {config['feat_h']} x {config['feat_w']}\n")
        f.write(f"Decoder type: AttentionDecoder\n")
        f.write(f"Decoder parameters: {config['decoder_params']:,}\n")
        f.write(f"Discriminator: PatchGAN\n")
        f.write(f"Discriminator parameters: {config['discriminator_params']:,}\n")
        f.write(f"Transformer blocks: {config['num_blocks']}\n")
        f.write(f"Loss Configuration:\n")
        f.write(f"  - MSE weight: {config['mse_weight']}\n")
        f.write(f"  - Adversarial weight: {config['adv_weight']}\n")
        f.write(f"Training Configuration:\n")
        f.write(f"  - Epochs: {config['epochs']}\n")
        f.write(f"  - Learning rate: {config['lr']}\n")
        f.write(f"  - Optimizer: Adam (beta1=0.5, beta2=0.999)\n")
        f.write(f"  - Scheduler: ReduceLROnPlateau\n")
        f.write(f"  - Device: {config['device']}\n")
    
    print(f"[SAVED] Model info: {save_path}")


def plot_training_curves(history, save_path):
    """
    Plot training curves for generator and discriminator losses.
    
    Creates 4 subplots:
    1. Generator total loss (train + val)
    2. MSE component
    3. Adversarial component
    4. Discriminator loss and scores
    
    Args:
        history: Training history dictionary from AdversarialTrainer
        save_path: Path to save figure
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = history['epoch']
    
    # Subplot 1: Generator total loss
    axes[0, 0].plot(epochs, history['train_loss_G'], label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss_G'], label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Total Generator Loss', fontsize=12)
    axes[0, 0].set_title('Generator Loss (MSE + Adversarial)', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Subplot 2: MSE component
    axes[0, 1].plot(epochs, history['train_mse'], label='Train MSE', linewidth=2, color='orange')
    axes[0, 1].plot(epochs, history['val_mse'], label='Val MSE', linewidth=2, color='red')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('MSE Loss', fontsize=12)
    axes[0, 1].set_title('MSE Component (Pixel Accuracy)', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Subplot 3: Adversarial component
    axes[1, 0].plot(epochs, history['train_adv'], label='Train Adv', linewidth=2, color='green')
    axes[1, 0].plot(epochs, history['val_adv'], label='Val Adv', linewidth=2, color='lime')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Adversarial Loss', fontsize=12)
    axes[1, 0].set_title('Adversarial Component (Realism)', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Subplot 4: Discriminator loss and scores
    ax1 = axes[1, 1]
    ax1.plot(epochs, history['train_loss_D'], label='D Loss', linewidth=2, color='purple')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Discriminator Loss', fontsize=12, color='purple')
    ax1.tick_params(axis='y', labelcolor='purple')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add discriminator scores on second y-axis
    ax2 = ax1.twinx()
    ax2.plot(epochs, history['train_D_real'], label='D(real)', linewidth=2, 
             color='blue', linestyle='--')
    ax2.plot(epochs, history['train_D_fake'], label='D(fake)', linewidth=2, 
             color='red', linestyle='--')
    ax2.set_ylabel('Discriminator Scores', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.set_ylim([-1, 1])
    
    axes[1, 1].set_title('Discriminator Training', fontsize=14, fontweight='bold')
    
    # Overall title
    fig.suptitle('Run 2: VGG16 Block1 - Adversarial Loss Training', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] Training curves: {save_path}")


def main():
    """
    Main experiment runner.
    
    Steps:
    1. Parse command-line arguments
    2. Setup device and data loaders
    3. Create models (encoder, decoder, discriminator)
    4. Train with adversarial loss
    5. Save checkpoints and training curves
    """
    # Parse command-line arguments
    # This allows running with different configurations without editing code
    parser = argparse.ArgumentParser(
        description='Run 2: Train VGG16 Block1 decoder with adversarial loss',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size (use 1 for large feature maps)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate for both generator and discriminator')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of data loading workers')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit training images (for quick testing)')
    
    # Loss weights
    parser.add_argument('--mse-weight', type=float, default=1.0,
                       help='Weight for MSE loss')
    parser.add_argument('--adv-weight', type=float, default=0.01,
                       help='Weight for adversarial loss')
    
    args = parser.parse_args()
    
    # Print experiment header
    print("\n" + "="*70)
    print("RUN 2: ADVERSARIAL LOSS TRAINING - VGG16 BLOCK1")
    print("="*70)
    
    # Configuration
    # These settings define the experiment and must match between training/evaluation
    architecture = 'vgg16'
    layer_name = 'block1'
    num_blocks = 4  # Number of transformer blocks in decoder
    
    # Auto-detect safe device (handles MPS memory issues)
    # This function returns CPU on Mac M1 for VGG16 block1 to avoid OOM
    device = select_device_safe(architecture, layer_name)
    print(f"\nDevice: {device}")
    print(f"System: {sys.platform}")
    
    # Print configuration
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION - Run 2: Adversarial Loss")
    print("="*70)
    print(f"Architecture: {architecture}")
    print(f"Layer: {layer_name}")
    print(f"Device: {device}")
    print(f"System: {sys.platform}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"MSE Weight: {args.mse_weight}")
    print(f"Adversarial Weight: {args.adv_weight}")
    print(f"Optimizer: Adam (beta1=0.5, beta2=0.999)")
    print(f"Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
    
    # Create results directories
    # Using pathlib ensures cross-platform path handling
    checkpoint_dir = Path('results') / architecture / 'checkpoints_adversarial'
    metrics_dir = Path('results') / architecture / 'metrics_adversarial'
    figures_dir = Path('results') / architecture / 'figures_adversarial'
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Checkpoint Dir: {checkpoint_dir}")
    print("="*70 + "\n")
    
    # Load data
    # get_dataloaders handles dataset creation and train/val/test splits
    # limit parameter allows quick testing with subset of data
    print("[INFO] Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}\n")
    
    # Create encoder (frozen feature extractor)
    # This extracts features from pre-trained VGG16 block1
    # Encoder weights are frozen (not trained)
    print("[INFO] Creating encoder...")
    encoder = FeatureExtractor(architecture=architecture, layer_name=layer_name)
    encoder.to(device)
    encoder.eval()  # Always in eval mode (frozen)
    
    # Get feature dimensions by passing dummy input
    # This is more robust than hardcoding dimensions
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        dummy_features = encoder(dummy_input)
        feat_channels, feat_h, feat_w = dummy_features.shape[1:]
    
    print(f"Feature shape: {feat_channels} x {feat_h} x {feat_w}")
    
    # Create decoder (generator)
    # AttentionDecoder uses transformer blocks for feature-to-image reconstruction
    print("[INFO] Creating decoder (generator)...")
    decoder = AttentionDecoder(
        input_channels=feat_channels,  # 64 for VGG16 block1
        input_size=feat_h,             # 112 for VGG16 block1
        output_size=224,               # Reconstruct to 224x224 RGB
        num_blocks=num_blocks          # 4 transformer blocks
    )
    
    decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Decoder parameters: {decoder_params:,}")
    
    # Create discriminator (PatchGAN)
    # Discriminator judges whether 70x70 patches look real or reconstructed
    print("[INFO] Creating discriminator (PatchGAN)...")
    discriminator = PatchGANDiscriminator(in_channels=3)
    
    discriminator_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Discriminator parameters: {discriminator_params:,}\n")
    
    # Save model configuration
    # Documents all hyperparameters for reproducibility
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
        'device': str(device)
    }
    
    model_info_path = Path('results') / architecture / f'model_info_{architecture}_{layer_name}_adversarial.txt'
    save_model_info(config, model_info_path)
    
    # Create trainer
    # AdversarialTrainer manages alternating updates of generator and discriminator
    print("[INFO] Creating adversarial trainer...")
    trainer = AdversarialTrainer(
        encoder=encoder,
        decoder=decoder,
        discriminator=discriminator,
        device=device,
        mse_weight=args.mse_weight,
        adv_weight=args.adv_weight,
        lr=args.lr
    )
    
    # Start training
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    start_time = time.time()
    
    # Train model
    # This runs the full training loop with alternating G/D updates
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=checkpoint_dir
    )
    
    # Calculate training time
    total_time = (time.time() - start_time) / 60  # Convert to minutes
    
    # Plot training curves
    # Visualizes loss progression and discriminator behavior
    curves_path = figures_dir / f'{architecture}_{layer_name}_adversarial_training_curves.png'
    plot_training_curves(history, curves_path)
    
    # Print final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE - Run 2: Adversarial Loss")
    print("="*70)
    print(f"Total Time: {total_time:.1f} minutes")
    print(f"Best Val Loss: {trainer.best_val_loss:.6f} at epoch {history['val_loss_G'].index(min(history['val_loss_G'])) + 1}")
    print(f"Final LR (G): {history['learning_rate_G'][-1]:.6f}")
    print(f"Final LR (D): {history['learning_rate_D'][-1]:.6f}")
    print(f"Checkpoints saved in: {checkpoint_dir}/")
    print("="*70 + "\n")
    
    # Print next steps
    print("="*70)
    print(f"RUN 2 COMPLETE: VGG16 BLOCK1 WITH ADVERSARIAL LOSS")
    print("="*70)
    print(f"Total training time: {total_time:.1f} minutes")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"Best epoch: {history['val_loss_G'].index(min(history['val_loss_G'])) + 1}")
    print(f"\nCheckpoints saved in: {checkpoint_dir}/")
    print(f"\nNext steps:")
    print(f"1. Evaluate on test set: python scripts/evaluate_adversarial.py")
    print(f"2. Generate reconstructions: python scripts/visualize_adversarial.py")
    print(f"3. Compare with baseline and Run 1 (perceptual loss)")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()