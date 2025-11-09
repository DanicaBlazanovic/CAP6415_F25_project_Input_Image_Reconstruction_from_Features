"""
Run 1: VGG16 Block1 with Perceptual Loss (MSE + LPIPS)

This script runs the first enhanced experiment to isolate the contribution
of perceptual loss to reconstruction quality.

Comparison:
- Baseline: MSE loss only (14.45 dB PSNR, 0.530 SSIM)
- Run 1: MSE + LPIPS loss (expected: +2-3 dB improvement)

Usage:
    python scripts/run_perceptual_vgg_block1.py

Requirements:
    - Google Colab Pro with A100 GPU (40GB RAM)
    - Estimated time: 200 minutes (3.3 hours)
    - Batch size: 1 (due to 112x112 feature maps)

Cross-platform compatible: Windows, Mac, Linux
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
from models import FeatureExtractor, AttentionDecoder
from dataset import get_dataloaders
from train_perceptual import train_perceptual
import time
import argparse



def parse_args():
    """Parse command line arguments for testing."""
    parser = argparse.ArgumentParser(description='Run VGG16 block1 with perceptual loss')
    parser.add_argument('--limit', type=int, default=None, 
                        help='Limit number of training images (for testing)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (default: 1 for VGG block1)')
    return parser.parse_args()


def main():
    """
    Run VGG16 block1 experiment with perceptual loss.
    
    This experiment uses the same decoder architecture as baseline,
    only changing the loss function to MSE + LPIPS.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Print experiment header
    print("\n" + "="*80)
    print("RUN 1: VGG16 BLOCK1 WITH PERCEPTUAL LOSS")
    print("="*80)
    print("\nObjective: Isolate contribution of perceptual loss (LPIPS)")
    print("Baseline: 14.45 dB PSNR, 0.530 SSIM (MSE loss only)")
    print("Expected: +2-3 dB improvement from perceptual optimization")
    print("\n" + "="*80 + "\n")
    
    # Configuration (use command-line args)
    architecture = 'vgg16'
    layer_name = 'block1'
    batch_size = args.batch_size  # From command line
    epochs = args.epochs  # From command line
    lr = 0.001
    mse_weight = 0.5  # Balance MSE and LPIPS equally
    lpips_weight = 0.5
    
    # Detect device (cross-platform)
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"[INFO] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("[INFO] Using Apple Silicon MPS")
    else:
        device = 'cpu'
        print("[WARNING] Using CPU - training will be very slow!")
    
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Epochs: {epochs}")
    print(f"[INFO] Learning rate: {lr}")
    print(f"[INFO] MSE weight: {mse_weight}")
    print(f"[INFO] LPIPS weight: {lpips_weight}")
    if args.limit:
        print(f"[INFO] TEST MODE: Limited to {args.limit} images")
    print()
    
    # Create data loaders
    print("[INFO] Loading DIV2K dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=2,  # Reduce for stability with batch_size=1
        limit=args.limit  # Use command-line limit (None for full dataset)
    )
    
    print(f"[INFO] Training samples: {len(train_loader.dataset)}")
    print(f"[INFO] Validation samples: {len(val_loader.dataset)}")
    print(f"[INFO] Test samples: {len(test_loader.dataset)}\n")
    
    # Create encoder (frozen VGG16)
    print("[INFO] Creating VGG16 block1 encoder (frozen)...")
    encoder = FeatureExtractor(
        architecture=architecture,
        layer_name=layer_name
    )
    
    # Get feature dimensions for decoder
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        dummy_features = encoder(dummy_input)
        feat_channels, feat_h, feat_w = dummy_features.shape[1:]
    
    print(f"[INFO] Feature shape: {feat_channels} x {feat_h} x {feat_w}")
    
    # Create decoder (trainable, attention-based)
    print("[INFO] Creating attention-based decoder...")
    decoder = AttentionDecoder(
        input_channels=feat_channels,
        input_size=feat_h,
        output_size=224,
        num_blocks=4  # Same as baseline
    )
    
    # Count decoder parameters
    num_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"[INFO] Decoder parameters: {num_params:,}\n")
    
    # Save model architecture info for documentation
    save_path = Path('results') / architecture
    model_info_path = save_path / f'model_info_{architecture}_{layer_name}_perceptual.txt'
    with open(model_info_path, 'w') as f:
        f.write(f"Run 1: Perceptual Loss Experiment\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Architecture: {architecture}\n")
        f.write(f"Layer: {layer_name}\n")
        f.write(f"Feature shape: {feat_channels} x {feat_h} x {feat_w}\n")
        f.write(f"Decoder type: AttentionDecoder\n")
        f.write(f"Decoder parameters: {num_params:,}\n")
        f.write(f"Transformer blocks: 4\n\n")
        f.write(f"Loss Configuration:\n")
        f.write(f"  - MSE weight: {mse_weight}\n")
        f.write(f"  - LPIPS weight: {lpips_weight}\n\n")
        f.write(f"Training Configuration:\n")
        f.write(f"  - Epochs: {epochs}\n")
        f.write(f"  - Learning rate: {lr}\n")
        f.write(f"  - Optimizer: Adam\n")
        f.write(f"  - Scheduler: ReduceLROnPlateau\n")
        f.write(f"  - Device: {device}\n")
    
    print(f"[SAVED] Model info: {model_info_path}\n")
    
    # Record start time
    start_time = time.time()
    
    # Train with perceptual loss
    print("[INFO] Starting training with MSE + LPIPS loss...\n")
    history = train_perceptual(
        encoder=encoder,
        decoder=decoder,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        save_dir='results',
        architecture=architecture,
        layer_name=layer_name,
        mse_weight=mse_weight,
        lpips_weight=lpips_weight
    )
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Print final summary
    print("\n" + "="*80)
    print("RUN 1 COMPLETE: VGG16 BLOCK1 WITH PERCEPTUAL LOSS")
    print("="*80)
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Best validation loss: {history['best_val_loss']:.6f}")
    print(f"Best epoch: {history['best_epoch']}")
    print(f"\nCheckpoints saved in: results/vgg16/checkpoints_perceptual/")
    print("\nNext steps:")
    print("1. Evaluate on test set: python scripts/evaluate_perceptual.py")
    print("2. Generate reconstructions: python scripts/visualize_perceptual.py")
    print("3. Compare with baseline (14.45 dB PSNR)")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()