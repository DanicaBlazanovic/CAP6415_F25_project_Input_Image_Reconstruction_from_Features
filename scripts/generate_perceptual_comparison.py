"""
Generate side-by-side comparison: Baseline (MSE) vs Run 1 (Perceptual Loss)

Shows visual quality differences between MSE-only and MSE+LPIPS training.

Usage:
    python scripts/generate_perceptual_comparison.py

Cross-platform compatible: Windows, Mac, Linux
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
import matplotlib.pyplot as plt
import numpy as np
from models import FeatureExtractor, AttentionDecoder
from dataset import get_dataloaders, denormalize


def load_model(architecture, layer_name, checkpoint_path, device):
    """
    Load trained encoder-decoder model from checkpoint.
    
    Args:
        architecture: Architecture name ('vgg16')
        layer_name: Layer name ('block1')
        checkpoint_path: Path to .pth checkpoint file
        device: Device to load on ('cuda', 'mps', 'cpu')
        
    Returns:
        encoder: Loaded encoder (frozen)
        decoder: Loaded decoder (trained weights)
    """
    # Create encoder (frozen feature extractor)
    encoder = FeatureExtractor(architecture=architecture, layer_name=layer_name)
    encoder.to(device)
    encoder.eval()
    
    # Get feature dimensions for decoder
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(device)
        feat = encoder(dummy)
        feat_channels, feat_h, feat_w = feat.shape[1:]
    
    # Create decoder with same architecture as training
    decoder = AttentionDecoder(
        input_channels=feat_channels,
        input_size=feat_h,
        output_size=224,
        num_blocks=4
    )
    decoder.to(device)
    
    # Load trained weights from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.eval()
    
    return encoder, decoder


def generate_comparison_grid(num_samples = 8):
    """
    Generate side-by-side comparison of Baseline vs Run 1 reconstructions.
    
    Creates a grid showing:
    - Row 1: Original images
    - Row 2: Baseline reconstructions (MSE only)
    - Row 3: Run 1 reconstructions (MSE + LPIPS)
    
    Args:
        num_samples: Number of test images to show (default 8)
    """
    print("\n" + "="*80)
    print("GENERATING BASELINE VS RUN 1 COMPARISON")
    print("="*80)
    
    # Configuration
    architecture = 'vgg16'
    layer_name = 'block1'
    
    # Paths to checkpoints
    baseline_checkpoint = 'results/vgg16/checkpoints/vgg16_block1_attention_best.pth'
    perceptual_checkpoint = 'results/vgg16/checkpoints_perceptual/vgg16_block1_perceptual_best.pth'
    
    # Check if checkpoints exist
    if not Path(baseline_checkpoint).exists():
        print(f"ERROR: Baseline checkpoint not found: {baseline_checkpoint}")
        return
    
    if not Path(perceptual_checkpoint).exists():
        print(f"ERROR: Perceptual checkpoint not found: {perceptual_checkpoint}")
        return
    
    # Detect device (force CPU on Mac due to MPS memory issues)
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"[INFO] Using CUDA GPU")
    else:
        device = 'cpu'
        print(f"[INFO] Using CPU (VGG16 block1 large features)")
    
    # Load test data
    print("[INFO] Loading test images...")
    _, _, test_loader = get_dataloaders(
        batch_size=num_samples,
        num_workers=0,  # Avoid multiprocessing issues
        limit=None
    )
    
    # Get first batch of test images
    test_images = next(iter(test_loader)).to(device)
    actual_samples = min(num_samples, test_images.size(0))
    test_images = test_images[:actual_samples]
    
    print(f"[INFO] Using {actual_samples} test images")
    
    # Load baseline model (MSE only)
    print("\n[INFO] Loading baseline model (MSE loss)...")
    baseline_encoder, baseline_decoder = load_model(
        architecture, layer_name, baseline_checkpoint, device
    )
    
    # Load perceptual model (MSE + LPIPS)
    print("[INFO] Loading Run 1 model (MSE + LPIPS loss)...")
    perceptual_encoder, perceptual_decoder = load_model(
        architecture, layer_name, perceptual_checkpoint, device
    )
    
    # Generate reconstructions
    print("\n[INFO] Generating reconstructions...")
    with torch.no_grad():
        # Baseline reconstructions
        baseline_features = baseline_encoder(test_images)
        baseline_recon = baseline_decoder(baseline_features)
        
        # Perceptual reconstructions
        perceptual_features = perceptual_encoder(test_images)
        perceptual_recon = perceptual_decoder(perceptual_features)
    
    # Denormalize for visualization
    # Images are normalized with ImageNet stats, need to reverse
    test_images_denorm = denormalize(test_images).cpu().numpy()
    baseline_recon_denorm = denormalize(baseline_recon).cpu().numpy()
    perceptual_recon_denorm = denormalize(perceptual_recon).cpu().numpy()
    
    # Create comparison figure
    # Layout: 3 rows (Original, Baseline, Run 1) x num_samples columns
    print("\n[INFO] Creating visualization...")
    fig, axes = plt.subplots(3, actual_samples, figsize=(actual_samples * 3, 10))
    
    # Handle single column case
    if actual_samples == 1:
        axes = axes.reshape(3, 1)
    
    # Plot each sample
    for i in range(actual_samples):
        # Convert from CHW to HWC format for matplotlib
        # PyTorch uses channels-first, matplotlib uses channels-last
        orig_img = np.transpose(test_images_denorm[i], (1, 2, 0))
        base_img = np.transpose(baseline_recon_denorm[i], (1, 2, 0))
        perc_img = np.transpose(perceptual_recon_denorm[i], (1, 2, 0))
        
        # Clip to [0, 1] range for display
        orig_img = np.clip(orig_img, 0, 1)
        base_img = np.clip(base_img, 0, 1)
        perc_img = np.clip(perc_img, 0, 1)
        
        # Row 1: Original images
        axes[0, i].imshow(orig_img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=14, fontweight='bold', pad=10)
        
        # Row 2: Baseline reconstructions (MSE only)
        axes[1, i].imshow(base_img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Baseline (MSE)\n14.45 dB PSNR', 
                                fontsize=14, fontweight='bold', pad=10)
        
        # Row 3: Run 1 reconstructions (MSE + LPIPS)
        axes[2, i].imshow(perc_img)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Run 1 (MSE+LPIPS)\n13.93 dB PSNR', 
                                fontsize=14, fontweight='bold', pad=10)
    
    # Add row labels on the left
    fig.text(0.02, 0.83, 'Original', rotation=90, 
            fontsize=16, fontweight='bold', va='center')
    fig.text(0.02, 0.50, 'Baseline\n(MSE)', rotation=90, 
            fontsize=16, fontweight='bold', va='center', ha='center')
    fig.text(0.02, 0.17, 'Run 1\n(Perceptual)', rotation=90, 
            fontsize=16, fontweight='bold', va='center', ha='center')
    
    # Overall title
    plt.suptitle('VGG16 Block1: Baseline vs Perceptual Loss Comparison\n' +
                'Run 1 trades PSNR (-0.52 dB) for perceptual quality (LPIPS -31.7%)',
                fontsize = 18, fontweight = 'bold', y = 0.98)
    
    plt.tight_layout(rect = [0.03, 0, 1, 0.96])
    
    # Save figure
    save_dir = Path('results/vgg16/figures_perceptual')
    save_dir.mkdir(parents=True, exist_ok = True)
    save_path = save_dir / 'baseline_vs_run1_comparison.png'
    
    plt.savefig(save_path, dpi = 200, bbox_inches = 'tight')
    plt.close()
    
    print(f"\n[SAVED] Comparison figure: {save_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print("\nMetrics Summary:")
    print("  Baseline (MSE):      14.45 dB PSNR | 0.530 SSIM | 0.398 LPIPS")
    print("  Run 1 (MSE+LPIPS):   13.93 dB PSNR | 0.565 SSIM | 0.272 LPIPS")
    print("\nKey Insight:")
    print("  Run 1 sacrifices pixel accuracy (PSNR) for better perceptual quality (LPIPS)")
    print("  Visual inspection should show sharper textures and better perceptual quality")
    print("="* 80 + "\n")


if __name__ == '__main__':
    # Entry point - generate comparison visualization
    generate_comparison_grid(num_samples=8)