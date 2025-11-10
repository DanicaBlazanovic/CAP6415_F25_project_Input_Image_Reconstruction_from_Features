"""
Evaluate Run 1: VGG16 Block1 with Perceptual Loss on Test Set

This script evaluates the perceptual loss model (Run 1) on the held-out test set
to measure final reconstruction quality and compare with baseline.

Process:
1. Load best trained model from checkpoint
2. Evaluate on 100-image test set (DIV2K_valid_HR)
3. Compute metrics: PSNR, SSIM, LPIPS, MSE
4. Compare with baseline results (MSE-only loss)
5. Save metrics to CSV for documentation

Baseline (MSE loss only):
- PSNR: 14.45 dB
- SSIM: 0.530
- LPIPS: 0.398

Expected Run 1 improvement: +2-3 dB PSNR from perceptual loss

Cross-platform compatible: Windows, Mac (M1/M2), Linux
"""

import sys
from pathlib import Path

# Add src directory to Python path for imports
# This allows us to import from src/ without installing as package
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
from models import FeatureExtractor, AttentionDecoder
from dataset import get_dataloaders
from evaluate import MetricsCalculator, print_metrics, save_metrics


def main():
    """
    Main evaluation function for Run 1.
    
    Steps:
    1. Setup: Load configuration and detect device
    2. Data: Load test set (100 images)
    3. Model: Create encoder and decoder, load trained weights
    4. Evaluate: Run inference on test set and compute metrics
    5. Compare: Show improvement over baseline
    6. Save: Store results to CSV
    """
    
    # Print header
    print("\n" + "="*80)
    print("EVALUATING RUN 1: VGG16 BLOCK1 WITH PERCEPTUAL LOSS")
    print("="*80)
    
    # Configuration for Run 1
    # These must match the training configuration
    architecture = 'vgg16'  # VGG16 architecture
    layer_name = 'block1'   # Extract features from block1 (112x112 features)
    
    # Path to best model checkpoint from training
    # This checkpoint was saved at epoch with lowest validation loss
    checkpoint_path = 'results/vgg16/checkpoints_perceptual/vgg16_block1_perceptual_best.pth'
    
    # Detect available device (cross-platform)
    # Note: MPS has memory issues with large feature maps (112x112x64)
    # so we force CPU on Mac for VGG16 block1
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"[INFO] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("[INFO] Using CPU (VGG16 block1 requires significant memory)")
        print("[INFO] MPS on Mac M1 causes memory errors with 112x112 features")
    
    print(f"[INFO] Device: {device}\n")
    
    # Load test dataset
    # We only need the test set for evaluation
    # Test set: 100 images from DIV2K_valid_HR (held-out, never seen during training)
    print("[INFO] Loading test set...")
    _, _, test_loader = get_dataloaders(
        batch_size = 1,      # Batch size for evaluation (can be larger than training)
        num_workers = 2,     # Parallel data loading workers
        limit = None         # Use all 100 test images
    )
    print(f"[INFO] Test samples: {len(test_loader.dataset)}\n")
    
    # Create frozen encoder (VGG16 block1)
    # Encoder extracts 64-channel features at 112x112 spatial resolution
    # Pre-trained on ImageNet, weights frozen (no training)
    print("[INFO] Creating VGG16 block1 encoder (frozen)...")
    encoder = FeatureExtractor(
        architecture=architecture,
        layer_name=layer_name
    )
    
    # Determine feature dimensions for decoder initialization
    # We pass a dummy input through encoder to get output shape
    # This is more robust than hardcoding dimensions
    print("[INFO] Determining feature dimensions...")
    with torch.no_grad():  # No gradients needed for dummy forward pass
        dummy_input = torch.randn(1, 3, 224, 224)  # Single RGB image
        dummy_features = encoder(dummy_input)
        feat_channels, feat_h, feat_w = dummy_features.shape[1:]  # Extract C, H, W
    
    print(f"[INFO] Feature shape: {feat_channels} x {feat_h} x {feat_w}")
    
    # Create decoder with same architecture as training
    # Decoder takes encoder features and reconstructs 224x224 RGB image
    # AttentionDecoder uses transformer blocks for better reconstruction
    print("[INFO] Creating attention-based decoder...")
    decoder = AttentionDecoder(
        input_channels = feat_channels,  # 64 channels for VGG16 block1
        input_size = feat_h,             # 112 for VGG16 block1
        output_size = 224,               # Reconstruct to 224x224
        num_blocks = 4                   # 4 transformer blocks (same as training)
    )
    
    # Count trainable parameters in decoder
    num_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"[INFO] Decoder parameters: {num_params:,}")
    
    # Load trained decoder weights from checkpoint
    # Checkpoint contains:
    # - decoder_state_dict: trained model parameters
    # - epoch: which epoch this checkpoint is from
    # - val_loss: validation loss at this epoch
    # - history: training history (losses, learning rates)
    print(f"\n[INFO] Loading checkpoint: {checkpoint_path}")
    
    # map_location ensures checkpoint loads on available device
    # Important for loading GPU-trained model on CPU or vice versa
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load decoder weights into model
    # This restores the trained parameters
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    # Display checkpoint information
    print(f"[INFO] Loaded model from epoch {checkpoint['epoch']}")
    print(f"[INFO] Training validation loss: {checkpoint['val_loss']:.6f}")
    print(f"[INFO] This was the best model (lowest validation loss)\n")
    
    # Evaluate on test set
    # MetricsCalculator computes PSNR, SSIM, LPIPS, MSE
    # Test set was never seen during training (true generalization test)
    print("[INFO] Evaluating on test set (100 images)...")
    print("[INFO] Computing PSNR, SSIM, LPIPS, and MSE metrics...")
    
    # Initialize metrics calculator with LPIPS model
    # LPIPS uses AlexNet features to measure perceptual similarity
    metrics_calc = MetricsCalculator(device=device)
    
    # Run evaluation on entire test set
    # This will:
    # 1. For each batch: extract features, reconstruct, denormalize
    # 2. Compute metrics per batch
    # 3. Aggregate metrics across all batches
    results = metrics_calc.evaluate_model(
        encoder=encoder,
        decoder=decoder,
        dataloader=test_loader,
        device=device
    )
    
    # Print results in formatted table
    # Shows mean and std for each metric
    print_metrics(results, title="Run 1: Perceptual Loss - Test Set Results")
    
    # Compare with baseline results
    # Baseline used MSE loss only (no perceptual loss)
    # These baseline numbers come from your original experiments
    print("\n" + "="*80)
    print("COMPARISON WITH BASELINE")
    print("="*80)
    
    # Baseline metrics (from your baseline experiments)
    baseline_psnr = 14.45  # dB
    baseline_ssim = 0.530
    baseline_lpips = 0.398
    
    print(f"Baseline (MSE only):        {baseline_psnr:.2f} dB PSNR, "
          f"{baseline_ssim:.3f} SSIM, {baseline_lpips:.3f} LPIPS")
    print(f"Run 1 (MSE + LPIPS):        {results['psnr_mean']:.2f} dB PSNR, "
          f"{results['ssim_mean']:.3f} SSIM, {results['lpips_mean']:.3f} LPIPS")
    
    # Calculate improvements
    # PSNR and SSIM: higher is better, so positive change is improvement
    # LPIPS: lower is better, so negative change is improvement
    improvement_psnr = results['psnr_mean'] - baseline_psnr
    improvement_ssim = results['ssim_mean'] - baseline_ssim
    improvement_lpips = baseline_lpips - results['lpips_mean']  # Reversed (lower is better)
    
    print(f"\nImprovement over baseline:")
    print(f"  PSNR:  {improvement_psnr:+.2f} dB")
    print(f"  SSIM:  {improvement_ssim:+.3f}")
    print(f"  LPIPS: {improvement_lpips:+.3f} (negative means better perceptual quality)")
    
    # Interpretation of results
    print("\nInterpretation:")
    if improvement_psnr > 1.0:
        print(f"  ✓ Significant PSNR improvement ({improvement_psnr:.2f} dB)")
    elif improvement_psnr > 0:
        print(f"  ~ Modest PSNR improvement ({improvement_psnr:.2f} dB)")
    else:
        print(f"  ✗ PSNR decreased ({improvement_psnr:.2f} dB)")
    
    if improvement_ssim > 0.05:
        print(f"  ✓ Significant SSIM improvement ({improvement_ssim:.3f})")
    elif improvement_ssim > 0:
        print(f"  ~ Modest SSIM improvement ({improvement_ssim:.3f})")
    else:
        print(f"  ✗ SSIM decreased ({improvement_ssim:.3f})")
    
    if improvement_lpips > 0.05:
        print(f"  ✓ Significant perceptual improvement ({improvement_lpips:.3f} LPIPS reduction)")
    elif improvement_lpips > 0:
        print(f"  ~ Modest perceptual improvement ({improvement_lpips:.3f} LPIPS reduction)")
    else:
        print(f"  ✗ Perceptual quality decreased ({improvement_lpips:.3f} LPIPS increase)")
    
    print("="*80 + "\n")
    
    # Save metrics to CSV for documentation
    # This creates: results/vgg16/metrics/vgg16_block1_perceptual_metrics.csv
    # CSV format allows easy import into papers, reports, or analysis tools
    config = {
        'architecture': architecture,
        'layer_name': layer_name,
        'decoder_type': 'perceptual'  # Distinguishes from baseline 'attention'
    }
    
    print("[INFO] Saving metrics to CSV...")
    save_metrics(results, config, save_dir='results')
    
    print("\n✓ Evaluation complete!")
    print("\nNext steps:")
    print("1. Generate reconstruction visualizations")
    print("2. Update README with Run 1 results")
    print("3. Commit results to GitHub")


if __name__ == '__main__':
    # Entry point when script is run directly
    # Calls main() function to start evaluation
    main()