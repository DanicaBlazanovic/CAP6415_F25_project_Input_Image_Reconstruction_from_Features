"""
Model Evaluation and Metrics Computation for Image Reconstruction
=================================================================

This module provides comprehensive evaluation functionality for trained image
reconstruction models. It computes quantitative metrics, generates visualizations,
and saves detailed results for analysis.

Key Functionality:
- Quantitative metrics: PSNR and SSIM on test set
- Image-level and aggregate statistics
- Side-by-side reconstruction comparisons
- Metric distribution histograms
- Reproducible evaluation with deterministic processing

Evaluation Pipeline:
    1. Load trained model checkpoint
    2. Run inference on test set (no gradients)
    3. Compute metrics for each image (PSNR, SSIM)
    4. Generate visualizations (comparisons, distributions)
    5. Save detailed results (CSV, JSON, PNG)
    6. Report aggregate statistics

Metrics Used:

PSNR (Peak Signal-to-Noise Ratio):
    Measures pixel-level fidelity between original and reconstruction.
    
    Formula:
        PSNR = 10 * log10(MAX^2 / MSE)
        
        Where:
        - MAX = maximum possible pixel value (1.0 for normalized images)
        - MSE = mean squared error between images
    
    Units: Decibels (dB)
    Range: 0 to infinity (higher is better)
    
    Interpretation:
        - >30 dB: Excellent quality (very close to original)
        - 25-30 dB: Good quality (minor differences)
        - 20-25 dB: Fair quality (noticeable differences)
        - <20 dB: Poor quality (significant differences)
    
    For this project:
        - Best model: 17.64 dB (ensemble)
        - Best single: 17.35 dB (VGG16 block1)
        - These are relatively low due to challenging reconstruction task

SSIM (Structural Similarity Index):
    Measures perceptual similarity considering luminance, contrast, and structure.
    
    Formula:
        SSIM = [luminance * contrast * structure]
        
        Where each component compares local patches between images.
    
    Units: Dimensionless
    Range: -1 to 1 (higher is better, 1 = identical)
    
    Interpretation:
        - >0.9: Excellent similarity
        - 0.7-0.9: Good similarity
        - 0.5-0.7: Fair similarity
        - <0.5: Poor similarity
    
    For this project:
        - Best model: 0.586 SSIM (ensemble)
        - Best single: 0.560 SSIM (VGG16 block1)
        - These reflect the difficulty of reconstruction from features

Why Both Metrics:
    PSNR: Pixel-accurate measure, sensitive to small errors
    SSIM: Perceptual measure, better matches human perception
    
    Both are complementary:
    - High PSNR + High SSIM = Excellent reconstruction
    - High PSNR + Low SSIM = Pixel-accurate but perceptually different
    - Low PSNR + High SSIM = Perceptually similar but pixel errors
    
    For image reconstruction, both matter:
    - PSNR ensures numerical fidelity
    - SSIM ensures visual quality

Output Directory Structure:
    results/
    ├── single/                              (single architecture models)
    │   └── evaluation_{exp_name}/
    │       ├── reconstructions/             (reconstructed image arrays)
    │       ├── metrics/
    │       │   ├── {exp_name}_detailed_metrics.csv    (per-image metrics)
    │       │   └── {exp_name}_summary.json            (aggregate stats)
    │       └── visualizations/
    │           ├── comparison_0000.png      (original vs reconstructed)
    │           ├── comparison_0001.png
    │           └── {exp_name}_metric_distributions.png
    └── ensemble/                            (ensemble models)
        └── evaluation_{exp_name}/
            └── (same structure as single)

Reproducibility Considerations:
    1. Deterministic inference: No dropout, batchnorm in eval mode
    2. Fixed ordering: Test set not shuffled
    3. Consistent metrics: Same skimage versions
    4. Saved configurations: All settings recorded in JSON
    5. Version tracking: Can compare across runs

Usage Example:
    >>> from models import SingleArchModel
    >>> from evaluation import evaluate_model, load_best_checkpoint
    >>> 
    >>> # Load model
    >>> config = {'exp_name': 'vgg16_block1_transposed_conv', ...}
    >>> model = SingleArchModel(config)
    >>> checkpoint_path = 'results/single/checkpoints_vgg16_block1_transposed_conv/best.pth'
    >>> model, val_loss = load_best_checkpoint(checkpoint_path, model, 'cuda')
    >>> 
    >>> # Evaluate
    >>> metrics = evaluate_model(model, test_loader, config, device='cuda')
    >>> print(f"PSNR: {metrics['psnr_mean']:.2f} dB")
    >>> print(f"SSIM: {metrics['ssim_mean']:.4f}")

Note on Naming:
    This module is in src/ folder as evaluation.py (imported by other modules).
    There is also scripts/evaluate.py which uses this module.
    Consider renaming to metrics.py or evaluation_utils.py to avoid confusion.

Author: Danica Blazanovic, Abbas Khan
Course: CAP6415 - Computer Vision, Fall 2025
Institution: Florida Atlantic University
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import json


def evaluate_model(model, test_loader, config, device='cuda'):
    """
    Evaluate trained model on test set with comprehensive metrics and visualizations.
    
    This function performs complete evaluation of a trained image reconstruction
    model. It computes PSNR and SSIM metrics for each test image, generates
    visualizations comparing original and reconstructed images, and saves
    detailed results for analysis.
    
    Evaluation Process:
    
    1. Setup Phase:
        - Set model to evaluation mode (disables dropout, batchnorm updates)
        - Move model to specified device (CUDA, MPS, or CPU)
        - Create output directories for results and visualizations
    
    2. Inference Phase:
        - Iterate through test set without gradient computation
        - Run forward pass to generate reconstructions
        - Move tensors to CPU for metric computation
        - Convert from PyTorch tensors to numpy arrays
    
    3. Metrics Computation Phase:
        - Calculate PSNR for each image (pixel-level fidelity)
        - Calculate SSIM for each image (perceptual similarity)
        - Store per-image metrics for detailed analysis
        - Accumulate metrics for aggregate statistics
    
    4. Visualization Phase:
        - Save first 50 images as side-by-side comparisons
        - Show original, reconstructed, and difference images
        - Include metric values in visualization titles
        - Create metric distribution histograms
    
    5. Results Saving Phase:
        - Save per-image metrics to CSV (detailed_metrics.csv)
        - Save aggregate statistics to JSON (summary.json)
        - Save all visualizations as PNG files
        - Print summary statistics to console
    
    Why Evaluation Mode:
        model.eval() is critical because it:
        - Disables dropout (uses all neurons instead of random subset)
        - Uses running statistics for batchnorm (not batch statistics)
        - Ensures deterministic behavior for reproducible results
        - Without eval(), results would vary between runs
    
    Why No Gradients:
        torch.no_grad() disables gradient computation:
        - Reduces memory usage significantly (no gradient storage)
        - Speeds up inference (no backward pass computation)
        - Evaluation doesn't need gradients (not training)
        - Test set is never used for parameter updates
    
    Metrics Computation Details:
        
        PSNR Calculation:
            - Computed using skimage.metrics.peak_signal_noise_ratio
            - data_range=1.0 indicates images in [0, 1] range
            - Formula: 10 * log10(1.0 / MSE)
            - Higher values indicate better reconstruction
        
        SSIM Calculation:
            - Computed using skimage.metrics.structural_similarity
            - data_range=1.0 for normalized images
            - channel_axis=2 indicates RGB channels in last dimension
            - Considers luminance, contrast, and structure
            - Value range: -1 to 1 (1 = perfect match)
    
    Image Format Conversions:
        
        PyTorch Tensor Format: (Batch, Channels, Height, Width) = (B, C, H, W)
            - Channels first for efficient GPU operations
            - Example: (16, 3, 224, 224) for batch of 16 RGB images
        
        NumPy/Visualization Format: (Height, Width, Channels) = (H, W, C)
            - Channels last for matplotlib and skimage
            - Example: (224, 224, 3) for single RGB image
        
        Conversion: tensor.transpose(1, 2, 0) converts CHW -> HWC
    
    Value Clipping:
        Images are clipped to [0, 1] range before metrics:
        - Neural networks may output slightly outside [0, 1]
        - Metrics computation requires valid range
        - Clipping ensures no invalid values
        - Minimal impact if model is well-trained
    
    Visualization Strategy:
        
        Only First 50 Images:
            - Full test set (100 images) would create too many files
            - 50 images sufficient for visual inspection
            - Reduces storage and processing time
            - All images still included in quantitative metrics
        
        Side-by-Side Layout:
            - Original | Reconstructed | Difference
            - Easy visual comparison
            - Difference amplified 5x for visibility
            - Metrics displayed on reconstruction
    
    Output Files:
        
        CSV File (detailed_metrics.csv):
            Columns: image_idx, psnr, ssim
            One row per test image
            Useful for per-image analysis and plotting
        
        JSON File (summary.json):
            Contains: mean, std, min, max for both metrics
            Also includes experiment name and model type
            Machine-readable for automated analysis
        
        PNG Files:
            comparison_XXXX.png: Individual image comparisons
            metric_distributions.png: Histograms of PSNR/SSIM
    
    Statistical Summary:
        Reports for each metric (PSNR, SSIM):
        - Mean: Average across all test images
        - Std: Standard deviation (variability)
        - Min: Worst-case performance
        - Max: Best-case performance
        
        These statistics characterize model performance comprehensively.
    
    Args:
        model (nn.Module): Trained PyTorch model to evaluate.
                          Must implement forward() that takes images and
                          returns reconstructed images.
                          Should be compatible with specified device.
        
        test_loader (DataLoader): PyTorch DataLoader for test set.
                                 Should provide batches of images.
                                 Typically batch_size=4-16 for evaluation.
                                 Should NOT be shuffled (consistent order).
        
        config (dict): Configuration dictionary containing:
                      - exp_name: Experiment identifier for file naming
                      - save_dir: Root directory for saving results
                      - architectures: Present if ensemble model
                      Other config values may be saved but not used here.
        
        device (str, optional): Device for computation.
                               Options: 'cuda', 'mps', 'cpu'
                               Default: 'cuda'
    
    Returns:
        dict: Dictionary containing evaluation metrics with keys:
            - experiment (str): Experiment name
            - model_type (str): 'single' or 'ensemble'
            - num_images (int): Number of images evaluated
            - psnr_mean (float): Mean PSNR in dB
            - psnr_std (float): PSNR standard deviation
            - psnr_min (float): Minimum PSNR
            - psnr_max (float): Maximum PSNR
            - ssim_mean (float): Mean SSIM
            - ssim_std (float): SSIM standard deviation
            - ssim_min (float): Minimum SSIM
            - ssim_max (float): Maximum SSIM
    
    Example:
        >>> config = {
        ...     'exp_name': 'vgg16_block1_transposed_conv',
        ...     'architecture': 'vgg16',
        ...     'vgg_block': 'block1',
        ...     'decoder_type': 'transposed_conv',
        ...     'save_dir': 'results'
        ... }
        >>> model = SingleArchModel(config)
        >>> model.load_state_dict(torch.load('checkpoint.pth')['model_state_dict'])
        >>> 
        >>> metrics = evaluate_model(model, test_loader, config, device='cuda')
        >>> 
        >>> print(f"PSNR: {metrics['psnr_mean']:.2f} +/- {metrics['psnr_std']:.2f} dB")
        >>> print(f"SSIM: {metrics['ssim_mean']:.4f} +/- {metrics['ssim_std']:.4f}")
        >>> print(f"Best PSNR: {metrics['psnr_max']:.2f} dB")
        >>> print(f"Worst PSNR: {metrics['psnr_min']:.2f} dB")
    
    Performance:
        Evaluation time depends on:
        - Test set size: 100 images for DIV2K
        - Batch size: Larger batches faster but more memory
        - Device: GPU much faster than CPU
        - Model complexity: Ensemble slower than single
        
        Typical times (100 images, batch_size=16):
        - CUDA GPU: 30-60 seconds
        - Apple MPS: 60-120 seconds
        - CPU: 300-600 seconds (not recommended)
    
    Note on Reproducibility:
        For identical results across runs:
        1. Use same model checkpoint
        2. Use same test set order (no shuffle)
        3. Set model to eval mode
        4. Use torch.no_grad()
        5. Use same device if possible (CPU results identical, GPU may vary slightly)
    """
    
    # Set model to evaluation mode
    # This disables dropout and uses running stats for batchnorm
    # Critical for deterministic, reproducible results
    model.eval()
    
    # Move model to specified device
    # All operations will run on this device
    model.to(device)
    
    # Extract experiment name for file naming
    exp_name = config['exp_name']
    
    # Determine model type from config
    # Ensemble models have 'architectures' key (list of networks)
    # Single models have 'architecture' key (single network)
    model_type = 'ensemble' if 'architectures' in config else 'single'
    
    # Create directory structure for saving results
    # Root: results/single/ or results/ensemble/
    # Subdirectory: evaluation_{exp_name}/
    eval_dir = Path(config['save_dir']) / model_type / f'evaluation_{exp_name}'
    
    # Subdirectories for different output types
    recon_dir = eval_dir / 'reconstructions'      # Reconstructed image arrays
    metrics_dir = eval_dir / 'metrics'            # CSV and JSON metrics
    vis_dir = eval_dir / 'visualizations'         # PNG visualizations
    
    # Create all directories (parents=True creates intermediate dirs)
    # exist_ok=True prevents error if directories already exist
    for dir_path in [recon_dir, metrics_dir, vis_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Print evaluation header for logging
    print(f"\n{'='*70}")
    print(f"EVALUATING: {exp_name}")
    print(f"{'='*70}\n")
    
    # Initialize storage for metrics
    # Lists to accumulate metrics for all images
    all_psnr = []  # PSNR value for each image
    all_ssim = []  # SSIM value for each image
    
    # List of dictionaries for per-image metrics
    # Will be converted to DataFrame and saved as CSV
    image_metrics = []
    
    # Disable gradient computation for inference
    # Reduces memory usage and speeds up computation
    # Gradients not needed since we're not training
    with torch.no_grad():
        # Iterate through test set batches
        # tqdm creates progress bar for user feedback
        for batch_idx, images in enumerate(tqdm(test_loader, desc='Evaluating')):
            # Move input images to device (GPU/CPU)
            images = images.to(device)
            
            # Run forward pass to generate reconstructions
            # Model in eval mode, so deterministic (no dropout)
            reconstructed = model(images)
            
            # Move results to CPU for metric computation
            # skimage metrics require numpy arrays on CPU
            images_np = images.cpu().numpy()
            reconstructed_np = reconstructed.cpu().numpy()
            
            # Process each image in the batch individually
            for i in range(images.shape[0]):
                # Calculate global image index across all batches
                img_idx = batch_idx * test_loader.batch_size + i
                
                # Extract single image and transpose to HWC format
                # Original: (C, H, W) in [0, 1] range, normalized
                # After transpose: (H, W, C) for skimage/matplotlib
                original = images_np[i].transpose(1, 2, 0)  # CHW -> HWC
                recon = reconstructed_np[i].transpose(1, 2, 0)
                
                # Clip values to valid [0, 1] range
                # Neural networks may output slightly outside range
                # Clipping ensures valid input for metrics
                original = np.clip(original, 0, 1)
                recon = np.clip(recon, 0, 1)
                
                # Calculate PSNR (Peak Signal-to-Noise Ratio)
                # Measures pixel-level reconstruction fidelity
                # data_range=1.0 indicates images in [0, 1] range
                # Returns value in decibels (dB), higher is better
                psnr_val = psnr(original, recon, data_range=1.0)
                
                # Calculate SSIM (Structural Similarity Index)
                # Measures perceptual similarity
                # data_range=1.0 for normalized images
                # channel_axis=2 indicates RGB channels in last dimension
                # Returns value in [0, 1] range, higher is better
                ssim_val = ssim(original, recon, data_range=1.0, channel_axis=2)
                
                # Append metrics to lists for aggregate statistics
                all_psnr.append(psnr_val)
                all_ssim.append(ssim_val)
                
                # Store per-image metrics for detailed analysis
                # Will be saved to CSV file
                image_metrics.append({
                    'image_idx': img_idx,      # Image identifier
                    'psnr': float(psnr_val),   # Convert numpy scalar to Python float
                    'ssim': float(ssim_val)
                })
                
                # Save reconstruction visualizations for first 50 images
                # Full test set would create too many files
                # 50 images sufficient for visual inspection
                if img_idx < 50:
                    save_reconstruction_comparison(
                        original, recon, img_idx, 
                        psnr_val, ssim_val, vis_dir
                    )
    
    # Calculate aggregate statistics across all images
    # These summarize overall model performance
    metrics_summary = {
        # Experiment identification
        'experiment': exp_name,
        'model_type': model_type,
        'num_images': len(all_psnr),
        
        # PSNR statistics (in dB)
        'psnr_mean': float(np.mean(all_psnr)),   # Average reconstruction quality
        'psnr_std': float(np.std(all_psnr)),     # Variability in quality
        'psnr_min': float(np.min(all_psnr)),     # Worst-case performance
        'psnr_max': float(np.max(all_psnr)),     # Best-case performance
        
        # SSIM statistics (dimensionless, 0-1)
        'ssim_mean': float(np.mean(all_ssim)),   # Average perceptual similarity
        'ssim_std': float(np.std(all_ssim)),     # Variability in similarity
        'ssim_min': float(np.min(all_ssim)),     # Worst perceptual match
        'ssim_max': float(np.max(all_ssim))      # Best perceptual match
    }
    
    # Save detailed per-image metrics to CSV
    # Useful for analyzing performance on individual images
    # Can identify which images are hardest to reconstruct
    df_metrics = pd.DataFrame(image_metrics)
    df_metrics.to_csv(metrics_dir / f'{exp_name}_detailed_metrics.csv', index=False)
    
    # Save summary statistics to JSON
    # Machine-readable format for automated analysis
    # Can be easily loaded and compared across experiments
    with open(metrics_dir / f'{exp_name}_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    # Print summary to console for immediate feedback
    # Using plain text instead of special characters
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS: {exp_name}")
    print(f"{'='*70}")
    print(f"Images evaluated: {len(all_psnr)}")
    print(f"PSNR: {metrics_summary['psnr_mean']:.4f} +/- {metrics_summary['psnr_std']:.4f}")
    print(f"      Range: [{metrics_summary['psnr_min']:.4f}, {metrics_summary['psnr_max']:.4f}]")
    print(f"SSIM: {metrics_summary['ssim_mean']:.4f} +/- {metrics_summary['ssim_std']:.4f}")
    print(f"      Range: [{metrics_summary['ssim_min']:.4f}, {metrics_summary['ssim_max']:.4f}]")
    print(f"{'='*70}\n")
    
    # Create distribution plots showing metric spread
    # Histograms reveal whether model is consistent or variable
    plot_metric_distributions(all_psnr, all_ssim, exp_name, vis_dir)
    
    # Return summary dictionary for programmatic access
    return metrics_summary


def save_reconstruction_comparison(original, reconstructed, idx, psnr_val, ssim_val, save_dir):
    """
    Create and save side-by-side comparison of original and reconstructed images.
    
    This function generates a three-panel visualization:
    1. Original image from test set
    2. Reconstructed image from model
    3. Absolute difference (amplified for visibility)
    
    The visualization provides immediate visual feedback on reconstruction quality
    and helps identify specific artifacts or failure modes.
    
    Visualization Design:
        
        Panel Layout: Original | Reconstructed | Difference
        
        Original Panel:
            - Ground truth image from test set
            - Shows what model should reconstruct
            - Reference for quality assessment
        
        Reconstructed Panel:
            - Model output after inference
            - Includes PSNR and SSIM values in title
            - Enables quick quality assessment
        
        Difference Panel:
            - Absolute difference: |original - reconstructed|
            - Amplified 5x for visibility (small errors hard to see)
            - Bright areas indicate large errors
            - Dark areas indicate accurate reconstruction
    
    Why Amplify Difference:
        Raw differences are typically small (0.01-0.05 range)
        Human eye cannot see such small variations
        5x amplification makes errors visible
        Still clamped to [0, 1] to prevent overflow
    
    Image Format:
        Input: NumPy arrays, shape (H, W, C), values in [0, 1]
        Output: PNG file, 150 DPI, tight bounding box
    
    File Naming:
        Format: comparison_XXXX.png
        XXXX: 4-digit zero-padded index (e.g., comparison_0001.png)
        Ensures alphabetical ordering matches image order
    
    Args:
        original (np.ndarray): Original image, shape (H, W, C), range [0, 1]
        reconstructed (np.ndarray): Reconstructed image, shape (H, W, C), range [0, 1]
        idx (int): Image index for file naming
        psnr_val (float): PSNR value in dB
        ssim_val (float): SSIM value (0-1)
        save_dir (Path): Directory to save visualization
    
    Returns:
        None: Saves PNG file to disk
    
    Example:
        >>> original = np.random.rand(224, 224, 3)
        >>> reconstructed = original + np.random.rand(224, 224, 3) * 0.1
        >>> save_reconstruction_comparison(
        ...     original, reconstructed, 5, 25.3, 0.85, Path('vis/')
        ... )
        >>> # Creates vis/comparison_0005.png
    """
    
    # Create figure with three subplots side by side
    # figsize=(15, 5) gives each panel 5x5 inches
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Display original image
    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')  # Remove axis ticks and labels
    
    # Panel 2: Display reconstructed image with metrics
    axes[1].imshow(reconstructed)
    axes[1].set_title(f'Reconstructed\nPSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}')
    axes[1].axis('off')
    
    # Panel 3: Display amplified difference
    # Compute absolute difference element-wise
    diff = np.abs(original - reconstructed)
    
    # Amplify by 5x to make small errors visible
    # Clamp to [0, 1] range to prevent overflow
    diff_amplified = np.clip(diff * 5, 0, 1)
    
    axes[2].imshow(diff_amplified)
    axes[2].set_title('Difference (5x)')
    axes[2].axis('off')
    
    # Adjust layout to prevent overlapping elements
    plt.tight_layout()
    
    # Save figure to disk
    # {idx:04d} formats index with leading zeros (e.g., 5 -> 0005)
    # dpi=150 provides good quality without excessive file size
    # bbox_inches='tight' removes extra whitespace
    plt.savefig(save_dir / f'comparison_{idx:04d}.png', dpi=150, bbox_inches='tight')
    
    # Close figure to free memory
    # Important when generating many plots
    plt.close()


def plot_metric_distributions(psnr_values, ssim_values, exp_name, save_dir):
    """
    Create and save histograms showing distribution of PSNR and SSIM metrics.
    
    This function generates a two-panel visualization showing how reconstruction
    quality varies across the test set. The histograms reveal:
    - Central tendency (where most values cluster)
    - Spread (how variable the quality is)
    - Shape (symmetric, skewed, multi-modal)
    - Outliers (unusually good or bad reconstructions)
    
    These distributions help assess model consistency and identify potential issues.
    
    Distribution Interpretation:
        
        Narrow Distribution (small std):
            - Model performs consistently across images
            - Indicates robust reconstruction
            - Desirable for production systems
        
        Wide Distribution (large std):
            - Performance varies significantly by image
            - Some images reconstruct well, others poorly
            - May indicate model limitations or data issues
        
        Skewed Distribution:
            - If skewed right: Few very good reconstructions
            - If skewed left: Few very poor reconstructions
            - Asymmetry suggests systematic biases
        
        Multi-Modal Distribution:
            - Multiple peaks indicate distinct groups
            - May correspond to different image types
            - Could reveal dataset characteristics
    
    Visualization Features:
        
        Histogram:
            - 50 bins for detailed distribution shape
            - Black edges for bin visibility
            - 0.7 alpha for slight transparency
        
        Mean Line:
            - Red dashed vertical line
            - Shows average performance
            - Easy reference point
        
        Grid:
            - Light gray grid (0.3 alpha)
            - Helps read values
            - Not distracting
        
        Title and Labels:
            - Clear axis labels with units
            - Experiment name in title
            - Legend shows mean value
    
    Statistical Context:
        
        PSNR (dB):
            Typical ranges for reconstruction:
            - >25 dB: Very good (rare for feature reconstruction)
            - 20-25 dB: Good
            - 15-20 dB: Fair (typical for this task)
            - <15 dB: Poor
        
        SSIM:
            Typical ranges:
            - >0.8: Very good
            - 0.6-0.8: Good
            - 0.4-0.6: Fair (typical for this task)
            - <0.4: Poor
    
    Args:
        psnr_values (list of float): PSNR value for each test image
        ssim_values (list of float): SSIM value for each test image
        exp_name (str): Experiment name for title and filename
        save_dir (Path): Directory to save visualization
    
    Returns:
        None: Saves PNG file to disk
    
    Example:
        >>> psnr_values = [15.2, 16.8, 14.9, 17.1, 15.5, ...]  # 100 values
        >>> ssim_values = [0.52, 0.58, 0.49, 0.61, 0.54, ...]  # 100 values
        >>> plot_metric_distributions(
        ...     psnr_values, ssim_values, 
        ...     'vgg16_block1_transposed_conv',
        ...     Path('results/visualizations/')
        ... )
        >>> # Creates metric_distributions.png showing both histograms
    """
    
    # Create figure with two subplots side by side
    # figsize=(12, 5) gives each subplot 6x5 inches
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left subplot: PSNR distribution
    # Plot histogram with 50 bins
    axes[0].hist(psnr_values, bins=50, edgecolor='black', alpha=0.7)
    
    # Add vertical line at mean value
    # Red dashed line makes it stand out
    # Label shows mean value rounded to 2 decimal places
    axes[0].axvline(np.mean(psnr_values), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(psnr_values):.2f}')
    
    # Set axis labels and title
    axes[0].set_xlabel('PSNR (dB)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('PSNR Distribution')
    
    # Add legend showing mean value
    axes[0].legend()
    
    # Add grid for easier value reading
    # alpha=0.3 makes grid subtle
    axes[0].grid(True, alpha=0.3)
    
    # Right subplot: SSIM distribution
    axes[1].hist(ssim_values, bins=50, edgecolor='black', alpha=0.7)
    
    # Add mean line
    # SSIM uses 4 decimal places for precision
    axes[1].axvline(np.mean(ssim_values), color='red', linestyle='--',
                    label=f'Mean: {np.mean(ssim_values):.4f}')
    
    # Set axis labels and title
    axes[1].set_xlabel('SSIM')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('SSIM Distribution')
    
    # Add legend and grid
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add overall title showing experiment name
    # fontsize=14, fontweight='bold' makes it prominent
    plt.suptitle(f'Metric Distributions: {exp_name}', fontsize=14, fontweight='bold')
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    # Save figure to disk
    # Filename includes experiment name for identification
    plt.savefig(save_dir / f'{exp_name}_metric_distributions.png', dpi=150, bbox_inches='tight')
    
    # Close figure to free memory
    plt.close()


def load_best_checkpoint(checkpoint_path, model, device):
    """
    Load trained model weights from checkpoint file.
    
    This function loads a saved model checkpoint and restores the model's
    parameters to their trained state. It handles device placement and
    sets the model to evaluation mode.
    
    Checkpoint File Format:
        PyTorch checkpoints are saved as dictionaries containing:
        - model_state_dict: Model parameters (weights and biases)
        - optimizer_state_dict: Optimizer state (for resuming training)
        - epoch: Training epoch when saved
        - val_loss: Validation loss at this checkpoint
        - Other metadata as needed
    
    Loading Process:
        1. Load checkpoint dictionary from disk
        2. Extract model_state_dict
        3. Load parameters into model
        4. Set model to evaluation mode
        5. Return model and validation loss
    
    Device Mapping:
        map_location argument handles device compatibility:
        - Checkpoint saved on CUDA, loading on CPU: Works
        - Checkpoint saved on CPU, loading on CUDA: Works
        - Checkpoint saved on CUDA:0, loading on CUDA:1: Works
        
        Without map_location, loading fails if devices don't match.
    
    Evaluation Mode:
        model.eval() is called to ensure:
        - Dropout layers disabled (use all neurons)
        - BatchNorm uses running statistics (not batch statistics)
        - Deterministic behavior for evaluation
        - Consistent results across runs
    
    Why Return Validation Loss:
        The validation loss from training helps contextualize test results:
        - Compare test performance to validation performance
        - Check for overfitting (test worse than validation)
        - Verify model loaded correctly (losses should be similar)
    
    Error Handling:
        If checkpoint file not found or corrupted:
        - PyTorch raises FileNotFoundError or RuntimeError
        - Caller should handle these exceptions
        - Always verify checkpoint path exists before calling
    
    Args:
        checkpoint_path (str or Path): Path to checkpoint file.
                                      Typically ends in .pth or .pt
                                      Should be absolute or relative to cwd
        
        model (nn.Module): Model instance with matching architecture.
                          Must have same structure as saved model.
                          Parameters will be overwritten.
        
        device (str): Device to load model onto.
                     Options: 'cuda', 'mps', 'cpu'
                     Should match device used for evaluation.
    
    Returns:
        tuple: (model, val_loss)
            model (nn.Module): Model with loaded parameters in eval mode
            val_loss (float or None): Validation loss from checkpoint
                                     None if not saved in checkpoint
    
    Example:
        >>> from models import SingleArchModel
        >>> 
        >>> # Create model instance
        >>> config = {'architecture': 'vgg16', 'vgg_block': 'block1', ...}
        >>> model = SingleArchModel(config)
        >>> 
        >>> # Load trained weights
        >>> checkpoint_path = 'results/single/checkpoints_vgg16_block1/best.pth'
        >>> model, val_loss = load_best_checkpoint(checkpoint_path, model, 'cuda')
        >>> 
        >>> print(f"Loaded model with validation loss: {val_loss:.6f}")
        >>> print(f"Model in eval mode: {not model.training}")  # True
        >>> 
        >>> # Now ready for evaluation
        >>> with torch.no_grad():
        ...     output = model(input)
    
    Note on State Dict:
        The state_dict is a Python dictionary mapping parameter names to tensors:
        {
            'encoder.conv1.weight': tensor(...),
            'encoder.conv1.bias': tensor(...),
            'decoder.layer1.weight': tensor(...),
            ...
        }
        
        Model architecture must exactly match for successful loading.
        Extra keys in checkpoint: Warning, ignored
        Missing keys in checkpoint: Error, loading fails
    """
    
    # Print loading message for user feedback
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint dictionary from disk
    # map_location ensures compatibility across devices
    # CPU can load CUDA checkpoints and vice versa
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model parameters from checkpoint dictionary
    # load_state_dict copies parameters into model
    # strict=True (default) requires exact parameter match
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    # Disables dropout, uses running stats for batchnorm
    # Critical for reproducible evaluation
    model.eval()
    
    # Extract validation loss if saved in checkpoint
    # get() returns None if 'val_loss' key doesn't exist
    # Older checkpoints may not have validation loss saved
    val_loss = checkpoint.get('val_loss', None)
    
    # Return loaded model and validation loss
    return model, val_loss
