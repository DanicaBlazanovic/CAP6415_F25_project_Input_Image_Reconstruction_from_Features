"""
Utility Functions for Image Reconstruction Project
==================================================

This module provides helper functions for common tasks throughout the project:
- Device detection and configuration
- Visualization of results (grids, plots, heatmaps)
- Model information and parameter counting
- Results aggregation and comparison
- Reproducibility through random seed setting

These utilities are used across training, evaluation, and analysis scripts
to maintain consistency and reduce code duplication.

Key Functionality Categories:

1. Device Management:
   - Auto-detect best available device (CUDA, MPS, CPU)
   - Cross-platform compatibility

2. Visualization:
   - Reconstruction comparison grids
   - Training history plots
   - Layer and architecture comparisons
   - Metrics heatmaps

3. Model Information:
   - Parameter counting (trainable vs frozen)
   - Model configuration documentation
   - Architecture summaries

4. Results Management:
   - CSV export for metrics
   - Comprehensive results tables
   - Cross-experiment comparisons

5. Reproducibility:
   - Random seed setting for all libraries
   - Deterministic behavior configuration

Cross-Platform Design:
    All functions use pathlib.Path for file operations, ensuring code works
    identically on Windows, macOS, and Linux without modification.

Author: Danica Blazanovic, Abbas Khan
Course: CAP6415 - Computer Vision, Fall 2025
Institution: Florida Atlantic University
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from dataset import denormalize


def get_device():
    """
    Auto-detect best available device for PyTorch computations.
    
    This function automatically selects the fastest available hardware for
    training and inference. It checks for available accelerators in order
    of performance and returns the appropriate torch.device object.
    
    Device Priority (fastest to slowest):
    
    1. CUDA (NVIDIA GPU):
        - Hardware: NVIDIA GPUs (RTX 6000, A100, etc.)
        - Performance: Fastest for deep learning
        - Best for: Training large models, batch processing
        - Typical speedup: 10-100x vs CPU
        - Detection: torch.cuda.is_available()
    
    2. MPS (Apple Metal Performance Shaders):
        - Hardware: Apple Silicon (M1, M2, M3, M4 chips)
        - Performance: Fast, but slower than CUDA
        - Best for: Mac users without NVIDIA GPUs
        - Typical speedup: 5-20x vs CPU
        - Detection: torch.backends.mps.is_available()
        - Note: Available in PyTorch 1.12+ on macOS 12.3+
    
    3. CPU:
        - Hardware: Any CPU (Intel, AMD, Apple)
        - Performance: Slowest but always available
        - Best for: Small-scale testing, debugging
        - Fallback: When no GPU available
    
    Why Auto-Detection Matters:
        - Portability: Same code runs on any hardware
        - Performance: Always uses fastest available device
        - Simplicity: No manual device configuration needed
        - Development: Easy to test on laptop, train on server
    
    CUDA Details:
        When CUDA is available, the function also prints the GPU name
        for verification. This helps ensure you're using the expected GPU
        in multi-GPU systems or cloud environments.
        
        Example output: "Using CUDA: NVIDIA RTX 6000 Ada Generation"
    
    MPS Details:
        Apple's Metal Performance Shaders provide GPU acceleration on
        Apple Silicon. While not as mature as CUDA, MPS provides
        significant speedup over CPU for Mac users.
        
        Note: Some operations may be slower on MPS than CPU due to
        limited optimization. This typically improves with PyTorch updates.
    
    CPU Fallback:
        If no GPU is available, the function falls back to CPU with a
        warning message. This allows code to run anywhere but warns users
        that training will be significantly slower.
    
    Returns:
        torch.device: Device object for use with model.to(device)
                     One of: torch.device('cuda'), torch.device('mps'),
                     or torch.device('cpu')
    
    Example:
        >>> device = get_device()
        Using CUDA: NVIDIA RTX 6000 Ada Generation
        >>> model = model.to(device)
        >>> data = data.to(device)
        >>> output = model(data)  # Runs on GPU
    
    Usage Pattern:
        Typical workflow:
        1. Call get_device() once at start of script
        2. Move model to device with model.to(device)
        3. Move each data batch to device in training loop
        4. All operations automatically run on selected device
    
    Note on Multi-GPU:
        This function returns the first CUDA device (cuda:0) if multiple
        GPUs are available. For multi-GPU training, use torch.nn.DataParallel
        or torch.nn.parallel.DistributedDataParallel.
    
    Note on Determinism:
        GPU operations may have slight non-determinism even with fixed seeds.
        For perfect reproducibility, use CPU. For practical work, GPU
        non-determinism is typically <0.1% variation.
    """
    
    # Check if CUDA (NVIDIA GPU) is available
    # CUDA provides the best performance for deep learning tasks
    if torch.cuda.is_available():
        # Create CUDA device object
        device = torch.device('cuda')
        
        # Print GPU name for user information
        # get_device_name(0) returns name of first GPU
        # Useful for verifying correct GPU in multi-GPU systems
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    
    # Check if MPS (Metal Performance Shaders) is available
    # MPS is Apple's GPU acceleration framework for M1/M2/M3/M4 chips
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Create MPS device object
        device = torch.device('mps')
        
        # Print confirmation message
        print("Using Apple MPS (Metal Performance Shaders)")
    
    # Fall back to CPU if no GPU available
    # CPU is slower but works on all systems
    else:
        # Create CPU device object
        device = torch.device('cpu')
        
        # Print warning that CPU will be slower
        # This alerts users to performance implications
        print("Using CPU (this will be slower)")
    
    # Return device object for use throughout code
    return device


def save_comparison_grid(original, reconstructed, config, num_images=8):
    """
    Save side-by-side comparison grid of original and reconstructed images.
    
    This function creates a visualization with two rows showing original images
    (top row) and their reconstructions (bottom row). This provides immediate
    visual feedback on reconstruction quality and helps identify artifacts.
    
    Visualization Layout:
        
        Row 1: [Original 1] [Original 2] [Original 3] ... [Original N]
        Row 2: [Recon 1]    [Recon 2]    [Recon 3]    ... [Recon N]
        
        Where N = num_images (default 8)
    
    Why This Visualization:
        
        Side-by-side comparison:
            - Easy to spot differences between original and reconstruction
            - Natural way for humans to compare images
            - Common in image reconstruction papers
        
        Multiple images:
            - Shows model performance across different inputs
            - Reveals consistency or inconsistency
            - Identifies which types of images are hardest
        
        Top/bottom layout:
            - Aligns images vertically for direct comparison
            - Easy to scan horizontally across batch
            - Fits well in typical page widths
    
    File Organization:
        Files are saved in experiment-specific directories:
        results/{architecture}/figures/{exp_name}_reconstruction.png
        
        This organization:
        - Groups results by architecture for easy comparison
        - Unique names prevent overwriting
        - Figures directory separates visualizations from metrics
    
    Denormalization:
        Images in PyTorch are normalized using ImageNet statistics for
        compatibility with pre-trained models:
        - Mean: [0.485, 0.456, 0.406]
        - Std: [0.229, 0.224, 0.225]
        
        Before visualization, we must denormalize back to [0, 1] range:
        - Matplotlib expects [0, 1] for proper display
        - Without denormalization, colors would be wrong
        - denormalize() function handles this automatically
    
    Image Format Handling:
        
        PyTorch format: (Batch, Channels, Height, Width) = (B, C, H, W)
        - Channels first for efficient GPU operations
        - Example: (16, 3, 224, 224) for batch of 16 RGB images
        
        Matplotlib format: (Height, Width, Channels) = (H, W, C)
        - Channels last for visualization
        - Example: (224, 224, 3) for single RGB image
        
        Conversion: tensor.permute(1, 2, 0) converts CHW -> HWC
    
    Args:
        original (torch.Tensor): Original images from dataset.
                                Shape: (B, C, H, W)
                                Normalized with ImageNet statistics
                                Typical: (4-16, 3, 224, 224)
        
        reconstructed (torch.Tensor): Model reconstructions.
                                     Shape: (B, C, H, W)
                                     Same normalization as original
        
        config (dict): Configuration dictionary containing:
                      - architecture: Network architecture name (e.g., 'vgg16')
                      - layer_name: Layer used for features (e.g., 'block1')
                      - decoder_type: Decoder architecture (e.g., 'transposed_conv')
        
        num_images (int, optional): Number of image pairs to display.
                                   Will be capped by batch size if larger.
                                   Default: 8
    
    Returns:
        None: Saves PNG file to disk, prints save confirmation
    
    Example:
        >>> # During training
        >>> images = next(iter(train_loader))  # Get batch
        >>> reconstructed = model(images.to(device))
        >>> 
        >>> config = {
        ...     'architecture': 'vgg16',
        ...     'layer_name': 'block1',
        ...     'decoder_type': 'transposed_conv'
        ... }
        >>> 
        >>> save_comparison_grid(images, reconstructed, config, num_images=8)
        [SAVED] Reconstruction comparison: results/vgg16/figures/...
    
    Note on Memory:
        This function moves tensors to CPU for visualization. This is necessary
        because matplotlib runs on CPU. GPU memory is freed after conversion.
    
    Note on Image Count:
        If batch size is smaller than num_images, only batch_size images are
        displayed. For example, if batch_size=4 but num_images=8, only 4
        image pairs will be shown.
    """
    
    # Create unique experiment name from configuration
    # Format: {architecture}_{layer}_{decoder}
    # Example: "vgg16_block1_transposed_conv"
    arch_name = config['architecture']
    layer_name = config['layer_name']
    decoder_type = config['decoder_type']
    experiment_name = f"{arch_name}_{layer_name}_{decoder_type}"
    
    # Create save path in architecture-specific directory
    # Structure: results/vgg16/figures/vgg16_block1_transposed_conv_reconstruction.png
    # This organizes visualizations by architecture for easy comparison
    save_path = Path(f'results/{arch_name}/figures/{experiment_name}_reconstruction.png')
    
    # Create directory if it doesn't exist
    # parents=True creates intermediate directories (results/, results/vgg16/, etc.)
    # exist_ok=True doesn't raise error if directory already exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Denormalize images from ImageNet normalization to [0, 1] range
    # .cpu() moves tensors from GPU to CPU for numpy conversion
    # denormalize() reverses: (img - mean) / std -> img
    original = denormalize(original.cpu())
    reconstructed = denormalize(reconstructed.cpu())
    
    # Determine how many images to show
    # Use minimum of requested number and available images in batch
    # If batch_size=4 but num_images=8, only show 4 images
    num_images = min(num_images, original.shape[0])
    
    # Create figure with matplotlib
    # Subplot grid: 2 rows (original, reconstructed), num_images columns
    # figsize: 3 inches per image width, 6 inches total height
    # This creates readable images without excessive file size
    fig, axes = plt.subplots(2, num_images, figsize=(3 * num_images, 6))
    
    # Iterate over each image pair
    for i in range(num_images):
        # TOP ROW: Original images
        
        # Extract single image from batch
        # Convert from PyTorch format [C, H, W] to matplotlib format [H, W, C]
        # permute(1, 2, 0) transposes dimensions: (C, H, W) -> (H, W, C)
        img_orig = original[i].permute(1, 2, 0).numpy()
        
        # Display image in top row subplot
        # imshow expects [H, W, C] array with values in [0, 1]
        axes[0, i].imshow(img_orig)
        
        # Hide axis ticks and labels for clean appearance
        # Images don't need coordinate axes
        axes[0, i].axis('off')
        
        # Add title to first image only to save space
        # Other images are clearly in the same row
        if i == 0:
            axes[0, i].set_title('Original', fontsize=12, fontweight='bold')
        
        # BOTTOM ROW: Reconstructed images
        
        # Same process as original images
        img_recon = reconstructed[i].permute(1, 2, 0).numpy()
        axes[1, i].imshow(img_recon)
        axes[1, i].axis('off')
        
        # Add title to first reconstructed image
        # Include experiment name in title for identification
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=12, fontweight='bold')
    
    # Adjust spacing between subplots for clean appearance
    # tight_layout() automatically optimizes spacing to prevent overlap
    plt.tight_layout()
    
    # Save figure to disk
    # dpi=150: High quality (default is 100)
    # Higher dpi = larger file but sharper image
    # 150 is good balance for most uses
    # bbox_inches='tight': Remove extra whitespace around figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Close figure to free memory
    # Important when generating many figures in batch
    # Without close(), figures accumulate in memory
    plt.close()
    
    # Print confirmation message with save path
    print(f"[SAVED] Reconstruction comparison: {save_path}")


def plot_training_history(history, config):
    """
    Plot training and validation loss curves over epochs.
    
    This function generates a visualization of the training process showing
    how losses evolved over time. The plots help diagnose training behavior
    and identify issues like overfitting or convergence problems.
    
    Plot Layout:
        Two side-by-side subplots:
        
        Left: Loss Curves
            - Blue line: Training loss
            - Red line: Validation loss
            - Shows model performance over time
        
        Right: Learning Rate Schedule
            - Green line: Learning rate
            - Log scale y-axis
            - Shows optimizer adjustments
    
    Training Diagnostics from Plots:
    
    Good Training:
        - Both losses decrease smoothly
        - Training and validation losses stay close
        - Learning rate decreases when losses plateau
        - Convergence by end of training
    
    Overfitting:
        - Training loss continues decreasing
        - Validation loss starts increasing
        - Large gap between train and validation
        - Sign: Need regularization or early stopping
    
    Underfitting:
        - Both losses high at end
        - Both losses still decreasing
        - Sign: Need more epochs or larger model
    
    Learning Rate Issues:
        - Loss spikes: Learning rate too high
        - Slow convergence: Learning rate too low
        - Plateaus: Need LR reduction (scheduler should handle)
    
    Loss Interpretation:
        
        Combined Loss (MSE + LPIPS):
            - Total loss = 0.5 * MSE + 0.5 * LPIPS (typical weights)
            - Lower is better
            - Typical range: 0.1-0.5 for this project
        
        Why Log Scale for Learning Rate:
            - Learning rates span orders of magnitude (1e-4 to 1e-6)
            - Linear scale would compress small changes
            - Log scale shows all changes clearly
            - Step-downs appear as downward steps
    
    File Organization:
        Saved to: results/{architecture}/figures/{exp_name}_training.png
        This keeps training plots with reconstruction visualizations.
    
    Args:
        history (dict): Training history containing:
                       - train_loss (list): Training loss per epoch
                       - val_loss (list): Validation loss per epoch
                       - lr (list, optional): Learning rate per epoch
                       All lists should have same length (one value per epoch)
        
        config (dict): Configuration with experiment identification:
                      - architecture: Network architecture
                      - layer_name: Feature layer
                      - decoder_type: Decoder architecture
    
    Returns:
        None: Saves PNG file to disk, prints confirmation
    
    Example:
        >>> history = {
        ...     'train_loss': [0.45, 0.32, 0.28, 0.25, 0.23, 0.22, 0.21],
        ...     'val_loss': [0.48, 0.35, 0.30, 0.27, 0.26, 0.25, 0.25],
        ...     'lr': [1e-4, 1e-4, 1e-4, 5e-5, 5e-5, 2.5e-5, 2.5e-5]
        ... }
        >>> 
        >>> config = {
        ...     'architecture': 'vgg16',
        ...     'layer_name': 'block1',
        ...     'decoder_type': 'transposed_conv'
        ... }
        >>> 
        >>> plot_training_history(history, config)
        [SAVED] Training history: results/vgg16/figures/...
    
    Note on Missing Learning Rate:
        If 'lr' key is not in history, the right subplot is skipped.
        This handles older training runs that didn't track learning rate.
    """
    
    # Create experiment name from configuration
    # Format: {architecture}_{layer}_{decoder}
    arch_name = config['architecture']
    layer_name = config['layer_name']
    decoder_type = config['decoder_type']
    experiment_name = f"{arch_name}_{layer_name}_{decoder_type}"
    
    # Create save path in architecture-specific directory
    # Example: results/vgg16/figures/vgg16_block1_transposed_conv_training.png
    save_path = Path(f'results/{arch_name}/figures/{experiment_name}_training.png')
    
    # Create directory if needed
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure with two side-by-side subplots
    # figsize=(14, 5): 14 inches wide, 5 inches tall
    # Gives each subplot good space without being too large
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Create list of epoch numbers for x-axis
    # Start at 1 (not 0) because humans count epochs from 1
    # Length matches training history
    epochs = range(1, len(history['train_loss']) + 1)
    
    # SUBPLOT 1: Loss curves
    
    # Plot training loss in blue
    # 'b-': blue solid line
    # linewidth=2: Thicker line for visibility
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    
    # Plot validation loss in red
    # 'r-': red solid line
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    
    # Configure subplot
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    
    # Add legend to identify lines
    # fontsize=11 for readability
    ax1.legend(fontsize=11)
    
    # Add grid for easier value reading
    # alpha=0.3: Light gray, not distracting
    ax1.grid(True, alpha=0.3)
    
    # SUBPLOT 2: Learning rate schedule
    
    # Check if learning rate was tracked in history
    if 'lr' in history:
        # Plot learning rate in green
        # 'g-': green solid line
        ax2.plot(epochs, history['lr'], 'g-', linewidth=2)
        
        # Configure subplot
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Learning Rate', fontsize=12)
        ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        
        # Use log scale for y-axis
        # Learning rates are small numbers (1e-3, 1e-4, 1e-5)
        # Log scale shows changes clearly
        # Reductions appear as downward steps
        ax2.set_yscale('log')
        
        # Add grid
        ax2.grid(True, alpha=0.3)
    
    # Adjust spacing between subplots
    # tight_layout() optimizes spacing to prevent overlap
    plt.tight_layout()
    
    # Save figure to disk
    # dpi=150: Good quality for viewing and reports
    # bbox_inches='tight': Remove extra whitespace
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Close figure to free memory
    plt.close()
    
    # Print confirmation message
    print(f"[SAVED] Training history: {save_path}")


def plot_layer_comparison(results_dict, architecture, metric='psnr_mean'):
    """
    Create bar chart comparing metric across layers for one architecture.
    
    This visualization answers the question: "Which layer preserves the most
    information for reconstruction?" It shows how reconstruction quality varies
    with network depth.
    
    Typical Pattern:
        
        Shallow Layers (layer1, block1):
            - High spatial resolution
            - More low-level features (edges, textures)
            - Better reconstruction of fine details
            - Less semantic information
        
        Middle Layers (layer2, block2-3):
            - Balanced spatial resolution
            - Mix of low and high-level features
            - Often BEST for reconstruction
            - Good balance of detail and semantics
        
        Deep Layers (layer4, block5):
            - Low spatial resolution
            - High-level semantic features
            - Poor reconstruction of details
            - Best for classification, not reconstruction
    
    Expected Results:
        For most architectures:
        - layer1/block1: Good (high resolution)
        - layer2/block2: Best (optimal balance)
        - layer3/block3: Worse (losing detail)
        - layer4/block4: Worst (too abstract)
    
    Why This Matters:
        Understanding layer-wise performance helps:
        - Select optimal layer for reconstruction
        - Understand what information is preserved
        - Design better decoder architectures
        - Explain reconstruction limitations
    
    Visualization Design:
        - Bar chart for easy comparison
        - Steel blue color (professional)
        - Horizontal grid lines for value reading
        - Y-axis shows metric value
        - X-axis shows layer names
    
    Args:
        results_dict (dict): Mapping from layer names to result dictionaries.
                            Example: {
                                'layer1': {'psnr_mean': 22.5, 'ssim_mean': 0.75, ...},
                                'layer2': {'psnr_mean': 24.1, 'ssim_mean': 0.82, ...},
                                'layer3': {'psnr_mean': 21.8, 'ssim_mean': 0.70, ...},
                                'layer4': {'psnr_mean': 18.5, 'ssim_mean': 0.55, ...}
                            }
        
        architecture (str): Architecture name (e.g., 'vgg16', 'resnet34')
                           Used for title and file naming
        
        metric (str, optional): Metric to plot.
                               Options: 'psnr_mean', 'ssim_mean', 'lpips_mean', 'mse_mean'
                               Default: 'psnr_mean'
    
    Returns:
        None: Saves PNG file to disk, prints confirmation
    
    Example:
        >>> results = {
        ...     'block1': {'psnr_mean': 17.35, 'ssim_mean': 0.560},
        ...     'block2': {'psnr_mean': 15.20, 'ssim_mean': 0.485},
        ...     'block3': {'psnr_mean': 12.80, 'ssim_mean': 0.390},
        ... }
        >>> 
        >>> plot_layer_comparison(results, 'vgg16', metric='psnr_mean')
        [SAVED] Layer comparison: results/vgg16/figures/vgg16_layer_comparison_psnr_mean.png
    
    Note:
        The function assumes all layers have the requested metric in their
        results dictionary. KeyError will be raised if metric is missing.
    """
    
    # Create save path in architecture-specific directory
    # Example: results/vgg16/figures/vgg16_layer_comparison_psnr_mean.png
    save_path = Path(f'results/{architecture}/figures/{architecture}_layer_comparison_{metric}.png')
    
    # Create directory if needed
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract layer names and metric values from results dictionary
    # layers: ['layer1', 'layer2', 'layer3', 'layer4']
    layers = list(results_dict.keys())
    
    # Extract metric value for each layer
    # values: [22.5, 24.1, 21.8, 18.5] (example for PSNR)
    values = [results_dict[layer][metric] for layer in layers]
    
    # Create figure for bar chart
    # figsize=(10, 6): Good size for readability
    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    # layers: x-axis categories
    # values: bar heights
    # color='steelblue': Professional blue color
    # alpha=0.8: Slight transparency for better appearance
    plt.bar(layers, values, color='steelblue', alpha=0.8)
    
    # Configure plot
    plt.xlabel('Layer', fontsize=12)
    
    # Convert metric name for y-axis label
    # 'psnr_mean' -> 'PSNR MEAN'
    # replace('_', ' '): Replace underscores with spaces
    # upper(): Convert to uppercase
    plt.ylabel(metric.upper().replace('_', ' '), fontsize=12)
    
    # Create title with architecture name
    # Example: "VGG16 - PSNR MEAN by Layer"
    plt.title(f'{architecture.upper()} - {metric.upper()} by Layer', 
              fontsize=14, fontweight='bold')
    
    # Add horizontal grid lines for easier reading
    # alpha=0.3: Light gray, not distracting
    # axis='y': Only horizontal lines (not vertical)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Optimize layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Close figure to free memory
    plt.close()
    
    # Print confirmation
    print(f"[SAVED] Layer comparison: {save_path}")


def plot_architecture_comparison(results_dict, layer_name, metric='psnr_mean'):
    """
    Create bar chart comparing metric across architectures for one layer.
    
    This visualization answers: "Which architecture is best for feature
    inversion?" It compares different network architectures (VGG, ResNet,
    ViT, etc.) at the same layer depth.
    
    Why Compare Architectures:
        
        Different Design Philosophies:
            - VGG: Simple, deep, uniform structure
            - ResNet: Skip connections, residual learning
            - ViT: Self-attention, global context
            - Each may preserve different information
        
        Feature Quality:
            - Some architectures learn better features
            - Better features -> easier reconstruction
            - Helps select best backbone for task
    
    Typical Results:
        For shallow layers:
        - VGG often best (high spatial resolution)
        - ResNet similar (residual connections help)
        - ViT worst (patch-based, lower resolution)
    
    Visualization Design:
        - Bar chart for easy comparison
        - Coral color (distinguishes from layer comparison)
        - Horizontal grid lines for value reading
        - Y-axis shows metric value
        - X-axis shows architecture names
    
    File Location:
        Saved to root results/ directory (not architecture-specific)
        since it compares multiple architectures.
    
    Args:
        results_dict (dict): Mapping from architecture names to results.
                            Example: {
                                'vgg16': {'psnr_mean': 17.35, 'ssim_mean': 0.560},
                                'resnet34': {'psnr_mean': 16.80, 'ssim_mean': 0.545},
                                'vit': {'psnr_mean': 14.20, 'ssim_mean': 0.475}
                            }
        
        layer_name (str): Layer being compared (e.g., 'layer2', 'block1')
                         Used in title and filename
        
        metric (str, optional): Metric to plot.
                               Default: 'psnr_mean'
    
    Returns:
        None: Saves PNG file to disk, prints confirmation
    
    Example:
        >>> results = {
        ...     'vgg16': {'psnr_mean': 17.35, 'ssim_mean': 0.560},
        ...     'resnet34': {'psnr_mean': 16.80, 'ssim_mean': 0.545}
        ... }
        >>> 
        >>> plot_architecture_comparison(results, 'layer1', 'psnr_mean')
        [SAVED] Architecture comparison: results/architecture_comparison_layer1_psnr_mean.png
    
    Note:
        Unlike plot_layer_comparison, this saves to root results/ directory
        since it spans multiple architectures.
    """
    
    # Create save path in root results directory
    # Not architecture-specific since comparing multiple architectures
    # Example: results/architecture_comparison_layer1_psnr_mean.png
    save_path = Path(f'results/architecture_comparison_{layer_name}_{metric}.png')
    
    # Create directory if needed
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract architecture names and metric values
    architectures = list(results_dict.keys())
    values = [results_dict[arch][metric] for arch in architectures]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    # Use coral color to distinguish from layer comparison (steel blue)
    plt.bar(architectures, values, color='coral', alpha=0.8)
    
    # Configure plot
    plt.xlabel('Architecture', fontsize=12)
    plt.ylabel(metric.upper().replace('_', ' '), fontsize=12)
    
    # Title includes layer name for context
    # Example: "PSNR MEAN Comparison - layer1"
    plt.title(f'{metric.upper()} Comparison - {layer_name}', 
              fontsize=14, fontweight='bold')
    
    # Add horizontal grid
    plt.grid(True, alpha=0.3, axis='y')
    
    # Optimize layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Close figure
    plt.close()
    
    # Print confirmation
    print(f"[SAVED] Architecture comparison: {save_path}")


def plot_metrics_heatmap(results_dict, save_name='metrics_heatmap'):
    """
    Create heatmap showing all metrics across all experiments.
    
    This visualization provides a comprehensive overview of experimental
    results in a single figure. Colors indicate performance:
    - Green: Good performance
    - Yellow: Medium performance  
    - Red: Poor performance
    
    The heatmap makes it easy to:
    - Identify best performing experiments at a glance
    - Spot patterns across experiments
    - Find experiments with balanced metrics
    - Create publication-ready comparison figures
    
    Heatmap Design:
        
        Rows: Different experiments
            - Each row is one experiment configuration
            - Example: "vgg16_block1_transposed_conv"
        
        Columns: Different metrics
            - PSNR, SSIM, LPIPS, etc.
            - Only mean values (not std) for clarity
        
        Colors: Performance indication
            - RdYlGn colormap (Red-Yellow-Green)
            - Green: High values (good for PSNR/SSIM)
            - Red: Low values (poor for PSNR/SSIM)
            - Note: LPIPS should be reversed (low is good)
        
        Annotations: Numeric values in cells
            - Shows exact metric values
            - Formatted to 3 decimal places
    
    Interpreting the Heatmap:
        
        Greenest Row:
            - Best overall experiment
            - High PSNR, high SSIM, low LPIPS
            - Consider this configuration
        
        Greenest Column:
            - Metric where all experiments do well
            - Or metric with larger scale
        
        Reddest Row:
            - Worst overall experiment
            - Avoid this configuration
        
        Mixed Colors:
            - Some metrics good, others poor
            - May indicate trade-offs
    
    Limitations:
        - Colormap assumes higher = better
        - LPIPS is opposite (lower = better)
        - May need manual adjustment for LPIPS
        - Metrics on different scales affect colors
    
    Args:
        results_dict (dict): Nested dictionary with all results.
                            Structure: {
                                'exp1': {'psnr_mean': 22.5, 'ssim_mean': 0.75, ...},
                                'exp2': {'psnr_mean': 24.1, 'ssim_mean': 0.82, ...},
                                ...
                            }
        
        save_name (str, optional): Base filename for saved heatmap.
                                  Default: 'metrics_heatmap'
                                  Extension .png added automatically
    
    Returns:
        None: Saves PNG file to disk, prints confirmation
    
    Example:
        >>> results = {
        ...     'vgg16_block1_transposed': {'psnr_mean': 17.35, 'ssim_mean': 0.560, 'lpips_mean': 0.25},
        ...     'vgg16_block3_transposed': {'psnr_mean': 12.80, 'ssim_mean': 0.390, 'lpips_mean': 0.42},
        ...     'resnet34_layer1_attention': {'psnr_mean': 16.20, 'ssim_mean': 0.520, 'lpips_mean': 0.30}
        ... }
        >>> 
        >>> plot_metrics_heatmap(results, save_name='final_results_heatmap')
        [SAVED] Metrics heatmap: results/final_results_heatmap.png
    
    Note on Pandas:
        This function imports pandas locally to avoid requiring it as a
        global dependency if not using heatmap functionality.
    """
    
    # Import pandas locally
    # This allows other functions to work without pandas installed
    import pandas as pd
    
    # Convert nested dictionary to DataFrame
    # .T transposes so experiments are rows and metrics are columns
    # Example:
    #                           psnr_mean  ssim_mean  lpips_mean
    # vgg16_block1_transposed      17.35      0.560        0.25
    # vgg16_block3_transposed      12.80      0.390        0.42
    df = pd.DataFrame(results_dict).T
    
    # Select only mean metrics for heatmap
    # We don't include _std columns to keep heatmap focused
    # Adjust this list based on available metrics
    metrics = ['psnr_mean', 'ssim_mean', 'lpips_mean']
    
    # Extract only these columns
    # This fails if any metric is missing - caller should ensure all present
    df_metrics = df[metrics]
    
    # Create figure for heatmap
    # figsize=(10, 8): Good size for multiple experiments
    plt.figure(figsize=(10, 8))
    
    # Create heatmap using seaborn
    # annot=True: Show numeric values in cells
    # fmt='.3f': Format numbers to 3 decimal places
    # cmap='RdYlGn': Red-Yellow-Green colormap
    #                Green = high values (good for PSNR/SSIM)
    #                Red = low values (poor for PSNR/SSIM)
    #                Note: LPIPS is opposite (low is good)
    # cbar_kws: Customize colorbar label
    sns.heatmap(df_metrics, annot=True, fmt='.3f', cmap='RdYlGn', 
                cbar_kws={'label': 'Score'})
    
    # Configure plot
    plt.title('Metrics Heatmap Across Experiments', fontsize=14, fontweight='bold')
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Experiment', fontsize=12)
    
    # Optimize layout
    plt.tight_layout()
    
    # Create save path in root results directory
    # Not experiment-specific since comparing all experiments
    save_path = Path(f'results/{save_name}.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Close figure
    plt.close()
    
    # Print confirmation
    print(f"[SAVED] Metrics heatmap: {save_path}")


def count_parameters(model):
    """
    Count number of trainable parameters in a PyTorch model.
    
    Parameters are the learned weights and biases that define a neural
    network. This function counts only trainable parameters (those with
    requires_grad=True), excluding frozen parameters.
    
    Why Count Parameters:
        
        Model Capacity:
            - More parameters = more learning capacity
            - Can fit more complex patterns
            - But also more prone to overfitting
        
        Memory Requirements:
            - Each float32 parameter uses 4 bytes
            - 1 million params = 4 MB
            - Important for GPU memory planning
        
        Training Time:
            - More parameters = longer training
            - Gradient computation scales with parameter count
        
        Comparison:
            - Compare decoder complexity
            - Understand model efficiency
            - Parameters per performance unit
    
    Parameter Types:
        
        Weights:
            - Connection strengths between layers
            - Most parameters are weights
            - Example: Conv2d(64, 128, 3x3) has 64*128*3*3 = 73,728 weights
        
        Biases:
            - Offset terms added to weighted sums
            - One per output channel
            - Example: Conv2d(64, 128) has 128 biases
        
        Total = Weights + Biases
    
    Trainable vs Frozen:
        
        Trainable (requires_grad=True):
            - Parameters that will be updated during training
            - Decoder parameters in this project
            - Counted by this function
        
        Frozen (requires_grad=False):
            - Parameters that stay fixed during training
            - Encoder parameters in this project
            - Not counted by this function
    
    Example Parameter Counts:
        
        Simple TransposedConv Decoder:
            - ~200K parameters
            - Smallest, fastest
            - Best performance in this project
        
        Wavelet Decoder:
            - ~8M parameters
            - Medium complexity
        
        Attention Decoder:
            - ~12M parameters
            - High complexity
        
        FrequencyAware Decoder:
            - ~35M parameters
            - Most complex
            - Slower, no better performance
    
    Args:
        model (nn.Module): PyTorch model to count parameters.
                          Can be full model or submodule.
        
    Returns:
        int: Total number of trainable parameters
    
    Example:
        >>> from models import TransposedConvDecoder
        >>> decoder = TransposedConvDecoder(in_channels=64, spatial_size=112)
        >>> num_params = count_parameters(decoder)
        >>> print(f"Decoder has {num_params:,} trainable parameters")
        Decoder has 204,931 trainable parameters
        >>> 
        >>> # Check memory requirement (float32)
        >>> memory_mb = num_params * 4 / (1024 * 1024)
        >>> print(f"Memory: {memory_mb:.2f} MB")
        Memory: 0.78 MB
    
    Note on Efficiency:
        This function is efficient even for large models because it only
        sums element counts, not the actual parameter tensors. It works
        on models with billions of parameters.
    """
    
    # Sum up the number of elements in each trainable parameter
    # 
    # p.numel(): Returns number of elements in parameter tensor
    #           Example: Weight tensor (64, 128, 3, 3) has 73,728 elements
    # 
    # p.requires_grad: Boolean flag indicating if parameter is trainable
    #                 True: Parameter will be updated during training
    #                 False: Parameter is frozen
    # 
    # Generator expression: (value for p in ... if condition)
    #                      Efficient iteration without creating list
    # 
    # sum(): Add up all parameter counts
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_results_table(results_dict, save_path='results/metrics/comparison.csv'):
    """
    Create formatted CSV table of all experimental results.
    
    This function converts a nested results dictionary into a structured
    CSV table that can be:
    - Imported into papers and reports
    - Analyzed with pandas or Excel
    - Shared with collaborators
    - Used for statistical analysis
    
    The CSV format is universal and can be opened by:
    - Microsoft Excel
    - Google Sheets
    - Python pandas
    - R
    - Any text editor
    
    Table Structure:
        
        Rows: Experiments
            - Each row is one experimental configuration
            - Row index is experiment name
        
        Columns: Metrics
            - Each column is one metric
            - Includes mean, std, min, max for each metric
        
        Cells: Numeric values
            - All values preserved with full precision
            - No rounding or formatting
    
    Why CSV Format:
        
        Portability:
            - Plain text format
            - Works on any platform
            - Future-proof
        
        Simplicity:
            - Easy to read and edit
            - No special software needed
            - Can be version controlled
        
        Compatibility:
            - Import into any analysis tool
            - Easy to programmatically process
            - Standard in research
    
    Typical Use Cases:
        
        Statistical Analysis:
            - Load in pandas for analysis
            - Calculate significance tests
            - Generate custom plots
        
        Reporting:
            - Import into LaTeX tables
            - Copy to Word documents
            - Create slides
        
        Collaboration:
            - Share results easily
            - Others can analyze independently
            - Transparent and verifiable
    
    Args:
        results_dict (dict): Nested dictionary with all experimental results.
                            Structure: {
                                'exp1_name': {
                                    'psnr_mean': 22.5,
                                    'psnr_std': 1.2,
                                    'ssim_mean': 0.82,
                                    ...
                                },
                                'exp2_name': {...},
                                ...
                            }
        
        save_path (str or Path, optional): Where to save CSV file.
                                          Default: 'results/metrics/comparison.csv'
                                          Parent directories created if needed
    
    Returns:
        pd.DataFrame: The created DataFrame (also saved to CSV).
                     Useful for immediate analysis after creation.
    
    Example:
        >>> results = {
        ...     'vgg16_block1_transposed': {
        ...         'psnr_mean': 17.35, 'psnr_std': 2.1,
        ...         'ssim_mean': 0.560, 'ssim_std': 0.08
        ...     },
        ...     'resnet34_layer1_attention': {
        ...         'psnr_mean': 16.20, 'psnr_std': 1.9,
        ...         'ssim_mean': 0.520, 'ssim_std': 0.07
        ...     }
        ... }
        >>> 
        >>> df = create_results_table(results, 'results/final_metrics.csv')
        [SAVED] Results table: results/final_metrics.csv
        >>> 
        >>> # Can immediately use returned DataFrame
        >>> print(df['psnr_mean'].max())  # Find best PSNR
        17.35
    
    CSV File Example:
        ,psnr_mean,psnr_std,ssim_mean,ssim_std
        vgg16_block1_transposed,17.35,2.1,0.560,0.08
        resnet34_layer1_attention,16.20,1.9,0.520,0.07
    
    Note on Pandas:
        This function imports pandas locally to avoid requiring it globally
        if this functionality is not used.
    """
    
    # Import pandas locally
    import pandas as pd
    
    # Convert nested dictionary to DataFrame
    # .T transposes so experiments are rows (index) and metrics are columns
    # 
    # Before transpose:
    #                     exp1    exp2    exp3
    # psnr_mean          22.5    24.1    21.8
    # ssim_mean          0.75    0.82    0.70
    # 
    # After transpose:
    #       psnr_mean  ssim_mean
    # exp1      22.5       0.75
    # exp2      24.1       0.82
    # exp3      21.8       0.70
    df = pd.DataFrame(results_dict).T
    
    # Convert string path to Path object for cross-platform compatibility
    save_path = Path(save_path)
    
    # Create parent directories if they don't exist
    # Example: If save_path is 'results/metrics/comparison.csv',
    #          this creates 'results/' and 'results/metrics/'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save DataFrame to CSV file
    # This creates a plain text file with comma-separated values
    # First column is index (experiment names)
    # Remaining columns are metrics
    df.to_csv(save_path)
    
    # Print confirmation message
    print(f"[SAVED] Results table: {save_path}")
    
    # Return DataFrame for immediate use
    return df


def save_model_info(encoder, decoder, config, save_path='results/model_info.txt'):
    """
    Save detailed text file documenting model configuration.
    
    This function creates a human-readable documentation file containing:
    - Model architectures used (encoder and decoder)
    - Parameter counts for each component
    - All configuration settings
    - Training hyperparameters
    
    This documentation is crucial for:
    - Reproducibility: Others can recreate exact setup
    - Record keeping: Remember what was run
    - Comparison: Easy to compare configurations
    - Debugging: Verify correct setup
    
    Documentation Philosophy:
        
        Human Readable:
            - Plain text format
            - Clear formatting with headers
            - No special software needed
        
        Comprehensive:
            - All information needed to reproduce
            - Architecture details
            - Training settings
            - Parameter counts
        
        Persistent:
            - Saved with results
            - Future-proof format
            - Easy to archive
    
    File Format:
        
        Header:
            =================================================
            MODEL CONFIGURATION
            =================================================
        
        Encoder Section:
            Architecture: vgg16
            Layer: block1
            Parameters: 14,714,688 (frozen)
        
        Decoder Section:
            Type: transposed_conv
            Parameters: 204,931 (trainable)
        
        Config Section:
            lr: 0.0001
            epochs: 30
            batch_size: 4
            ...
    
    Why Text Format:
        
        Advantages:
            - Universal compatibility
            - Easy to read and edit
            - Can be version controlled
            - Searchable
            - No dependencies
        
        Alternative formats considered:
            - JSON: Machine readable but less human friendly
            - YAML: Good but requires parser
            - XML: Too verbose
            - Plain text: Best balance
    
    Args:
        encoder (nn.Module): Feature extractor model.
                            Should have .architecture and .layer_name attributes
        
        decoder (nn.Module): Decoder model.
                            Parameter count will be computed
        
        config (dict): Configuration dictionary containing all settings.
                      Should include:
                      - decoder_type: Type of decoder
                      - lr: Learning rate
                      - epochs: Number of epochs
                      - Any other hyperparameters
        
        save_path (str or Path, optional): Where to save text file.
                                          Default: 'results/model_info.txt'
    
    Returns:
        None: Saves text file to disk, prints confirmation
    
    Example:
        >>> from models import VGGExtractor, TransposedConvDecoder
        >>> 
        >>> encoder = VGGExtractor('vgg16', 'block1')
        >>> decoder = TransposedConvDecoder(in_channels=64, spatial_size=112)
        >>> 
        >>> config = {
        ...     'decoder_type': 'transposed_conv',
        ...     'lr': 1e-4,
        ...     'epochs': 30,
        ...     'batch_size': 4,
        ...     'mse_weight': 0.5,
        ...     'lpips_weight': 0.5
        ... }
        >>> 
        >>> save_model_info(encoder, decoder, config, 'results/model_info.txt')
        [SAVED] Model info: results/model_info.txt
    
    Example Output File:
        =================================================
        MODEL CONFIGURATION
        =================================================
        
        ENCODER:
          Architecture: vgg16
          Layer: block1
          Parameters: 14,714,688 (frozen)
        
        DECODER:
          Type: transposed_conv
          Parameters: 204,931 (trainable)
        
        TRAINING CONFIG:
          decoder_type: transposed_conv
          lr: 0.0001
          epochs: 30
          batch_size: 4
          mse_weight: 0.5
          lpips_weight: 0.5
        
        =================================================
    """
    
    # Convert to Path object for cross-platform compatibility
    save_path = Path(save_path)
    
    # Create parent directories if needed
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open file for writing
    # 'w' mode: Write text, overwrite if exists
    with open(save_path, 'w') as f:
        # Write header
        # = characters create visual separator
        f.write("="*60 + "\n")
        f.write("MODEL CONFIGURATION\n")
        f.write("="*60 + "\n\n")
        
        # Write encoder information
        # Assumes encoder has .architecture and .layer_name attributes
        f.write(f"ENCODER:\n")
        f.write(f"  Architecture: {encoder.architecture}\n")
        f.write(f"  Layer: {encoder.layer_name}\n")
        
        # Count and format encoder parameters
        # {:,} adds thousand separators (1,234,567)
        f.write(f"  Parameters: {count_parameters(encoder):,} (frozen)\n\n")
        
        # Write decoder information
        f.write(f"DECODER:\n")
        
        # Get decoder type from config with default fallback
        f.write(f"  Type: {config.get('decoder_type', 'unknown')}\n")
        
        # Count and format decoder parameters
        f.write(f"  Parameters: {count_parameters(decoder):,} (trainable)\n\n")
        
        # Write training configuration
        # This section includes all hyperparameters
        f.write(f"TRAINING CONFIG:\n")
        
        # Iterate through all config items
        # Indent each line for readability
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        
        # Write footer
        f.write("\n" + "=" * 60 + "\n")
    
    # Print confirmation message
    print(f"[SAVED] Model info: {save_path}")


def set_seed(seed=42):
    """
    Set random seeds for reproducibility across all libraries.
    
    Deep learning involves randomness in many places:
    - Weight initialization
    - Data shuffling
    - Dropout
    - Data augmentation
    - Some GPU operations
    
    Setting seeds ensures experiments are reproducible - running the same
    code multiple times gives identical results. This is critical for
    scientific research and debugging.
    
    Why Reproducibility Matters:
        
        Scientific Research:
            - Results must be verifiable
            - Others can reproduce your findings
            - Required for publication
        
        Debugging:
            - Can reproduce bugs consistently
            - Can verify fixes work
            - Easier to isolate issues
        
        Development:
            - Compare code changes fairly
            - Verify improvements are real
            - Build confidence in code
    
    What This Function Sets:
        
        1. Python Random:
            - Built-in random module
            - Used by some libraries
            - Affects random.choice, random.shuffle, etc.
        
        2. NumPy Random:
            - NumPy random number generator
            - Used by data augmentation
            - Affects np.random.rand, np.random.choice, etc.
        
        3. PyTorch CPU Random:
            - PyTorch random number generator (CPU)
            - Used for weight initialization
            - Affects torch.rand, torch.randn, etc.
        
        4. PyTorch GPU Random:
            - PyTorch random number generator (CUDA)
            - Used for GPU operations
            - Affects torch.cuda.rand, dropout, etc.
        
        5. CuDNN Determinism:
            - Makes CuDNN operations deterministic
            - Some operations have non-deterministic backends
            - Enables reproducibility at cost of slight performance
    
    Performance Impact:
        
        CuDNN Determinism:
            - deterministic=True: Ensures reproducible results
            - benchmark=False: Disables algorithm selection
            
            Performance impact:
            - Typically 5-10% slower
            - Worth it for reproducibility
            - Can disable for final production runs
    
    Why Seed = 42:
        - Reference to "The Hitchhiker's Guide to the Galaxy"
        - "42" is the "Answer to Life, Universe, and Everything"
        - Has become common convention in machine learning
        - Any fixed value works, but 42 is recognizable
    
    Limitations:
        
        Perfect Reproducibility:
            - CPU operations: Fully reproducible
            - GPU operations: Mostly reproducible
            - Some GPU ops have inherent non-determinism
            - Multi-GPU: More challenging
            - Different hardware: May vary slightly
        
        Practical Reproducibility:
            - Same code, same hardware: Identical results
            - Different GPU models: ~0.1% variation
            - This is acceptable for research
    
    Args:
        seed (int, optional): Random seed value to use.
                             Any integer works.
                             Convention: 42 (default)
    
    Returns:
        None: Sets global random states, prints confirmation
    
    Example:
        >>> # At start of training script
        >>> set_seed(42)
        Random seed set to 42
        >>> 
        >>> # All subsequent random operations are deterministic
        >>> weights = torch.randn(100, 100)  # Same every run
        >>> shuffled_data = torch.randperm(1000)  # Same every run
        >>> 
        >>> # Train model
        >>> model = train()  # Results will be identical each run
    
    Usage Pattern:
        Call this function once at the very beginning of your script,
        before any model creation or data loading:
        
        if __name__ == '__main__':
            set_seed(42)  # First thing
            
            # Now everything is reproducible
            model = create_model()
            data = load_data()
            train(model, data)
    
    Note on Timing:
        Must call this BEFORE:
        - Creating models (weight initialization)
        - Loading data (shuffling)
        - Any random operations
        
        Calling after random operations have occurred will not
        make those operations reproducible.
    """
    
    # Import random module
    import random
    
    # Set seed for Python's built-in random module
    # Affects: random.random(), random.choice(), random.shuffle()
    random.seed(seed)
    
    # Set seed for NumPy's random number generator
    # Affects: np.random.rand(), np.random.randint(), etc.
    # Important for data augmentation and preprocessing
    np.random.seed(seed)
    
    # Set seed for PyTorch's random number generator (CPU)
    # Affects: torch.rand(), torch.randn(), weight initialization
    # Important for model initialization
    torch.manual_seed(seed)
    
    # Set seed for PyTorch's random number generator (GPU)
    # Only needed if CUDA is available
    if torch.cuda.is_available():
        # For single GPU
        # Affects: torch.cuda.rand(), GPU operations
        torch.cuda.manual_seed(seed)
        
        # For multi-GPU systems
        # Sets seed for all GPUs, not just GPU 0
        torch.cuda.manual_seed_all(seed)
    
    # Make PyTorch operations deterministic
    # This forces PyTorch to use deterministic algorithms
    # Some operations have non-deterministic backends by default for speed
    # 
    # deterministic=True: Use only deterministic operations
    # Ensures reproducibility but may be slightly slower
    torch.backends.cudnn.deterministic = True
    
    # Disable CuDNN benchmarking
    # benchmark=True: Auto-select fastest algorithm (non-deterministic)
    # benchmark=False: Use fixed algorithm (deterministic)
    # 
    # Benchmarking tests different algorithms and selects fastest
    # This introduces non-determinism as selection may vary
    torch.backends.cudnn.benchmark = False
    
    # Print confirmation message
    print(f"Random seed set to {seed}")
