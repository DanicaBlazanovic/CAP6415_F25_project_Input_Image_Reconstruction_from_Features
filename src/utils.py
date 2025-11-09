"""
Utility functions for visualization, device detection, and helper operations.

Cross-platform compatible utilities for the project.
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
    
    Device priority (fastest to slowest):
    1. CUDA (NVIDIA GPU) - fastest, best for deep learning
    2. MPS (Apple Silicon GPU) - fast on M1/M2/M3 Macs
    3. CPU - slowest but always available
    
    Returns:
        device: torch.device object
        
    Example:
        >>> device = get_device()
        >>> model.to(device)  # Move model to best available device
    """
    # Check if CUDA (NVIDIA GPU) is available
    # CUDA provides best performance for deep learning
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Print GPU name for user information
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    
    # Check if MPS (Metal Performance Shaders) is available
    # MPS is Apple's GPU acceleration framework for M1/M2/M3 chips
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS (Metal Performance Shaders)")
    
    # Fall back to CPU if no GPU available
    # CPU is slower but works on all systems
    else:
        device = torch.device('cpu')
        print("Using CPU (this will be slower)")
    
    return device


def save_comparison_grid(original, reconstructed, config, num_images = 8):
    """
    Save side-by-side comparison of original and reconstructed images.
    
    Creates a grid visualization with two rows:
    - Top row: Original images
    - Bottom row: Reconstructed images
    
    This visual comparison helps assess reconstruction quality at a glance.
    Automatically names file based on experiment configuration.
    
    Args:
        original: Original images [B, C, H, W] (normalized with ImageNet stats)
        reconstructed: Reconstructed images [B, C, H, W] (normalized)
        config: Configuration dict with keys:
            - architecture: e.g., 'resnet34'
            - layer_name: e.g., 'layer3'
            - decoder_type: e.g., 'attention'
        num_images: Number of image pairs to display (default 8)
    
    Note:
        Images are automatically denormalized from ImageNet normalization
        to [0, 1] range for proper visualization.
    """
    # Create unique experiment name from configuration
    # This ensures each experiment's visualizations are easily identifiable
    # Example: "resnet34_layer3_attention"
    arch_name = config['architecture']
    layer_name = config['layer_name']
    decoder_type = config['decoder_type']
    experiment_name = f"{arch_name}_{layer_name}_{decoder_type}"
    
    # Create save path in architecture-specific folder
    # Structure: results/resnet34/figures/resnet34_layer3_attention_reconstruction.png
    # This organizes visualizations by architecture for easy comparison

    save_path = Path(f'results/{arch_name}/figures/{experiment_name}_reconstruction.png')
    # Create directory if it doesn't exist (cross-platform with Path)
    save_path.parent.mkdir(parents=True, exist_ok = True)
    
    # Denormalize images from ImageNet normalization to [0, 1] range
    # Our images were normalized with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # Denormalization: img = (normalized * std) + mean
    # This is necessary because matplotlib expects images in [0, 1] range
    original = denormalize(original.cpu())  # Move to CPU for numpy conversion
    reconstructed = denormalize(reconstructed.cpu())
    
    # Determine how many images to show
    # Use minimum of requested number and available images in batch
    num_images = min(num_images, original.shape[0])
    
    # Create figure with matplotlib
    # - 2 rows: original (top), reconstructed (bottom)
    # - num_images columns: one column per image
    # - Figure size: 3 inches per image width, 6 inches height
    fig, axes = plt.subplots(2, num_images, figsize=(3 * num_images, 6))
    
    # Iterate over each image pair
    for i in range(num_images):
        # TOP ROW: Original images
        # Convert from PyTorch format [C, H, W] to matplotlib format [H, W, C]
        img_orig = original[i].permute(1, 2, 0).numpy()
        
        # Display image in subplot
        axes[0, i].imshow(img_orig)
        axes[0, i].axis('off')  # Hide axis ticks and labels
        
        # Add title to first image only (to save space)
        if i == 0:
            axes[0, i].set_title('Original', fontsize = 12, fontweight = 'bold')
        # BOTTOM ROW: Reconstructed images
        # Same process as original images
        img_recon = reconstructed[i].permute(1, 2, 0).numpy()
        axes[1, i].imshow(img_recon)
        axes[1, i].axis('off')
        
        # Add title to first reconstructed image only
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize = 12, fontweight = 'bold')
        if i == 0:
            # Add experiment name to reconstruction title
            experiment_name = f"{config['architecture']}_{config['layer_name']}"
            axes[1, i].set_title(f'Reconstructed\n({experiment_name})', fontsize=10, fontweight='bold')
            
    # Adjust spacing between subplots for clean appearance
    plt.tight_layout()
    
    # Save figure to disk
    # - dpi=150: High quality (default is 100)
    # - bbox_inches='tight': Remove extra whitespace
    plt.savefig(save_path, dpi = 150, bbox_inches = 'tight')
    
    # Close figure to free memory
    # Important when generating many figures in batch
    plt.close()
    
    print(f"[SAVED] Reconstruction comparison: {save_path}")


def plot_training_history(history, config):
    """
    Plot training and validation loss curves over epochs.
    
    Creates two subplots:
    1. Loss curves: Shows train and validation loss progression
       - Helps identify overfitting (val loss increases while train loss decreases)
       - Both should generally decrease over time
    
    2. Learning rate schedule: Shows how LR changes over time
       - Useful for understanding optimizer behavior
       - Should decrease when validation loss plateaus (ReduceLROnPlateau)
    
    Automatically names file based on experiment configuration.
    
    Args:
        history: Dict containing training history with keys:
            - 'train_loss': List of training losses per epoch
            - 'val_loss': List of validation losses per epoch
            - 'lr': List of learning rates per epoch
        config: Configuration dict with architecture/layer/decoder info
    """
    # Create experiment name from configuration
    arch_name = config['architecture']
    layer_name = config['layer_name']
    decoder_type = config['decoder_type']
    experiment_name = f"{arch_name}_{layer_name}_{decoder_type}"
    
    # Create save path in architecture-specific folder
    # Example: results/resnet34/figures/resnet34_layer3_attention_training.png
    save_path = Path(f'results/{arch_name}/figures/{experiment_name}_training.png')
    save_path.parent.mkdir(parents = True, exist_ok = True)
    
    # Create figure with two side-by-side subplots
    # Figure size: 14 inches wide, 5 inches tall
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 5))
    
    # Create list of epoch numbers for x-axis
    # Starts at 1 (not 0) because that's how we count epochs
    epochs = range(1, len(history['train_loss']) + 1)
    
    # SUBPLOT 1: Loss curves
    # Plot training loss in blue
    ax1.plot(epochs, history['train_loss'], 'b-', label = 'Train Loss', linewidth=2)
    
    # Plot validation loss in red
    ax1.plot(epochs, history['val_loss'], 'r-', label = 'Val Loss', linewidth=2)
    
    # Configure subplot
    ax1.set_xlabel('Epoch', fontsize = 12)
    ax1.set_ylabel('Loss', fontsize = 12)
    ax1.set_title('Training and Validation Loss', fontsize = 14, fontweight = 'bold')
    ax1.legend(fontsize = 11)
    ax1.grid(True, alpha = 0.3)  # Add light grid for readability
    
    # SUBPLOT 2: Learning rate schedule
    if 'lr' in history:
        # Plot learning rate in green
        ax2.plot(epochs, history['lr'], 'g-', linewidth = 2)
        
        # Configure subplot
        ax2.set_xlabel('Epoch', fontsize = 12)
        ax2.set_ylabel('Learning Rate', fontsize = 12)
        ax2.set_title('Learning Rate Schedule', fontsize = 14, fontweight='bold')
        
        # Use log scale for y-axis
        # Learning rates are typically small (1e-3, 1e-4) so log scale shows changes better
        ax2.set_yscale('log')
        ax2.grid(True, alpha = 0.3)
    
    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi = 150, bbox_inches = 'tight')
    plt.close()
    
    print(f"[SAVED] Training history: {save_path}")


def plot_layer_comparison(results_dict, architecture, metric = 'psnr_mean'):
    """
    Compare a specific metric across different layers for one architecture.
    
    Creates a bar chart showing how reconstruction quality varies with layer depth.
    This helps answer: "Which layer preserves the most information?"
    
    Expected pattern:
    - Shallow layers (layer1): More spatial detail, less semantic info
    - Deep layers (layer4): Less spatial detail, more semantic info
    - Best layer is usually in the middle (layer2 or layer3)
    
    Args:
        results_dict: Dict mapping layer names to results dicts
            Example: {
                'layer1': {'psnr_mean': 22.5, 'ssim_mean': 0.75, ...},
                'layer2': {'psnr_mean': 24.1, 'ssim_mean': 0.82, ...},
                ...
            }
        architecture: Architecture name (e.g., 'resnet34')
        metric: Which metric to plot (default 'psnr_mean')
            Options: 'psnr_mean', 'ssim_mean', 'lpips_mean', 'mse_mean'
    """
    # Create save path
    save_path = Path(f'../results/{architecture}/figures/{architecture}_layer_comparison_{metric}.png')
   
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract layer names and metric values
    # layers: ['layer1', 'layer2', 'layer3', 'layer4']
    layers = list(results_dict.keys())
    
    # values: corresponding metric values for each layer
    values = [results_dict[layer][metric] for layer in layers]
    
    # Create bar chart
    plt.figure(figsize = (10, 6))
    
    # Create bars
    # color='steelblue': Professional blue color
    # alpha=0.8: Slight transparency for better appearance
    plt.bar(layers, values, color = 'steelblue', alpha = 0.8)
    
    # Configure plot
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel(metric.upper().replace('_', ' '), fontsize=12)  # Convert 'psnr_mean' to 'PSNR MEAN'
    plt.title(f'{architecture.upper()} - {metric.upper()} by Layer', fontsize = 14, fontweight = 'bold')
    
    # Add horizontal grid lines for easier reading
    plt.grid(True, alpha = 0.3, axis = 'y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches = 'tight')
    plt.close()
    
    print(f"[SAVED] Layer comparison: {save_path}")


def plot_architecture_comparison(results_dict, layer_name, metric = 'psnr_mean'):
    """
    Compare a specific metric across different architectures for one layer.
    
    Creates a bar chart showing how different network architectures compare.
    This helps answer: "Which architecture is best for feature inversion?"
    
    Typical comparison: ResNet34 vs VGG16 vs ViT
    
    Args:
        results_dict: Dict mapping architecture names to results dicts
            Example: {
                'resnet34': {'psnr_mean': 24.1, ...},
                'vgg16': {'psnr_mean': 22.8, ...},
                'vit': {'psnr_mean': 23.5, ...}
            }
        layer_name: Layer name being compared (e.g., 'layer2', 'block3')
        metric: Which metric to plot (default 'psnr_mean')
    """
    # Create save path (not architecture-specific, goes in root results/)
    save_path = Path(f'results/architecture_comparison_{layer_name}_{metric}.png')
    save_path.parent.mkdir(parents = True, exist_ok = True)
    
    # Extract architecture names and metric values
    architectures = list(results_dict.keys())
    values = [results_dict[arch][metric] for arch in architectures]
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    
    # Use different color (coral) to distinguish from layer comparison
    plt.bar(architectures, values, color='coral', alpha = 0.8)
    
    # Configure plot
    plt.xlabel('Architecture', fontsize=12)
    plt.ylabel(metric.upper().replace('_', ' '), fontsize = 12)
    plt.title(f'{metric.upper()} Comparison - {layer_name}', fontsize = 14, fontweight = 'bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi = 150, bbox_inches = 'tight')
    plt.close()
    
    print(f"[SAVED] Architecture comparison: {save_path}")


def plot_metrics_heatmap(results_dict, save_name = 'metrics_heatmap'):
    """
    Create heatmap showing all metrics across all experiments.
    
    This provides a comprehensive overview of all experimental results.
    Heatmap uses color to show metric values:
    - Green: Good values (high PSNR/SSIM, low LPIPS)
    - Yellow: Medium values
    - Red: Poor values
    
    Useful for:
    - Identifying best performing experiments at a glance
    - Spotting patterns across experiments
    - Creating publication-ready comparison figures
    
    Args:
        results_dict: Nested dict with all experiment results
            Example: {
                'resnet34_layer1_attention': {
                    'psnr_mean': 22.5,
                    'ssim_mean': 0.75,
                    'lpips_mean': 0.35,
                    ...
                },
                'resnet34_layer2_attention': {...},
                ...
            }
        save_name: Base name for saved file (default 'metrics_heatmap')
    """
    import pandas as pd
    
    # Convert nested dict to DataFrame
    # Rows: experiments, Columns: metrics
    df = pd.DataFrame(results_dict).T
    
    # Select key metrics for heatmap
    # We don't include _std columns to keep heatmap focused on mean values
    metrics = ['psnr_mean', 'ssim_mean', 'lpips_mean']
    df_metrics = df[metrics]
    
    # Create heatmap using seaborn
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    # - annot=True: Show numeric values in cells
    # - fmt='.3f': Format numbers to 3 decimal places
    # - cmap='RdYlGn': Red-Yellow-Green colormap (green = good)
    # - cbar_kws: Customize colorbar label
    sns.heatmap(df_metrics, annot = True, fmt = '.3f', cmap = 'RdYlGn', cbar_kws = {'label': 'Score'})
    
    # Configure plot
    plt.title('Metrics Heatmap Across Experiments', fontsize = 14, fontweight = 'bold')
    plt.xlabel('Metric', fontsize = 12)
    plt.ylabel('Experiment', fontsize = 12)
    
    plt.tight_layout()
    
    # Save to results root (not architecture-specific)
    save_path = Path(f'results/{save_name}.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches = 'tight')
    plt.close()
    
    print(f"[SAVED] Metrics heatmap: {save_path}")


def count_parameters(model):
    """
    Count number of trainable parameters in a PyTorch model.
    
    Parameters are the weights and biases that the model learns during training.
    More parameters generally means:
    - More capacity to learn complex patterns
    - More memory required
    - Longer training time
    
    Args:
        model: PyTorch model (nn.Module)
        
    Returns:
        num_params: Integer count of trainable parameters
        
    Example:
        >>> decoder = SimpleDecoder(256, 7, 224)
        >>> print(f"Decoder has {count_parameters(decoder):,} parameters")
        Decoder has 1,234,567 parameters
        
    Note:
        We only count parameters where requires_grad=True.
        Frozen parameters (like in our encoder) are not counted.
    """
    # Sum up the number of elements (numel) in each trainable parameter tensor
    # p.numel() returns number of elements in parameter tensor
    # p.requires_grad is True for trainable parameters, False for frozen
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_results_table(results_dict, save_path = 'results/metrics/comparison.csv'):
    """
    Create formatted CSV table of all experimental results.
    
    This creates a comprehensive table with all metrics for all experiments.
    Useful for:
    - Importing into papers/reports
    - Statistical analysis
    - Sharing results with collaborators
    
    Args:
        results_dict: Dict mapping experiment names to metric dicts
            Example: {
                'resnet34_layer3_attention': {
                    'psnr_mean': 24.5,
                    'psnr_std': 1.2,
                    'ssim_mean': 0.82,
                    ...
                },
                ...
            }
        save_path: Where to save CSV file
        
    Returns:
        df: Pandas DataFrame with results (also saved to CSV)
    """
    import pandas as pd
    
    # Convert dict to DataFrame
    # .T transposes so experiments are rows and metrics are columns
    df = pd.DataFrame(results_dict).T
    
    # Ensure save directory exists
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    # This creates a simple text file that can be opened in Excel, Python, etc.
    df.to_csv(save_path)
    
    print(f"[SAVED] Results table: {save_path}")
    
    return df


def save_model_info(encoder, decoder, config, save_path = 'results/model_info.txt'):
    """
    Save detailed text file documenting model configuration.
    
    Creates human-readable documentation of:
    - Model architectures used
    - Number of parameters
    - All configuration settings
    
    This is crucial for reproducibility - someone should be able to
    recreate your exact setup from this file.
    
    Args:
        encoder: Feature extractor model
        decoder: Decoder model
        config: Training configuration dict
        save_path: Where to save text file
    """
    # Ensure save directory exists
    save_path = Path(save_path)
    save_path.parent.mkdir(parents = True, exist_ok = True)
    
    # Write to text file
    with open(save_path, 'w') as f:
        # Header
        f.write("="*60 + "\n")
        f.write("MODEL CONFIGURATION\n")
        f.write("="*60 + "\n\n")
        
        # Encoder information
        f.write(f"ENCODER:\n")
        f.write(f"  Architecture: {encoder.architecture}\n")
        f.write(f"  Layer: {encoder.layer_name}\n")
        f.write(f"  Parameters: {count_parameters(encoder):,} (frozen)\n\n")
        
        # Decoder information
        f.write(f"DECODER:\n")
        f.write(f"  Type: {config.get('decoder_type', 'unknown')}\n")
        f.write(f"  Parameters: {count_parameters(decoder):,} (trainable)\n\n")
        
        # Training configuration
        # This saves all hyperparameters for reproducibility
        f.write(f"TRAINING CONFIG:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        
        # Footer
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"[SAVED] Model info: {save_path}")


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Deep learning uses randomness in many places:
    - Weight initialization
    - Data shuffling
    - Dropout
    - Data augmentation
    
    Setting seeds ensures experiments are reproducible - running the same
    code twice will give identical results.
    
    Why 42? It's a reference to "The Hitchhiker's Guide to the Galaxy"
    and has become a common default in ML research.
    
    Args:
        seed: Random seed value (default 42)
        
    Example:
        >>> set_seed(42)
        >>> # Now all random operations are deterministic
        >>> # Running code again will give same results
    """
    import random
    
    # Set seed for Python's built-in random module
    random.seed(seed)
    
    # Set seed for NumPy's random number generator
    np.random.seed(seed)
    
    # Set seed for PyTorch's random number generator (CPU)
    torch.manual_seed(seed)
    
    # Set seed for PyTorch's random number generator (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # For single GPU
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Make PyTorch operations deterministic
    # This can slightly reduce performance but ensures reproducibility
    # Some operations have non-deterministic behavior by default for speed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")