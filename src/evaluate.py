"""
Evaluation metrics for image reconstruction.

Implements standard metrics for measuring reconstruction quality:
    - PSNR: Peak Signal-to-Noise Ratio
    - SSIM: Structural Similarity Index
    - LPIPS: Learned Perceptual Image Patch Similarity
    - MSE: Mean Squared Error
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim
import lpips
from tqdm import tqdm
from dataset import denormalize


class MetricsCalculator:
    """
    Calculate reconstruction quality metrics.
    
    This class provides methods to evaluate how well our decoder reconstructs images.
    We use multiple metrics because each captures different aspects of quality:
    
    1. PSNR (Peak Signal-to-Noise Ratio):
        Formula: PSNR = 10 * log10(MAX^2 / MSE)
        - Measures pixel-level accuracy
        - Higher is better (typically 20-40 dB for good reconstruction)
        - Good for comparing numerical similarity
        - Limitation: Doesn't always match human perception
    
    2. SSIM (Structural Similarity Index):
        Formula: SSIM = [luminance x contrast x structure]
        - Measures perceptual similarity based on human visual system
        - Range [0, 1], where 1 means identical, higher is better
        - Better correlates with human perception than PSNR
        - Considers luminance, contrast, and structural information
    
    3. LPIPS (Learned Perceptual Image Patch Similarity):
        - Uses deep network features (AlexNet) to measure similarity
        - Measures perceptual distance in feature space
        - Lower is better (typically 0.1-0.5)
        - Best correlation with human judgment of image quality
        - Based on: "The Unreasonable Effectiveness of Deep Features 
                     as a Perceptual Metric" (Zhang et al., 2018)
    
    4. MSE (Mean Squared Error):
        Formula: MSE = (1/N) * sum((x - x_hat)^2)
        - Basic pixel-wise error measure
        - Lower is better
        - Foundation for PSNR calculation
        - Simple but doesn't capture perceptual quality well
    
    Why use multiple metrics?
    - PSNR/MSE: Good for numerical accuracy
    - SSIM: Better for structural similarity
    - LPIPS: Best for perceptual quality (how humans see it)
    Together they give comprehensive quality assessment.
    """
    
    def __init__(self, device = 'cuda'):
        """
        Initialize metrics calculator.
        
        Sets up LPIPS model which requires loading a pre-trained AlexNet.
        This is done once at initialization to avoid reloading for each batch.
        
        Args:
            device: Device for LPIPS model (GPU recommended for speed)
                   CPU works but is slower for LPIPS computation
        """
        self.device = device
        
        # Initialize LPIPS model
        # LPIPS uses AlexNet features to measure perceptual similarity
        # 'alex' net is chosen because:
        # - Fast (smaller than VGG)
        # - Good perceptual correlation
        # - Standard choice in research
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        
        # Set to eval mode (no training, just inference)
        # This disables dropout and uses running batch norm statistics
        self.lpips_model.eval()
    
    @torch.no_grad()
    def calculate_metrics(self, original, reconstructed):
        """
        Calculate all metrics for a batch of images.
        
        This is the core function that computes all four metrics.
        We compute metrics on CPU (using scikit-image for PSNR/SSIM)
        and GPU (using LPIPS) for efficiency.
        
        Args:
            original: Original images [B, C, H, W] in range [0, 1]
                     Must be denormalized before passing to this function
            reconstructed: Reconstructed images [B, C, H, W] in range [0, 1]
                          Also denormalized
            
        Returns:
            metrics: Dict with mean and std for each metric:
                {
                    'psnr': mean PSNR across batch,
                    'psnr_std': standard deviation of PSNR,
                    'ssim': mean SSIM,
                    'ssim_std': std of SSIM,
                    'lpips': mean LPIPS (no std, computed on batch),
                    'mse': mean MSE
                }
        
        Note:
            @torch.no_grad() decorator disables gradient computation
            for efficiency (we're only evaluating, not training)
        """
        # Convert tensors to numpy arrays for scikit-image
        # scikit-image (used for PSNR/SSIM) requires numpy arrays
        # .cpu() moves tensor from GPU to CPU
        # .numpy() converts PyTorch tensor to numpy array
        orig_np = original.cpu().numpy()
        recon_np = reconstructed.cpu().numpy()
        
        # Get batch size (number of images)
        batch_size = orig_np.shape[0]
        
        # Initialize lists to store per-image scores
        # We compute PSNR and SSIM per image, then average
        psnr_scores = []
        ssim_scores = []
        
        # Calculate PSNR and SSIM per image
        # We loop through batch because scikit-image functions
        # work on single images, not batches
        for i in range(batch_size):
            # Convert from PyTorch format [C, H, W] to scikit-image format [H, W, C]
            # PyTorch uses channels-first (C, H, W)
            # scikit-image/matplotlib use channels-last (H, W, C)
            orig_img = np.transpose(orig_np[i], (1, 2, 0))  # [C, H, W] -> [H, W, C]
            recon_img = np.transpose(recon_np[i], (1, 2, 0))
            
            # Calculate PSNR (Peak Signal-to-Noise Ratio)
            # Formula: PSNR = 10 * log10(MAX^2 / MSE)
            # where MAX = 1.0 (since images are in [0, 1] range)
            # 
            # Interpretation:
            # - 20-25 dB: Low quality (visible artifacts)
            # - 25-30 dB: Medium quality
            # - 30-40 dB: Good quality
            # - >40 dB: Excellent quality
            psnr = skimage_psnr(orig_img, recon_img, data_range = 1.0)
            psnr_scores.append(psnr)
            
            # Calculate SSIM (Structural Similarity Index)
            # SSIM considers three components:
            # 1. Luminance: average intensity
            # 2. Contrast: standard deviation
            # 3. Structure: correlation
            # 
            # Parameters:
            # - data_range=1.0: images are in [0, 1]
            # - multichannel=True: image has multiple channels (RGB)
            # - channel_axis=2: channels are in dimension 2 (H, W, C format)
            # 
            # Interpretation:
            # - 0.0-0.5: Poor quality
            # - 0.5-0.7: Medium quality
            # - 0.7-0.9: Good quality
            # - 0.9-1.0: Excellent quality
            ssim = skimage_ssim(orig_img, recon_img, 
                               data_range = 1.0, 
                               multichannel = True,
                               channel_axis = 2)
            ssim_scores.append(ssim)
        
        # Calculate LPIPS (Learned Perceptual Image Patch Similarity)
        # LPIPS is computed on the entire batch at once (more efficient)
        # 
        # LPIPS expects images in range [-1, 1], but ours are in [0, 1]
        # Transform: [0, 1] -> [-1, 1] using: x_new = x * 2 - 1
        orig_lpips = original * 2 - 1
        recon_lpips = reconstructed * 2 - 1
        
        # Compute LPIPS distance
        # - Move tensors to GPU for faster computation
        # - LPIPS model extracts AlexNet features and compares them
        # - Lower LPIPS = more perceptually similar
        # - .mean() averages over spatial dimensions
        # - .item() converts single-element tensor to Python float
        # 
        # Interpretation:
        # - 0.0-0.2: Excellent perceptual quality
        # - 0.2-0.4: Good quality
        # - 0.4-0.6: Medium quality
        # - >0.6: Poor perceptual quality
        lpips_score = self.lpips_model(orig_lpips.to(self.device), 
                                        recon_lpips.to(self.device)).mean().item()
        
        # Calculate MSE (Mean Squared Error)
        # MSE = (1/N) * sum((x - x_hat)^2)
        # 
        # PyTorch's F.mse_loss computes this efficiently
        # Lower MSE = better reconstruction
        # 
        # Note: MSE is related to PSNR by: PSNR = 10 * log10(1 / MSE)
        mse = F.mse_loss(original, reconstructed).item()
        
        # Return all metrics as a dictionary
        # For PSNR and SSIM, we return both mean and std across batch
        # For LPIPS and MSE, we just return the value (already aggregated)
        return {
            'psnr': np.mean(psnr_scores),  # Average PSNR
            'psnr_std': np.std(psnr_scores),  # Standard deviation of PSNR
            'ssim': np.mean(ssim_scores),  # Average SSIM
            'ssim_std': np.std(ssim_scores),  # Standard deviation of SSIM
            'lpips': lpips_score,  # LPIPS distance
            'mse': mse  # Mean squared error
        }
    
    @torch.no_grad()
    def evaluate_model(self, encoder, decoder, dataloader, device, max_batches = None):
        """
        Evaluate model on entire dataset (or subset).
        
        This function runs the full evaluation pipeline:
        1. Load batch of images
        2. Extract features with encoder
        3. Reconstruct with decoder
        4. Denormalize both original and reconstruction
        5. Calculate metrics
        6. Aggregate metrics across all batches
        
        Process for each batch:
        - Forward pass: z = f_theta(x), x_hat = g_psi(z)
        - Denormalize: convert from ImageNet normalization to [0, 1]
        - Calculate metrics: PSNR, SSIM, LPIPS, MSE
        
        Args:
            encoder: Feature extractor (frozen, pre-trained)
            decoder: Decoder model (trained to invert features)
            dataloader: DataLoader with evaluation images
            device: Device to run on (cuda/mps/cpu)
            max_batches: Maximum number of batches to evaluate
                        None = evaluate entire dataset
                        Useful for quick evaluation during development
                        Example: max_batches=20 evaluates on 20 batches only
            
        Returns:
            results: Dict with aggregated metrics across all batches:
                {
                    'psnr_mean': average PSNR across all images,
                    'psnr_std': std of PSNR across batches,
                    'ssim_mean': average SSIM,
                    'ssim_std': std of SSIM,
                    'lpips_mean': average LPIPS,
                    'lpips_std': std of LPIPS across batches,
                    'mse_mean': average MSE,
                    'mse_std': std of MSE
                }
        """
        # Set models to evaluation mode
        # This is important because:
        # - Disables dropout (we want deterministic outputs)
        # - Batch normalization uses running statistics (not batch statistics)
        # - Some models behave differently in train vs eval mode
        encoder.eval()
        decoder.eval()
        
        # Move models to specified device
        encoder.to(device)
        decoder.to(device)
        
        # Initialize storage for metrics from each batch
        # We'll compute metrics per batch, then aggregate at the end
        all_metrics = {
            'psnr': [],  # List of PSNR values (one per batch)
            'ssim': [],  # List of SSIM values
            'lpips': [],  # List of LPIPS values
            'mse': []  # List of MSE values
        }
        
        # Iterate over batches in dataloader
        # tqdm provides progress bar showing evaluation progress
        for batch_idx, images in enumerate(tqdm(dataloader, desc = 'Evaluating')):
            # Stop if we've reached max_batches (for quick evaluation)
            if max_batches and batch_idx >= max_batches:
                break
            
            # Move batch to device (GPU/CPU/MPS)
            images = images.to(device)
            
            # Forward pass through encoder and decoder
            # Step 1: Extract features z = f_theta(x)
            # encoder is frozen, so this just extracts features
            features = encoder(images)
            
            # Step 2: Reconstruct images x_hat = g_psi(z)
            # decoder takes features and tries to reconstruct original
            reconstructed = decoder(features)
            
            # Denormalize both original and reconstructed images
            # Our images are normalized with ImageNet statistics:
            # - mean = [0.485, 0.456, 0.406]
            # - std = [0.229, 0.224, 0.225]
            # 
            # Metrics expect images in [0, 1] range, so we denormalize:
            # img = (normalized * std) + mean
            # 
            # This is crucial! Computing metrics on normalized images
            # would give incorrect results
            images_denorm = denormalize(images)
            recon_denorm = denormalize(reconstructed)
            
            # Calculate metrics for this batch
            # Returns dict with psnr, ssim, lpips, mse
            metrics = self.calculate_metrics(images_denorm, recon_denorm)
            
            # Store metrics from this batch
            # We'll aggregate these at the end
            all_metrics['psnr'].append(metrics['psnr'])
            all_metrics['ssim'].append(metrics['ssim'])
            all_metrics['lpips'].append(metrics['lpips'])
            all_metrics['mse'].append(metrics['mse'])
        
        # Aggregate results across all batches
        # We compute mean and standard deviation for each metric
        # 
        # Mean: average value across all batches (overall performance)
        # Std: how much metrics vary across batches (stability)
        # 
        # Lower std = more consistent performance across different images
        results = {
            'psnr_mean': np.mean(all_metrics['psnr']),
            'psnr_std': np.std(all_metrics['psnr']),
            'ssim_mean': np.mean(all_metrics['ssim']),
            'ssim_std': np.std(all_metrics['ssim']),
            'lpips_mean': np.mean(all_metrics['lpips']),
            'lpips_std': np.std(all_metrics['lpips']),
            'mse_mean': np.mean(all_metrics['mse']),
            'mse_std': np.std(all_metrics['mse'])
        }
        
        return results


def save_metrics(results, config, save_dir = 'results'):
    """
    Save evaluation metrics to CSV file.
    
    Creates a CSV file with all metrics for this experiment.
    This allows easy comparison across experiments and import into
    papers, reports, or analysis tools.
    
    File is saved in architecture-specific folder for organization.
    Example: results/resnet34/metrics/resnet34_layer3_attention_metrics.csv
    
    Args:
        results: Dict from MetricsCalculator.evaluate_model()
                Contains all metrics (psnr_mean, ssim_mean, etc.)
        config: Configuration dict with:
            - architecture: e.g., 'resnet34'
            - layer_name: e.g., 'layer3'
            - decoder_type: e.g., 'attention'
        save_dir: Base directory for results (default 'results')
    
    Returns:
        df: Pandas DataFrame with results (also saved to CSV)
    """
    import pandas as pd
    
    # Create unique experiment name from configuration
    # This ensures each experiment's metrics are stored separately
    arch_name = config['architecture']
    layer_name = config['layer_name']
    decoder_type = config['decoder_type']
    experiment_name = f"{arch_name}_{layer_name}_{decoder_type}"
    
    # Create save path in architecture-specific folder
    # Structure: results/resnet34/metrics/resnet34_layer3_attention_metrics.csv
    save_path = Path(save_dir) / arch_name / 'metrics' / f'{experiment_name}_metrics.csv'
    
    # Create directory if it doesn't exist (cross-platform)
    save_path.parent.mkdir(parents = True, exist_ok = True)
    
    # Convert results dict to DataFrame
    # [results] wraps dict in list to create single-row DataFrame
    # This makes it easy to concatenate results from multiple experiments later
    df = pd.DataFrame([results])
    
    # Set index to experiment name for clarity
    df.index = [experiment_name]
    
    # Save to CSV
    # CSV format is:
    # - Human readable (can open in Excel)
    # - Easy to parse programmatically
    # - Standard format for sharing data
    df.to_csv(save_path)
    
    print(f"[SAVED] Metrics: {save_path}")
    
    return df


def print_metrics(results, title = "Evaluation Results"):
    """
    Pretty print evaluation metrics to console.
    
    Formats metrics with appropriate precision for readability:
    - PSNR: 2 decimal places (e.g., 24.53 dB)
    - SSIM: 4 decimal places (e.g., 0.8234)
    - LPIPS: 4 decimal places (e.g., 0.3456)
    - MSE: 6 decimal places (e.g., 0.012345)
    
    Different precision levels match typical reporting in research papers.
    
    Args:
        results: Dict from MetricsCalculator.evaluate_model()
                Must contain keys: psnr_mean, psnr_std, ssim_mean, etc.
        title: Header title for the output (e.g., "ResNet34-layer3 Results")
    
    Example output:
        ============================================================
        Evaluation Results
        ============================================================
          PSNR:  24.53 +/- 1.23 dB
          SSIM:  0.8234 +/- 0.0456
          LPIPS: 0.3456 +/- 0.0234
          MSE:   0.012345 +/- 0.001234
        ============================================================
    """
    # Print header with separator line
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    # Print PSNR (Peak Signal-to-Noise Ratio)
    # Format: mean +/- std, 2 decimal places, with dB unit
    # Higher is better (typically 20-40 dB)
    print(f"  PSNR:  {results['psnr_mean']:.2f} +/- {results['psnr_std']:.2f} dB")
    
    # Print SSIM (Structural Similarity Index)
    # Format: mean +/- std, 4 decimal places
    # Range [0, 1], higher is better
    print(f"  SSIM:  {results['ssim_mean']:.4f} +/- {results['ssim_std']:.4f}")
    
    # Print LPIPS (Learned Perceptual Image Patch Similarity)
    # Format: mean +/- std, 4 decimal places
    # Lower is better (typically 0.1-0.5)
    print(f"  LPIPS: {results['lpips_mean']:.4f} +/- {results['lpips_std']:.4f}")
    
    # Print MSE (Mean Squared Error)
    # Format: mean +/- std, 6 decimal places
    # Lower is better
    print(f"  MSE:   {results['mse_mean']:.6f} +/- {results['mse_std']:.6f}")
    
    # Print footer separator
    print(f"{'='*60}\n")