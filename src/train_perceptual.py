"""
Training with perceptual loss (LPIPS) for enhanced feature inversion.

This is Run 1 of the enhanced experiments series.
Combines MSE loss (pixel accuracy) with LPIPS loss (perceptual quality).
Everything else remains identical to baseline to isolate perceptual loss contribution.

Cross-platform compatible: Windows, Mac (M1/M2), Linux
No special characters or emojis used for maximum compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import lpips
from pathlib import Path
import platform
import pandas as pd
import matplotlib.pyplot as plt

def train_epoch_perceptual(encoder, decoder, dataloader, optimizer, device, lpips_model, 
                           mse_weight = 0.5, lpips_weight = 0.5):
    """
    Train one epoch with combined MSE and LPIPS loss.
    
    The combined loss optimizes for both pixel-level accuracy (MSE) and
    perceptual quality (LPIPS), resulting in sharper reconstructions.
    
    Args:
        encoder: Frozen pre-trained feature extractor (ResNet/VGG/ViT)
        decoder: Trainable decoder network with attention mechanisms
        dataloader: PyTorch DataLoader with training images
        optimizer: Adam optimizer for decoder parameters
        device: Device to run on ('cuda', 'mps', or 'cpu')
        lpips_model: Pre-trained LPIPS model (AlexNet backbone)
        mse_weight: Weight for MSE loss component (default 0.5)
        lpips_weight: Weight for LPIPS loss component (default 0.5)
        
    Returns:
        avg_loss: Average total loss across all batches
        avg_mse: Average MSE component across all batches
        avg_lpips: Average LPIPS component across all batches
        
    Note:
        Encoder remains frozen (no gradient computation).
        Only decoder parameters are updated.
    """
    # Set models to appropriate modes
    # Decoder in training mode: enables dropout, batch norm updates
    decoder.train()
    
    # Encoder in eval mode: frozen, no gradient computation
    encoder.eval()
    
    # Initialize accumulators for loss tracking
    total_loss = 0.0
    total_mse = 0.0
    total_lpips = 0.0
    num_batches = 0
    
    # Iterate over training batches
    # tqdm provides progress bar with batch information
    for images in tqdm(dataloader, desc = 'Training', leave = False):
        # Move images to device (GPU/CPU)
        images = images.to(device)
        
        # Step 1: Extract features with frozen encoder
        # torch.no_grad() disables gradient computation for efficiency
        with torch.no_grad():
            features = encoder(images)
        
        # Step 2: Reconstruct images with trainable decoder
        reconstructed = decoder(features)
        
        # Step 3a: Compute MSE loss (pixel-level accuracy)
        # MSE = (1/N) * sum((x - x_hat)^2)
        # Lower MSE means better pixel-wise match
        mse_loss = F.mse_loss(reconstructed, images)
        
        # Step 3b: Compute LPIPS loss (perceptual quality)
        # LPIPS expects images in [-1, 1] range
        # Our images are in [0, 1], so we scale: x_new = x * 2 - 1
        recon_scaled = reconstructed * 2 - 1
        target_scaled = images * 2 - 1
        
        # LPIPS computes perceptual distance using deep features
        # Lower LPIPS means more perceptually similar
        lpips_loss = lpips_model(recon_scaled, target_scaled).mean()
        
        # Step 3c: Combine losses with weights
        # Total loss = alpha * MSE + beta * LPIPS
        # This balances pixel accuracy with perceptual quality
        loss = mse_weight * mse_loss + lpips_weight * lpips_loss
        
        # Step 4: Backward pass and optimization
        # Clear gradients from previous iteration
        optimizer.zero_grad()
        
        # Compute gradients via backpropagation
        loss.backward()
        
        # Update decoder parameters
        optimizer.step()
        
        # Step 5: Accumulate losses for epoch average
        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_lpips += lpips_loss.item()
        num_batches += 1
    
    # Compute average losses across all batches
    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches
    avg_lpips = total_lpips / num_batches
    
    return avg_loss, avg_mse, avg_lpips


def validate_perceptual(encoder, decoder, dataloader, device, lpips_model, 
                        mse_weight = 0.5, lpips_weight = 0.5):
    """
    Validate model with combined MSE and LPIPS loss.
    
    Computes validation loss without updating model parameters.
    Used for learning rate scheduling and checkpoint selection.
    
    Args:
        encoder: Frozen pre-trained feature extractor
        decoder: Trainable decoder network
        dataloader: PyTorch DataLoader with validation images
        device: Device to run on ('cuda', 'mps', or 'cpu')
        lpips_model: Pre-trained LPIPS model (AlexNet backbone)
        mse_weight: Weight for MSE loss component (default 0.5)
        lpips_weight: Weight for LPIPS loss component (default 0.5)
        
    Returns:
        avg_loss: Average total validation loss
        avg_mse: Average MSE component
        avg_lpips: Average LPIPS component
        
    Note:
        No gradient computation or parameter updates during validation.
        Both encoder and decoder in eval mode.
    """
    # Set both models to eval mode
    # This disables dropout and uses running batch norm statistics
    decoder.eval()
    encoder.eval()
    
    # Initialize loss accumulators
    total_loss = 0.0
    total_mse = 0.0
    total_lpips = 0.0
    num_batches = 0
    
    # Disable gradient computation for validation
    # This saves memory and speeds up computation
    with torch.no_grad():
        for images in tqdm(dataloader, desc = 'Validating', leave = False):
            # Move images to device
            images = images.to(device)
            
            # Forward pass through encoder and decoder
            features = encoder(images)
            reconstructed = decoder(features)
            
            # Compute MSE loss
            mse_loss = F.mse_loss(reconstructed, images)
            
            # Compute LPIPS loss
            # Scale images to [-1, 1] range for LPIPS
            recon_scaled = reconstructed * 2 - 1
            target_scaled = images * 2 - 1
            lpips_loss = lpips_model(recon_scaled, target_scaled).mean()
            
            # Combine losses
            loss = mse_weight * mse_loss + lpips_weight * lpips_loss
            
            # Accumulate losses
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_lpips += lpips_loss.item()
            num_batches += 1
    
    # Compute averages
    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches
    avg_lpips = total_lpips / num_batches
    
    return avg_loss, avg_mse, avg_lpips


def train_perceptual(encoder, decoder, train_loader, val_loader, 
                     epochs = 30, lr = 0.001, device='cuda', save_dir = 'results',
                     architecture = 'vgg16', layer_name = 'block1',
                     mse_weight = 0.5, lpips_weight=0.5):
    """
    Complete training pipeline with perceptual loss.
    
    Trains decoder to reconstruct images from encoder features using
    combined MSE and LPIPS loss. Implements learning rate scheduling,
    checkpoint saving, and early stopping.
    
    Args:
        encoder: Pre-trained frozen encoder (ResNet34/VGG16/ViT)
        decoder: Trainable decoder with attention mechanisms
        train_loader: DataLoader for training set (640 images)
        val_loader: DataLoader for validation set (160 images)
        epochs: Number of training epochs (default 30)
        lr: Initial learning rate (default 0.001)
        device: Device to run on ('cuda', 'mps', or 'cpu')
        save_dir: Base directory for saving results
        architecture: Architecture name ('resnet34', 'vgg16', 'vit_base_patch16_224')
        layer_name: Layer name ('layer1', 'block1', 'block0', etc.)
        mse_weight: Weight for MSE loss (default 0.5)
        lpips_weight: Weight for LPIPS loss (default 0.5)
        
    Returns:
        history: Dictionary with training history
            - train_loss: List of training losses per epoch
            - val_loss: List of validation losses per epoch
            - train_mse: List of training MSE per epoch
            - train_lpips: List of training LPIPS per epoch
            - best_val_loss: Best validation loss achieved
            - best_epoch: Epoch with best validation loss
        
    Checkpoint Strategy:
        - Saves best model (lowest validation loss)
        - Saves snapshots at epochs 10, 20, 30
        - Checkpoints saved in: save_dir/architecture/checkpoints/
        
    Learning Rate Schedule:
        - ReduceLROnPlateau: reduces LR when validation loss plateaus
        - Factor: 0.5 (halves learning rate)
        - Patience: 5 epochs
        - Minimum LR: 1e-6
        
    Cross-platform Compatibility:
        - Uses pathlib.Path for file paths (works on Windows/Mac/Linux)
        - Handles MPS (Apple Silicon), CUDA (NVIDIA), and CPU devices
        - No OS-specific code or special characters
    """
    # Initialize LPIPS model for perceptual loss
    # AlexNet backbone is fastest while maintaining quality
    print(f"[INFO] Initializing LPIPS model with AlexNet backbone...")
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    # Freeze LPIPS parameters (no training needed)
    for param in lpips_model.parameters():
        param.requires_grad = False
    lpips_model.eval()
    
    # Move models to device
    encoder.to(device)
    decoder.to(device)
    
    # Initialize optimizer for decoder parameters only
    # Adam optimizer with default betas (0.9, 0.999)
    optimizer = torch.optim.Adam(decoder.parameters(), lr = lr)
    
    # Initialize learning rate scheduler
    # Reduces LR by factor of 0.5 when validation loss plateaus
    # Patience of 5 means wait 5 epochs before reducing LR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode ='min', factor = 0.5, patience = 5, 
        min_lr = 1e-6
    )
        
    # Setup save directories using pathlib for cross-platform compatibility
    # pathlib.Path automatically handles Windows backslashes and Unix forward slashes
    save_dir = Path(save_dir)
    checkpoint_dir = save_dir / architecture / 'checkpoints_perceptual'
    checkpoint_dir.mkdir(parents=True, exist_ok = True)
    
    # Create experiment name for file naming
    experiment_name = f"{architecture}_{layer_name}_perceptual"
    
    # Initialize training history tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mse': [],
        'train_lpips': [],
        'val_mse': [],
        'val_lpips': [],
        'learning_rates': []
    }
    
    # Track best model for checkpoint saving
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Print training configuration
    print(f"\n{'='*70}")
    print(f"TRAINING CONFIGURATION - Run 1: Perceptual Loss")
    print(f"{'='*70}")
    print(f"Architecture: {architecture}")
    print(f"Layer: {layer_name}")
    print(f"Device: {device}")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {lr}")
    print(f"MSE Weight: {mse_weight}")
    print(f"LPIPS Weight: {lpips_weight}")
    print(f"Optimizer: Adam")
    print(f"Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
    print(f"Checkpoint Dir: {checkpoint_dir}")
    print(f"{'='*70}\n")
    
    # Record start time for total training duration
    training_start = time.time()
    
    # Main training loop
    for epoch in range(1, epochs + 1):
        # Record epoch start time
        epoch_start = time.time()
        
        # Print epoch header
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"{'-'*70}")
        
        # Training phase
        train_loss, train_mse, train_lpips = train_epoch_perceptual(
            encoder, decoder, train_loader, optimizer, device, lpips_model,
            mse_weight, lpips_weight
        )
        
        # Validation phase
        val_loss, val_mse, val_lpips = validate_perceptual(
            encoder, decoder, val_loader, device, lpips_model,
            mse_weight, lpips_weight
        )
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record metrics in history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mse'].append(train_mse)
        history['train_lpips'].append(train_lpips)
        history['val_mse'].append(val_mse)
        history['val_lpips'].append(val_lpips)
        history['learning_rates'].append(current_lr)
        
        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.6f} (MSE: {train_mse:.6f}, LPIPS: {train_lpips:.6f})")
        print(f"Val Loss:   {val_loss:.6f} (MSE: {val_mse:.6f}, LPIPS: {val_lpips:.6f})")
        print(f"LR: {current_lr:.6f} | Time: {epoch_duration:.1f}s")
        
        # Save checkpoint if best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            
            # Save best model checkpoint
            checkpoint_path = checkpoint_dir / f'{experiment_name}_best.pth'
            torch.save({
                'epoch': epoch,
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mse': val_mse,
                'val_lpips': val_lpips,
                'history': history
            }, checkpoint_path)
            
            print(f"[SAVED] Best model at epoch {epoch} (val_loss: {val_loss:.6f})")
        
        # Save periodic checkpoints at epochs 10, 20, 30
        if epoch in [10, 20, 30]:
            checkpoint_path = checkpoint_dir / f'{experiment_name}_epoch{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'history': history
            }, checkpoint_path)
            
            print(f"[SAVED] Epoch {epoch} checkpoint")
    
    # Calculate total training time
    total_training_time = time.time() - training_start


# After training loop completes, before final summary
    
    # Save training history to CSV
    metrics_dir = save_dir / architecture / 'metrics_perceptual'
    metrics_dir.mkdir(parents = True, exist_ok = True)
    
    # Create DataFrame with training history
    history_df = pd.DataFrame({
        'epoch': list(range(1, epochs + 1)),
        'train_loss': history['train_loss'],
        'train_mse': history['train_mse'],
        'train_lpips': history['train_lpips'],
        'val_loss': history['val_loss'],
        'val_mse': history['val_mse'],
        'val_lpips': history['val_lpips'],
        'learning_rate': history['learning_rates']
    })
    
    # Save to CSV
    csv_path = metrics_dir / f'{experiment_name}_training_history.csv'
    history_df.to_csv(csv_path, index=False)
    print(f"[SAVED] Training history: {csv_path}")
    
    # Generate training curves plot
    figures_dir = save_dir / architecture / 'figures_perceptual'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Combined Loss
    axes[0, 0].plot(history['train_loss'], label = 'Train', marker = 'o')
    axes[0, 0].plot(history['val_loss'], label='Validation', marker = 's')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Combined Loss (MSE + LPIPS)')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: MSE Component
    axes[0, 1].plot(history['train_mse'], label='Train MSE', marker = 'o')
    axes[0, 1].plot(history['val_mse'], label = 'Val MSE', marker = 's')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].set_title('MSE Component')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: LPIPS Component
    axes[1, 0].plot(history['train_lpips'], label = 'Train LPIPS', marker = 'o')
    axes[1, 0].plot(history['val_lpips'], label = 'Val LPIPS', marker = 's')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('LPIPS Loss')
    axes[1, 0].set_title('LPIPS (Perceptual) Component')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate
    axes[1, 1].plot(history['learning_rates'], marker = 'o', color = 'green')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Run 1: {architecture.upper()} {layer_name} - Perceptual Loss Training', 
                 fontsize = 14, fontweight = 'bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = figures_dir / f'{experiment_name}_training_curves.png'
    plt.savefig(plot_path, dpi = 150, bbox_inches = 'tight')
    plt.close()
    print(f"[SAVED] Training curves: {plot_path}")


    
    # Print training summary
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE - Run 1: Perceptual Loss")
    print(f"{'='*70}")
    print(f"Total Time: {total_training_time/60:.1f} minutes")
    print(f"Best Val Loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"Final LR: {history['learning_rates'][-1]:.6f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    print(f"{'='*70}\n")
    
    # Add summary to history
    history['best_val_loss'] = best_val_loss
    history['best_epoch'] = best_epoch
    history['total_time_minutes'] = total_training_time / 60
    
    return history