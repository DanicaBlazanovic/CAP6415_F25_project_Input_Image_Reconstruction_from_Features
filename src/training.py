"""
Model Training with Perceptual Loss for Image Reconstruction
============================================================

This module implements the core training loop for image reconstruction models.
It combines pixel-level MSE loss with perceptual LPIPS loss for high-quality
reconstructions that are both numerically accurate and visually appealing.

Key Features:
- Combined MSE + LPIPS loss for balanced reconstruction
- Early stopping to prevent overfitting
- Learning rate scheduling for optimal convergence
- Checkpoint saving (best, periodic, final)
- Comprehensive training history tracking
- Visualization of training curves
- Support for both single and ensemble models

Training Strategy:

Loss Function Design:
    Total Loss = alpha * MSE + beta * LPIPS
    
    Where:
    - alpha: Weight for pixel-level fidelity (default: 0.5)
    - beta: Weight for perceptual quality (default: 0.5)
    
    MSE (Mean Squared Error):
        - Measures pixel-wise reconstruction accuracy
        - Formula: mean((original - reconstructed)^2)
        - Ensures numerical fidelity
        - Fast to compute
        - Can produce blurry results if used alone
    
    LPIPS (Learned Perceptual Image Patch Similarity):
        - Measures perceptual similarity using deep features
        - Based on AlexNet features from multiple layers
        - Correlates better with human perception than MSE
        - Encourages sharp, natural-looking reconstructions
        - More expensive to compute than MSE
    
    Why Combine Both:
        MSE alone: Sharp but may miss perceptual quality
        LPIPS alone: Perceptually good but may drift from ground truth
        Combined: Best of both - accurate and visually appealing

LPIPS Details:
    Architecture: Pre-trained AlexNet feature extractor
    Layers used: conv1, conv2, conv3, conv4, conv5
    Input range: [-1, 1] (requires normalization from [0, 1])
    Output: Single scalar representing perceptual distance
    
    How it works:
    1. Extract features from both images at multiple layers
    2. Compute L2 distance between features at each layer
    3. Weight distances by learned importance weights
    4. Sum weighted distances across all layers
    
    Why AlexNet:
    - Fast inference (smaller than VGG or ResNet)
    - Sufficient perceptual quality
    - Well-calibrated for LPIPS metric
    - Widely used standard in image generation

Training Loop Design:

Epoch Structure:
    For each epoch:
    1. Training Phase:
       - Model in train mode (dropout active, batchnorm updates)
       - Iterate through training batches
       - Compute loss, backpropagate, update weights
       - Track training metrics
    
    2. Validation Phase:
       - Model in eval mode (deterministic behavior)
       - Iterate through validation batches
       - Compute loss without gradients
       - Track validation metrics
    
    3. Learning Rate Update:
       - Scheduler adjusts LR based on validation loss
       - ReduceLROnPlateau: Reduce LR when loss plateaus
       - Helps escape local minima
    
    4. Checkpoint Saving:
       - Save if validation loss improves (best model)
       - Save every 10 epochs (periodic checkpoints)
       - Save at end (final model)
    
    5. Early Stopping Check:
       - Stop if no improvement for N epochs
       - Prevents overfitting and saves time

Optimizer Configuration:
    Optimizer: Adam (Adaptive Moment Estimation)
        - Learning rate: 1e-4 (default)
        - Betas: (0.9, 0.999) (default)
        - Weight decay: 0 (no L2 regularization)
    
    Why Adam:
        - Adapts learning rate per parameter
        - Works well with sparse gradients
        - Requires minimal tuning
        - Standard choice for image reconstruction
    
    Learning Rate Schedule:
        ReduceLROnPlateau with:
        - Factor: 0.5 (halve LR on plateau)
        - Patience: 5 epochs before reduction
        - Min LR: 1e-6 (stop reducing below this)
        
        This allows model to make large steps initially,
        then fine-tune with smaller steps later.

Early Stopping Strategy:
    Patience: 15 epochs (default)
    
    How it works:
    - Track validation loss each epoch
    - If validation loss doesn't improve for 15 consecutive epochs, stop
    - This prevents overfitting to training data
    
    Why 15 epochs:
    - Training can plateau temporarily
    - Too short (5 epochs): May stop prematurely
    - Too long (30 epochs): Wastes time on overfitting
    - 15 epochs: Good balance for our dataset size
    
    Benefits:
    - Prevents overfitting automatically
    - Saves computation time
    - Best model is saved before overfitting begins

Checkpoint Saving Strategy:
    Three types of checkpoints:
    
    1. Best Model (saved when validation loss improves):
       - Filename: {exp_name}_best.pth
       - Contains: model weights, optimizer state, validation loss, history
       - Use this for final evaluation
       - Represents best generalization performance
    
    2. Periodic Checkpoints (every 10 epochs):
       - Filename: {exp_name}_epoch10.pth, epoch20.pth, etc.
       - Contains: model weights, optimizer state, history
       - Useful for resuming training
       - Useful for analyzing training progression
    
    3. Final Checkpoint (end of training):
       - Filename: {exp_name}_final.pth
       - Contains: model weights, history
       - May be overfit if training ran past best epoch
       - Useful for comparison with best model

Reproducibility Considerations:
    To ensure reproducible training:
    1. Set random seeds before training (handled in train_improved.py)
    2. Use deterministic DataLoader (fixed seed for train/val split)
    3. Disable cudnn benchmarking if needed (for perfect reproducibility)
    4. Save full config with checkpoint
    5. Track environment information
    
    Note: GPU operations may have slight non-determinism
    Results typically vary by <0.1% across runs even with fixed seeds

Training Progress Tracking:
    Metrics tracked per epoch:
    - Total loss (weighted sum of MSE + LPIPS)
    - MSE loss (pixel-level error)
    - LPIPS loss (perceptual error)
    - Learning rate
    - Best validation loss so far
    - Best epoch number
    - Time per epoch
    
    All metrics saved to:
    - CSV file: Easy to load in pandas/Excel
    - PNG plots: Visual training curves
    - Checkpoint history: Embedded in .pth files

Output Directory Structure:
    results/
    ├── single/                           (single architecture models)
    │   ├── checkpoints_{exp_name}/
    │   │   ├── {exp_name}_best.pth       (best validation loss)
    │   │   ├── {exp_name}_epoch10.pth    (periodic)
    │   │   ├── {exp_name}_epoch20.pth
    │   │   └── {exp_name}_final.pth      (last epoch)
    │   └── training_{exp_name}/
    │       ├── {exp_name}_history.csv    (per-epoch metrics)
    │       └── history_{exp_name}.png    (training curves)
    └── ensemble/                          (ensemble models)
        └── (same structure as single)

Usage Example:
    >>> from models import SingleArchModel
    >>> from train_perceptual import train_model
    >>> 
    >>> # Setup
    >>> config = {
    ...     'exp_name': 'vgg16_block1_transposed_conv',
    ...     'architecture': 'vgg16',
    ...     'vgg_block': 'block1',
    ...     'decoder_type': 'transposed_conv',
    ...     'lr': 1e-4,
    ...     'epochs': 30,
    ...     'mse_weight': 0.5,
    ...     'lpips_weight': 0.5,
    ...     'patience': 15,
    ...     'save_dir': 'results'
    ... }
    >>> 
    >>> model = SingleArchModel(config)
    >>> history = train_model(model, train_loader, val_loader, config, device='cuda')
    >>> 
    >>> print(f"Best validation loss: {history['best_val_loss']:.6f}")
    >>> print(f"Achieved at epoch: {history['best_epoch']}")

Naming Note:
    This module is currently named train_perceptual.py emphasizing the perceptual
    loss component. Consider renaming to:
    - training.py (simple, clear)
    - trainer.py (common ML pattern)
    - training_utils.py (clear it's utilities)
    - train_loop.py (describes functionality)

Author: Danica Blazanovic, Abbas Khan
Course: CAP6415 - Computer Vision, Fall 2025
Institution: Florida Atlantic University
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import lpips
import matplotlib.pyplot as plt
import pandas as pd
import time


def train_model(model, train_loader, val_loader, config, device='cuda'):
    """
    Train image reconstruction model with MSE + LPIPS loss.
    
    This is the main training loop that handles:
    - Forward and backward passes
    - Loss computation (MSE + LPIPS)
    - Optimizer updates
    - Validation evaluation
    - Learning rate scheduling
    - Early stopping
    - Checkpoint saving
    - Progress tracking and logging
    
    The function supports both single architecture models and ensemble models
    with automatic detection based on config structure.
    
    Training Process Overview:
    
    1. Setup Phase:
        - Determine model type (single vs ensemble)
        - Create checkpoint directory
        - Move model to device
        - Initialize loss functions (MSE, LPIPS)
        - Setup optimizer (Adam) and scheduler (ReduceLROnPlateau)
        - Initialize history tracking
    
    2. Training Loop (for each epoch):
        a. Training Phase:
           - Set model to training mode
           - Iterate through training batches
           - Forward pass: Generate reconstructions
           - Compute combined loss (MSE + LPIPS)
           - Backward pass: Compute gradients
           - Optimizer step: Update weights
           - Accumulate metrics
        
        b. Validation Phase:
           - Set model to evaluation mode
           - Iterate through validation batches (no gradients)
           - Forward pass: Generate reconstructions
           - Compute losses for monitoring
           - Accumulate metrics
        
        c. Learning Rate Update:
           - Scheduler reduces LR if validation plateaus
           - Helps model converge to better minimum
        
        d. Checkpoint Saving:
           - Save best model if validation improved
           - Save periodic checkpoints every 10 epochs
        
        e. Early Stopping Check:
           - Count epochs without improvement
           - Stop training if patience exceeded
    
    3. Finalization:
        - Save final checkpoint
        - Save training history to CSV
        - Generate training curve plots
        - Print summary statistics
    
    Loss Function Details:
    
    MSE Loss:
        - Pixel-level reconstruction error
        - Formula: mean((reconstruction - target)^2)
        - Input range: [0, 1] (ImageNet normalized)
        - Fast to compute
        - Ensures numerical accuracy
    
    LPIPS Loss:
        - Perceptual similarity using AlexNet features
        - Input range: [-1, 1] (requires normalization)
        - Conversion: images * 2 - 1 transforms [0,1] to [-1,1]
        - Slower to compute (forward pass through AlexNet)
        - Ensures visual quality
    
    Combined Loss:
        total_loss = mse_weight * mse + lpips_weight * lpips
        
        Default weights: 0.5 each (equal importance)
        Can adjust based on whether pixel accuracy or
        perceptual quality is more important
    
    Why Freeze LPIPS:
        LPIPS network is pre-trained and should not be updated:
        - Set requires_grad=False for all parameters
        - Set to eval mode (no batchnorm updates)
        - Only use for loss computation, not training
        - This is standard practice for perceptual losses
    
    Optimizer Details:
        
        Adam Optimizer:
            - Adaptive learning rates per parameter
            - Momentum with bias correction
            - Standard choice for deep learning
            - Learning rate: 1e-4 (default)
        
        Only Trainable Parameters:
            - Decoder parameters are trainable
            - Encoder parameters are frozen (pre-trained)
            - Filter trainable_params to save memory
            - Avoid computing gradients for frozen weights
    
    Learning Rate Scheduling:
        
        ReduceLROnPlateau:
            - Monitors validation loss
            - Reduces LR by factor (0.5) when loss plateaus
            - Patience: Wait 5 epochs before reducing
            - Min LR: 1e-6 (don't reduce below this)
        
        Why This Schedule:
            - Start with larger LR for fast initial progress
            - Reduce LR when stuck to fine-tune
            - Automatic adjustment based on validation
            - No need to manually tune schedule
    
    Early Stopping Logic:
        
        Track epochs_no_improve:
            - Reset to 0 when validation loss improves
            - Increment by 1 each epoch without improvement
            - Stop training when reaches patience (default: 15)
        
        Why Early Stopping:
            - Prevents overfitting on training data
            - Saves computation time
            - Best model already saved before overfitting
            - Validation loss starts increasing when overfit
    
    Checkpoint Strategy:
        
        Best Checkpoint:
            - Saved whenever validation loss improves
            - Contains: weights, optimizer, loss, history, config
            - This is the model to use for evaluation
            - Represents best generalization performance
        
        Periodic Checkpoints:
            - Saved every 10 epochs
            - Useful for resuming interrupted training
            - Useful for analyzing training progression
            - Contains: weights, optimizer, history
        
        Final Checkpoint:
            - Saved at end of training
            - May be overfit if trained past best epoch
            - Contains: weights, history
            - Useful for comparison
    
    Training vs Evaluation Mode:
        
        Training Mode (model.train()):
            - Dropout layers active (random neurons dropped)
            - BatchNorm updates running statistics
            - Stochastic behavior for regularization
            - Use during training phase
        
        Evaluation Mode (model.eval()):
            - Dropout disabled (use all neurons)
            - BatchNorm uses fixed running statistics
            - Deterministic behavior
            - Use during validation and inference
    
    Gradient Computation:
        
        Training: Gradients computed for backpropagation
            - Forward pass: Compute loss
            - Backward pass: loss.backward() computes gradients
            - Optimizer step: Updates weights using gradients
            - Zero gradients before each batch
        
        Validation: No gradient computation (torch.no_grad())
            - Saves memory (no gradient storage)
            - Speeds up computation
            - Validation doesn't update weights
    
    History Tracking:
        
        Per-Epoch Metrics:
            - Training: loss, MSE, LPIPS
            - Validation: loss, MSE, LPIPS
            - Learning rate
            - Best validation loss so far
            - Best epoch number
        
        Saved to CSV:
            - One row per epoch
            - Easy to analyze in pandas/Excel
            - Can plot custom curves
        
        Saved in Checkpoints:
            - Full history embedded
            - Can resume training with history
            - Can analyze training progression
    
    Memory Management:
        
        Gradient Accumulation:
            - Not implemented (batch size sufficient)
            - Could add for larger models
        
        Mixed Precision:
            - Not implemented (float32 only)
            - Could add for faster training
        
        Checkpoint Cleanup:
            - Periodic checkpoints accumulate
            - Manual cleanup may be needed
            - Consider keeping only last N checkpoints
    
    Args:
        model (nn.Module): Model to train (single or ensemble).
                          Should implement forward() returning reconstructions.
                          Decoder parameters should have requires_grad=True.
                          Encoder parameters should have requires_grad=False.
        
        train_loader (DataLoader): Training data loader.
                                  Should provide batches of images.
                                  Shuffled for varied mini-batches.
                                  Typical batch_size: 4-16.
        
        val_loader (DataLoader): Validation data loader.
                                Should provide batches of images.
                                Not shuffled for consistent evaluation.
                                Same batch_size as training.
        
        config (dict): Configuration dictionary containing:
            Required keys:
            - exp_name (str): Experiment name for file naming
            - save_dir (str): Root directory for saving
            - lr (float): Learning rate (default: 1e-4)
            - epochs (int): Maximum training epochs
            - mse_weight (float): Weight for MSE loss
            - lpips_weight (float): Weight for LPIPS loss
            
            Optional keys:
            - patience (int): Early stopping patience (default: 15)
            - architectures (list): Present if ensemble model
        
        device (str, optional): Device for training.
                               Options: 'cuda', 'mps', 'cpu'
                               Default: 'cuda'
    
    Returns:
        dict: Training history containing:
            - train_loss (list): Training loss per epoch
            - train_mse (list): Training MSE per epoch
            - train_lpips (list): Training LPIPS per epoch
            - val_loss (list): Validation loss per epoch
            - val_mse (list): Validation MSE per epoch
            - val_lpips (list): Validation LPIPS per epoch
            - learning_rates (list): Learning rate per epoch
            - best_val_loss (float): Best validation loss achieved
            - best_epoch (int): Epoch with best validation loss
            - total_time_minutes (float): Total training time
    
    Example:
        >>> config = {
        ...     'exp_name': 'vgg16_block1_transposed_conv',
        ...     'architecture': 'vgg16',
        ...     'vgg_block': 'block1',
        ...     'decoder_type': 'transposed_conv',
        ...     'lr': 1e-4,
        ...     'epochs': 30,
        ...     'mse_weight': 0.5,
        ...     'lpips_weight': 0.5,
        ...     'patience': 15,
        ...     'save_dir': 'results'
        ... }
        >>> 
        >>> from models import SingleArchModel
        >>> model = SingleArchModel(config)
        >>> 
        >>> history = train_model(model, train_loader, val_loader, config, 'cuda')
        >>> 
        >>> print(f"Training completed in {history['total_time_minutes']:.1f} minutes")
        >>> print(f"Best validation loss: {history['best_val_loss']:.6f}")
        >>> print(f"Best epoch: {history['best_epoch']}")
    
    Performance:
        Training time depends on:
        - Model size (ensemble 4x slower than single)
        - Dataset size (640 training images for DIV2K)
        - Batch size (larger = faster per epoch)
        - Device (CUDA >> MPS > CPU)
        
        Typical training times (30 epochs, batch_size=4):
        - VGG16 single on CUDA: 20-30 minutes
        - Ensemble on CUDA: 80-120 minutes
        - VGG16 single on MPS: 40-60 minutes
    
    Note on Reproducibility:
        For reproducible results:
        1. Set random seeds before calling (in train_improved.py)
        2. Use same DataLoader configuration
        3. Disable cudnn benchmarking if needed
        4. Use same PyTorch version
        5. Results may vary slightly on GPU (<0.1%)
    """
    
    # Determine model type from config structure
    # Ensemble models have 'architectures' key (list of networks)
    # Single models have 'architecture' key (single network)
    model_type = 'ensemble' if 'architectures' in config else 'single'
    
    # Get experiment name for file naming and logging
    arch_name = config.get('exp_name', 'experiment')
    
    # Create checkpoint directory for saving model weights
    # Structure: results/single/ or results/ensemble/
    save_path = Path(config['save_dir']) / model_type
    checkpoint_dir = save_path / f"checkpoints_{arch_name}"
    
    # Create directory if it doesn't exist
    # parents=True creates intermediate directories
    # exist_ok=True doesn't raise error if directory exists
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Print training configuration for logging
    print(f"\n{'='*70}")
    print(f"Training: {arch_name}")
    print(f"Type: {model_type.upper()}")
    print(f"Device: {device} | LR: {config['lr']} | Epochs: {config['epochs']}")
    print(f"MSE: {config['mse_weight']} | LPIPS: {config['lpips_weight']}")
    print(f"{'='*70}\n")
    
    # Move model to specified device (GPU or CPU)
    # All subsequent operations will run on this device
    model = model.to(device)
    
    # Initialize MSE loss function
    # Measures pixel-level reconstruction error
    # Formula: mean((reconstruction - target)^2)
    mse_loss_fn = nn.MSELoss()
    
    # Initialize LPIPS loss function
    # Pre-trained AlexNet for perceptual similarity
    # net='alex': Use AlexNet features (fast, good quality)
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    
    # Freeze LPIPS network parameters
    # LPIPS should not be trained, only used for loss computation
    # This saves memory and prevents unintended updates
    for param in lpips_loss_fn.parameters():
        param.requires_grad = False
    
    # Set LPIPS to evaluation mode
    # Uses fixed batchnorm statistics, no dropout
    # Ensures consistent loss computation
    lpips_loss_fn.eval()
    
    # Setup optimizer with only trainable parameters
    # Filter out frozen encoder parameters to save memory
    # Only decoder parameters have requires_grad=True
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Adam optimizer with specified learning rate
    # Adam adapts learning rate per parameter
    # Good default choice for deep learning
    optimizer = optim.Adam(trainable_params, lr=config['lr'])
    
    # Learning rate scheduler reduces LR when validation plateaus
    # mode='min': Reduce when validation loss stops decreasing
    # factor=0.5: Halve learning rate on plateau
    # patience=5: Wait 5 epochs before reducing
    # min_lr=1e-6: Don't reduce below this value
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Initialize history dictionary for tracking metrics
    # Lists accumulate values across epochs
    history = {
        # Training metrics per epoch
        'train_loss': [],       # Total loss (MSE + LPIPS)
        'train_mse': [],        # MSE component
        'train_lpips': [],      # LPIPS component
        
        # Validation metrics per epoch
        'val_loss': [],         # Total loss
        'val_mse': [],          # MSE component
        'val_lpips': [],        # LPIPS component
        
        # Training state
        'learning_rates': [],   # Learning rate per epoch
        'best_val_loss': float('inf'),  # Best validation loss so far
        'best_epoch': 0         # Epoch with best validation loss
    }
    
    # Early stopping counter
    # Increments when validation doesn't improve
    # Reset to 0 when validation improves
    epochs_no_improve = 0
    
    # Start timer for total training time
    training_start = time.time()
    
    # Main training loop over epochs
    # Epoch numbers start at 1 for human-readable logging
    for epoch in range(1, config['epochs'] + 1):
        # Start timer for this epoch
        epoch_start = time.time()
        
        # Print epoch header
        print(f"\nEpoch {epoch}/{config['epochs']}")
        print(f"{'-'*70}")
        
        # ==================== TRAINING PHASE ====================
        
        # Set model to training mode
        # Enables dropout, updates batchnorm running stats
        model.train()
        
        # Initialize accumulators for batch losses
        # Sum losses across all batches, then average
        train_loss_sum = 0.0
        train_mse_sum = 0.0
        train_lpips_sum = 0.0
        
        # Iterate through training batches
        # tqdm creates progress bar
        # leave=False removes progress bar after epoch completes
        for images in tqdm(train_loader, desc='Training', leave=False):
            # Move batch to device (GPU or CPU)
            images = images.to(device)
            
            # Forward pass: Generate reconstructions
            # Model takes normalized images and produces reconstructions
            reconstructed = model(images)
            
            # Compute MSE loss (pixel-level error)
            # Input: reconstructed and target images in [0, 1] range
            # Output: scalar mean squared error
            mse_loss = mse_loss_fn(reconstructed, images)
            
            # Compute LPIPS loss (perceptual error)
            # LPIPS expects inputs in [-1, 1] range
            # Convert from [0, 1]: x * 2 - 1
            # .mean() reduces per-image losses to single scalar
            lpips_loss = lpips_loss_fn(images * 2 - 1, reconstructed * 2 - 1).mean()
            
            # Compute weighted total loss
            # Combine MSE and LPIPS with specified weights
            # Default: 0.5 * MSE + 0.5 * LPIPS (equal importance)
            total_loss = config['mse_weight'] * mse_loss + config['lpips_weight'] * lpips_loss
            
            # Zero gradients from previous iteration
            # PyTorch accumulates gradients by default
            # Must zero before each backward pass
            optimizer.zero_grad()
            
            # Backward pass: Compute gradients
            # Computes gradient of total_loss w.r.t. all parameters
            # Gradients stored in parameter.grad
            total_loss.backward()
            
            # Optimizer step: Update parameters
            # Uses computed gradients to update weights
            # Adam applies adaptive learning rates
            optimizer.step()
            
            # Accumulate losses for this batch
            # .item() extracts Python float from tensor
            train_loss_sum += total_loss.item()
            train_mse_sum += mse_loss.item()
            train_lpips_sum += lpips_loss.item()
        
        # Calculate average training losses over all batches
        # Divide sum by number of batches for per-batch average
        avg_train_loss = train_loss_sum / len(train_loader)
        avg_train_mse = train_mse_sum / len(train_loader)
        avg_train_lpips = train_lpips_sum / len(train_loader)
        
        # ==================== VALIDATION PHASE ====================
        
        # Set model to evaluation mode
        # Disables dropout, uses fixed batchnorm stats
        # Ensures deterministic behavior for validation
        model.eval()
        
        # Initialize accumulators for validation losses
        val_loss_sum = 0.0
        val_mse_sum = 0.0
        val_lpips_sum = 0.0
        
        # Disable gradient computation for validation
        # Saves memory and speeds up forward pass
        # Gradients not needed since we're not training
        with torch.no_grad():
            # Iterate through validation batches
            for images in tqdm(val_loader, desc='Validating', leave=False):
                # Move batch to device
                images = images.to(device)
                
                # Forward pass: Generate reconstructions
                # No gradient tracking, faster than training
                reconstructed = model(images)
                
                # Compute MSE loss
                mse_loss = mse_loss_fn(reconstructed, images)
                
                # Compute LPIPS loss
                # Same normalization as training: [0,1] -> [-1,1]
                lpips_loss = lpips_loss_fn(images * 2 - 1, reconstructed * 2 - 1).mean()
                
                # Compute total loss
                # Same weighting as training for fair comparison
                total_loss = config['mse_weight'] * mse_loss + config['lpips_weight'] * lpips_loss
                
                # Accumulate validation losses
                val_loss_sum += total_loss.item()
                val_mse_sum += mse_loss.item()
                val_lpips_sum += lpips_loss.item()
        
        # Calculate average validation losses
        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_mse = val_mse_sum / len(val_loader)
        avg_val_lpips = val_lpips_sum / len(val_loader)
        
        # ==================== LEARNING RATE UPDATE ====================
        
        # Update learning rate based on validation loss
        # ReduceLROnPlateau monitors validation and reduces LR if plateaued
        scheduler.step(avg_val_loss)
        
        # Get current learning rate for logging
        # optimizer.param_groups[0] is the parameter group for all params
        current_lr = optimizer.param_groups[0]['lr']
        
        # ==================== HISTORY TRACKING ====================
        
        # Append all metrics to history lists
        history['train_loss'].append(avg_train_loss)
        history['train_mse'].append(avg_train_mse)
        history['train_lpips'].append(avg_train_lpips)
        history['val_loss'].append(avg_val_loss)
        history['val_mse'].append(avg_val_mse)
        history['val_lpips'].append(avg_val_lpips)
        history['learning_rates'].append(current_lr)
        
        # ==================== EPOCH TIMING ====================
        
        # Calculate time taken for this epoch
        epoch_duration = time.time() - epoch_start
        
        # ==================== LOGGING ====================
        
        # Print epoch summary with all metrics
        print(f"Train: {avg_train_loss:.6f} (MSE: {avg_train_mse:.6f}, LPIPS: {avg_train_lpips:.6f})")
        print(f"Val:   {avg_val_loss:.6f} (MSE: {avg_val_mse:.6f}, LPIPS: {avg_val_lpips:.6f})")
        print(f"LR: {current_lr:.6f} | Time: {epoch_duration:.1f}s")
        
        # ==================== CHECKPOINT SAVING ====================
        
        # Check if validation loss improved
        if avg_val_loss < history['best_val_loss']:
            # Update best validation loss and epoch
            history['best_val_loss'] = avg_val_loss
            history['best_epoch'] = epoch
            
            # Reset early stopping counter
            # Model is improving, continue training
            epochs_no_improve = 0
            
            # Save best model checkpoint
            # This is the model to use for evaluation
            torch.save({
                'epoch': epoch,                             # Current epoch number
                'model_state_dict': model.state_dict(),     # Model parameters
                'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
                'val_loss': avg_val_loss,                   # Validation loss
                'history': history,                         # Training history
                'config': config                            # Configuration
            }, checkpoint_dir / f"{arch_name}_best.pth")
            
            print(f"[SAVED] Best model (val_loss: {avg_val_loss:.6f})")
        
        else:
            # Validation loss did not improve
            # Increment early stopping counter
            epochs_no_improve += 1
        
        # ==================== EARLY STOPPING CHECK ====================
        
        # Check if patience exceeded
        # Default patience: 15 epochs without improvement
        if epochs_no_improve >= config.get('patience', 15):
            print(f"\n[EARLY STOP] No improvement for {config['patience']} epochs")
            break
        
        # ==================== PERIODIC CHECKPOINT ====================
        
        # Save checkpoint every 10 epochs
        # Useful for resuming training or analyzing progression
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, checkpoint_dir / f"{arch_name}_epoch{epoch}.pth")
    
    # ==================== FINAL CHECKPOINT ====================
    
    # Save final model state at end of training
    # May be overfit if training ran past best epoch
    torch.save({
        'epoch': epoch,                         # Final epoch number
        'model_state_dict': model.state_dict(),
        'history': history
    }, checkpoint_dir / f"{arch_name}_final.pth")
    
    # ==================== TRAINING COMPLETE ====================
    
    # Calculate total training time
    total_time = time.time() - training_start
    
    # ==================== SAVE METRICS ====================
    
    # Create training directory for CSV and plots
    training_dir = save_path / f"training_{arch_name}"
    training_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training history to CSV
    # One row per epoch with all metrics
    # Easy to load in pandas for analysis
    pd.DataFrame({
        'epoch': list(range(1, len(history['train_loss']) + 1)),
        'train_loss': history['train_loss'],
        'train_mse': history['train_mse'],
        'train_lpips': history['train_lpips'],
        'val_loss': history['val_loss'],
        'val_mse': history['val_mse'],
        'val_lpips': history['val_lpips'],
        'learning_rate': history['learning_rates']
    }).to_csv(training_dir / f"{arch_name}_history.csv", index=False)
    
    # ==================== PLOT TRAINING CURVES ====================
    
    # Generate visualization of training history
    # Shows loss curves and learning rate over time
    plot_history(history, training_dir, arch_name)
    
    # ==================== PRINT SUMMARY ====================
    
    # Print final training summary
    print(f"\n{'='*70}")
    print(f"COMPLETE | Time: {total_time/60:.1f}m | Best: {history['best_val_loss']:.6f} @ epoch {history['best_epoch']}")
    print(f"{'='*70}\n")
    
    # Add total time to history for reference
    history['total_time_minutes'] = total_time / 60
    
    # Return history dictionary for programmatic access
    return history


def plot_history(history, save_path, exp_name):
    """
    Create and save training history visualizations.
    
    This function generates a 4-panel visualization showing:
    1. Total loss (MSE + LPIPS) over epochs
    2. MSE loss component over epochs
    3. LPIPS loss component over epochs
    4. Learning rate schedule over epochs
    
    The plots help diagnose training behavior:
    - Convergence: Losses should decrease over time
    - Overfitting: Training loss decreases but validation increases
    - Plateau: Both losses flat, may need LR reduction
    - Instability: Large fluctuations indicate problems
    
    Visualization Features:
    
    Total Loss Panel:
        - Blue line: Training loss
        - Red line: Validation loss
        - Green dashed line: Best epoch marker
        - Shows combined MSE + LPIPS
        - Gap between train/val indicates overfitting
    
    MSE Loss Panel:
        - Pixel-level reconstruction error
        - Should decrease over training
        - Lower is better (more accurate pixels)
    
    LPIPS Loss Panel:
        - Perceptual similarity error
        - Should decrease over training
        - Lower is better (more perceptually similar)
    
    Learning Rate Panel:
        - Log scale (spans multiple orders of magnitude)
        - Shows when scheduler reduced LR
        - Reductions appear as downward steps
        - Should decrease when validation plateaus
    
    Interpreting the Plots:
    
    Good Training:
        - Smooth decreasing curves
        - Validation tracks training closely
        - LR reductions correspond to plateaus
        - Best epoch near end (not too early)
    
    Overfitting:
        - Training loss keeps decreasing
        - Validation loss increases or plateaus
        - Large gap between train and validation
        - Best epoch much earlier than end
    
    Underfitting:
        - Both losses high
        - Both losses still decreasing at end
        - Need more epochs or higher capacity
    
    Instability:
        - Large spikes in loss curves
        - Non-smooth training progress
        - May indicate LR too high or batch size too small
    
    Args:
        history (dict): Training history containing:
                       - train_loss, val_loss: Total loss lists
                       - train_mse, val_mse: MSE loss lists
                       - train_lpips, val_lpips: LPIPS loss lists
                       - learning_rates: Learning rate list
                       - best_epoch: Epoch with best validation loss
        
        save_path (Path): Directory to save plot
        
        exp_name (str): Experiment name for filename
    
    Returns:
        None: Saves PNG file to disk
    
    Example:
        >>> history = {
        ...     'train_loss': [0.5, 0.4, 0.3, ...],
        ...     'val_loss': [0.55, 0.45, 0.35, ...],
        ...     'train_mse': [0.3, 0.25, 0.2, ...],
        ...     'val_mse': [0.32, 0.27, 0.22, ...],
        ...     'train_lpips': [0.2, 0.15, 0.1, ...],
        ...     'val_lpips': [0.23, 0.18, 0.13, ...],
        ...     'learning_rates': [1e-4, 1e-4, 5e-5, ...],
        ...     'best_epoch': 15
        ... }
        >>> plot_history(history, Path('results/single'), 'vgg16_block1')
    """
    
    # Create figure with 2x2 subplot grid
    # figsize=(15, 10) gives each subplot good size
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Create epoch list for x-axis
    # Epochs start at 1, so range(1, N+1)
    epochs = range(1, len(history['train_loss']) + 1)
    
    # ==================== TOP LEFT: TOTAL LOSS ====================
    
    # Plot training loss (blue line, linewidth=2 for visibility)
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    
    # Plot validation loss (red line)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    
    # Mark best epoch with vertical green dashed line
    # Helps identify when model achieved best validation
    axes[0, 0].axvline(history['best_epoch'], color='g', linestyle='--', label=f'Best')
    
    # Set labels and title
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss (MSE + LPIPS)')
    
    # Add legend to identify lines
    axes[0, 0].legend()
    
    # Add grid for easier value reading
    # alpha=0.3 makes grid subtle
    axes[0, 0].grid(True, alpha=0.3)
    
    # ==================== TOP RIGHT: MSE LOSS ====================
    
    # Plot MSE component of loss
    axes[0, 1].plot(epochs, history['train_mse'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_mse'], 'r-', label='Val', linewidth=2)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].set_title('MSE Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # ==================== BOTTOM LEFT: LPIPS LOSS ====================
    
    # Plot LPIPS component of loss
    axes[1, 0].plot(epochs, history['train_lpips'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['val_lpips'], 'r-', label='Val', linewidth=2)
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('LPIPS Loss')
    axes[1, 0].set_title('LPIPS Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # ==================== BOTTOM RIGHT: LEARNING RATE ====================
    
    # Plot learning rate over epochs (green line)
    axes[1, 1].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate')
    
    # Use log scale for y-axis
    # Learning rates span multiple orders of magnitude (1e-4 to 1e-6)
    # Log scale makes reductions visible
    axes[1, 1].set_yscale('log')
    
    axes[1, 1].grid(True, alpha=0.3)
    
    # Adjust layout to prevent overlapping elements
    plt.tight_layout()
    
    # Save figure to disk
    # dpi=300: High resolution for publication quality
    # bbox_inches='tight': Remove extra whitespace
    plt.savefig(save_path / f'history_{exp_name}.png', dpi=300, bbox_inches='tight')
    
    # Close figure to free memory
    plt.close()


def load_and_evaluate(config, test_loader, device='cuda'):
    """
    Load trained model checkpoint and evaluate on test set.
    
    This convenience function combines checkpoint loading and evaluation
    into a single call. It automatically:
    1. Determines model type (single vs ensemble)
    2. Creates appropriate model instance
    3. Loads best checkpoint
    4. Evaluates on test set
    5. Returns evaluation metrics
    
    This is useful for scripts that need to evaluate multiple models
    or for quick testing of trained models.
    
    Model Type Detection:
        - Checks for 'architectures' key in config
        - If present: Creates EnsembleModel
        - If absent: Creates SingleArchModel
        - Automatically handles both cases
    
    Checkpoint Loading:
        - Looks for best checkpoint in appropriate directory
        - Filename: {exp_name}_best.pth
        - Contains model weights from best validation epoch
        - Warns if checkpoint not found
    
    Evaluation:
        - Uses evaluation module's evaluate_model function
        - Computes PSNR and SSIM on test set
        - Generates visualizations
        - Saves detailed metrics
        - Returns summary statistics
    
    Args:
        config (dict): Configuration dictionary containing:
                      - exp_name: Experiment identifier
                      - save_dir: Root directory for results
                      - architecture or architectures: Model type
                      - Other model-specific parameters
        
        test_loader (DataLoader): Test set data loader
        
        device (str, optional): Device for evaluation.
                               Default: 'cuda'
    
    Returns:
        dict or None: Evaluation metrics if successful, None if checkpoint not found.
                     Metrics dictionary contains:
                     - psnr_mean, psnr_std, psnr_min, psnr_max
                     - ssim_mean, ssim_std, ssim_min, ssim_max
                     - num_images, experiment, model_type
    
    Example:
        >>> config = {
        ...     'exp_name': 'vgg16_block1_transposed_conv',
        ...     'architecture': 'vgg16',
        ...     'vgg_block': 'block1',
        ...     'decoder_type': 'transposed_conv',
        ...     'save_dir': 'results'
        ... }
        >>> 
        >>> metrics = load_and_evaluate(config, test_loader, 'cuda')
        >>> 
        >>> if metrics:
        ...     print(f"PSNR: {metrics['psnr_mean']:.2f} dB")
        ...     print(f"SSIM: {metrics['ssim_mean']:.4f}")
        ... else:
        ...     print("Checkpoint not found")
    
    Note:
        This function imports models and evaluation modules internally
        to avoid circular dependencies. If these imports fail, the function
        will raise ImportError.
    """
    
    # Import required modules
    # Done inside function to avoid circular imports
    from models import SingleArchModel, EnsembleModel
    from evaluation import evaluate_model, load_best_checkpoint
    
    # Determine model type and create appropriate model instance
    if 'architectures' in config:
        # Config has 'architectures' key: Ensemble model
        model = EnsembleModel(config)
        model_type = 'ensemble'
    else:
        # Config has 'architecture' key: Single model
        model = SingleArchModel(config)
        model_type = 'single'
    
    # Construct path to best checkpoint
    # Structure: results/single/checkpoints_{exp_name}/{exp_name}_best.pth
    checkpoint_dir = Path(config['save_dir']) / model_type / f"checkpoints_{config['exp_name']}"
    checkpoint_path = checkpoint_dir / f"{config['exp_name']}_best.pth"
    
    # Check if checkpoint file exists
    if not checkpoint_path.exists():
        # Print warning and return None if not found
        # Caller should handle None return value
        print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
        return None
    
    # Load checkpoint into model
    # This restores trained weights and sets model to eval mode
    model, val_loss = load_best_checkpoint(checkpoint_path, model, device)
    
    # Print validation loss from checkpoint for context
    if val_loss:
        print(f"Loaded checkpoint with val_loss: {val_loss:.6f}")
    else:
        print("Loaded checkpoint")
    
    # Evaluate model on test set
    # Computes PSNR and SSIM metrics
    # Generates visualizations
    # Saves results to disk
    metrics = evaluate_model(model, test_loader, config, device)
    
    # Return evaluation metrics dictionary
    return metrics
