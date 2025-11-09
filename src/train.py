"""
Training utilities for decoder models.

Implements the optimization objective from Chapter 33.4:
    minimize: E_x[(x - g_psi(f_theta(x)))^2]
    
where:
    - f_theta(x) is the frozen encoder (feature extractor)
    - g_psi(z) is the trainable decoder
    - x is the original image
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import time
from pathlib import Path
from datetime import datetime


def train_one_epoch(encoder, decoder, dataloader, criterion, optimizer, device, epoch):
    """
    Train decoder for one epoch.
    
    Training procedure:
    1. Extract features z = f_theta(x) using frozen encoder
    2. Reconstruct x_hat = g_psi(z) using trainable decoder
    3. Compute loss L = ||x - x_hat||^2 (MSE)
    4. Backpropagate through decoder only (encoder frozen)
    5. Update decoder parameters psi
    
    Args:
        encoder: Frozen feature extractor f_theta(x)
        decoder: Trainable decoder g_psi(z)
        dataloader: Training data iterator
        criterion: Loss function (typically MSE)
        optimizer: Optimizer for decoder parameters
        device: Device to train on (cuda/mps/cpu)
        epoch: Current epoch number (for display)
        
    Returns:
        avg_loss: Average training loss for this epoch
    """
    # Set model modes
    # decoder.train() enables training mode (dropout active, batchnorm updates statistics)
    decoder.train()
    
    # encoder.eval() keeps encoder in evaluation mode (frozen, no gradient computation)
    # This is critical because we don't want to train the encoder
    encoder.eval()
    
    # Initialize loss accumulator
    total_loss = 0
    num_batches = len(dataloader)
    
    # Create progress bar for visual feedback during training
    # Shows current batch, loss, and estimated time remaining
    pbar = tqdm(dataloader, desc = f'Epoch {epoch}')
    
    # Iterate over batches of images
    for batch_idx, images in enumerate(pbar):
        # Move batch to device (GPU/CPU/MPS)
        # This is necessary because model and data must be on same device
        images = images.to(device)
        
        # Step 1: Extract features from encoder
        # We use torch.no_grad() because encoder is frozen
        # This saves memory and computation by not tracking gradients for encoder
        with torch.no_grad():
            # Forward pass through encoder: z = f_theta(x)
            # Input: images [B, 3, 224, 224]
            # Output: features [B, C, h, w] where C and h,w depend on layer
            # Example: ResNet34 layer3 gives [B, 256, 14, 14]
            features = encoder(images)
        
        # Step 2: Reconstruct images from features
        # This is where the trainable decoder comes in
        # x_hat = g_psi(z)
        # Input: features [B, C, h, w]
        # Output: reconstructed [B, 3, 224, 224] (same size as original)
        reconstructed = decoder(features)
        
        # Step 3: Compute reconstruction loss
        # We use Mean Squared Error (MSE) as our loss function
        # L = (1/N) * sum((x - x_hat)^2)
        # This measures pixel-wise difference between original and reconstruction
        loss = criterion(reconstructed, images)
        
        # Step 4: Backpropagation
        # Clear gradients from previous iteration
        # PyTorch accumulates gradients, so we need to zero them each iteration
        optimizer.zero_grad()
        
        # Compute gradients of loss with respect to decoder parameters
        # This calculates dL/dpsi for all trainable parameters in decoder
        # Gradients flow backward through decoder but stop at encoder (frozen)
        loss.backward()
        
        # Update decoder parameters using computed gradients
        # Parameter update: psi <- psi - learning_rate * gradient
        # Adam optimizer uses adaptive learning rates and momentum
        optimizer.step()
        
        # Track total loss for this epoch
        # .item() converts single-element tensor to Python float
        total_loss += loss.item()
        
        # Update progress bar with current loss
        # Shows real-time training progress
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate average loss across all batches
    # This gives us a single number representing training performance
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(encoder, decoder, dataloader, criterion, device, desc='Validating'):
    """
    Validate decoder on validation or test set.
    
    Same forward pass as training, but:
    - No gradient computation (faster, saves memory)
    - No parameter updates
    - Used to monitor overfitting (validation) or final evaluation (test)
    
    Args:
        encoder: Feature extractor
        decoder: Decoder to validate
        dataloader: Validation or test data
        criterion: Loss function
        device: Device
        desc: Progress bar description (default 'Validating')
        
    Returns:
        avg_loss: Average loss
    """
    # Set both models to evaluation mode
    # This disables dropout, makes batchnorm use running statistics (not batch statistics)
    decoder.eval()
    encoder.eval()
    
    # Initialize loss accumulator
    total_loss = 0
    num_batches = len(dataloader)
    
    # Iterate over validation batches
    # No gradient tracking needed (@torch.no_grad() decorator handles this)
    for images in tqdm(dataloader, desc=desc):
        # Move batch to device
        images = images.to(device)
        
        # Forward pass: extract features and reconstruct
        # z = f_theta(x)
        features = encoder(images)
        
        # x_hat = g_psi(z)
        reconstructed = decoder(features)
        
        # Compute loss (same as training, but no backprop)
        loss = criterion(reconstructed, images)
        total_loss += loss.item()
    
    # Return average loss
    avg_loss = total_loss / num_batches
    return avg_loss


def train_decoder(encoder, decoder, train_loader, val_loader, test_loader,
                  config, device, save_dir = 'results'):
    """
    Full training loop with validation and checkpointing.
    
    Implements training procedure from Chapter 33.4.3:
    - Optimize decoder g_psi to minimize reconstruction loss
    - Monitor validation loss to detect overfitting
    - Use learning rate scheduling (ReduceLROnPlateau)
    - Save best model based on validation performance
    - Evaluate on held-out test set at the end
    
    This is the main training function that orchestrates the entire training process.
    
    Args:
        encoder: Feature extractor f_theta(x) (frozen)
        decoder: Decoder g_psi(z) (trainable)
        train_loader: Training data loader (640 images, 80% of DIV2K_train_HR)
        val_loader: Validation data loader (160 images, 20% of DIV2K_train_HR)
        test_loader: Test data loader (100 images, DIV2K_test_HR)
        config: Training configuration dict with keys:
            - architecture: Model architecture name (e.g., 'resnet34')
            - layer_name: Layer to extract from (e.g., 'layer3')
            - decoder_type: Decoder type (e.g., 'simple', 'attention')
            - num_epochs: Number of training epochs
            - lr: Learning rate (typically 1e-3 to 1e-4)
            - weight_decay: L2 regularization (typically 1e-5)
        device: Device to train on (cuda/mps/cpu)
        save_dir: Base directory for results (default 'results')
        
    Returns:
        history: Dict containing training curves:
            - train_loss: List of training losses per epoch
            - val_loss: List of validation losses per epoch
            - test_loss: Final test loss (evaluated once at end)
            - lr: List of learning rates per epoch
    """
    # Create experiment name from configuration
    # This creates a unique identifier for this experiment
    # Example: "resnet34_layer3_attention"
    arch_name = config['architecture']
    layer_name = config['layer_name']
    decoder_type = config['decoder_type']
    experiment_name = f"{arch_name}_{layer_name}_{decoder_type}"
    
    # Setup architecture-specific save directory
    # Structure: results/resnet34/checkpoints/
    # This organizes experiments by architecture for easy comparison
    save_dir = Path(save_dir) / 'checkpoints'
    save_dir.mkdir(parents = True, exist_ok=True)
    
    # Define loss function: Mean Squared Error (MSE)
    # L = (1/N) * sum((x - x_hat)^2)
    # MSE is the standard loss for image reconstruction tasks
    # It penalizes large pixel-wise differences more than small ones
    criterion = nn.MSELoss()
    
    # Define optimizer: Adam (Adaptive Moment Estimation)
    # Adam is chosen because:
    # 1. Adaptive learning rates per parameter (good for different layer depths)
    # 2. Momentum helps escape local minima
    # 3. Works well for most deep learning tasks without tuning
    # 
    # Parameters:
    # - lr: Base learning rate (how big each update step is)
    # - weight_decay: L2 regularization (prevents overfitting by penalizing large weights)
    optimizer = torch.optim.Adam(
        decoder.parameters(),  # Only train decoder parameters (encoder frozen)
        lr = config.get('lr', 1e-3),  # Default 0.001 if not specified
        weight_decay = config.get('weight_decay', 1e-5)  # Small L2 penalty
    )
    
    # Define learning rate scheduler: ReduceLROnPlateau
    # This automatically reduces learning rate when validation loss stops improving
    # 
    # Why this helps:
    # - Early training: large LR makes fast progress
    # - Later training: smaller LR fine-tunes and prevents oscillation
    # - Automatic: no manual LR schedule needed
    #
    # Parameters:
    # - mode='min': we want to minimize validation loss
    # - factor=0.5: reduce LR by half when plateau detected
    # - patience=5: wait 5 epochs before reducing (avoid reducing too early)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode = 'min',
        factor = 0.5,
        patience = 5
    )
    
    # Move models to device (GPU/CPU/MPS)
    # This is essential - model and data must be on same device
    encoder.to(device)
    decoder.to(device)
    
    # Initialize training history
    # We'll track these metrics over time to analyze training
    history = {
        'train_loss': [],  # Training loss per epoch
        'val_loss': [],    # Validation loss per epoch
        'test_loss': None, # Test loss (evaluated once at end)
        'lr': []           # Learning rate per epoch
    }
    
    # Track best validation loss for model checkpointing
    # We save the model with lowest validation loss (best generalization)
    best_val_loss = float('inf')
    
    # Track training time
    start_time = time.time()
    
    # Print training configuration for documentation
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Experiment: {experiment_name}")
    print(f"  Epochs: {config.get('num_epochs', 20)}")
    print(f"  Learning Rate: {config.get('lr', 1e-3)}")
    print(f"  Batch Size: {train_loader.batch_size}")
    print(f"  Device: {device}")
    print(f"  Train Samples: {len(train_loader.dataset)}")
    print(f"  Val Samples: {len(val_loader.dataset)}")
    print(f"  Test Samples: {len(test_loader.dataset)}")
    print(f"  Train Batches: {len(train_loader)}")
    print(f"  Val Batches: {len(val_loader)}")
    print(f"  Test Batches: {len(test_loader)}")
    print(f"{'='*60}\n")
    
    # Main training loop - iterate over epochs
    # Each epoch = one complete pass through training data
    for epoch in range(1, config.get('num_epochs', 20) + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config.get('num_epochs', 20)}")
        print(f"{'='*60}")
        
        # Training phase
        # Train decoder on training set for one epoch
        train_loss = train_one_epoch(
            encoder, decoder, train_loader, 
            criterion, optimizer, device, epoch
        )
        
        # Store training loss in history
        history['train_loss'].append(train_loss)
        
        # Validation phase
        # Evaluate decoder on validation set (no training, just forward pass)
        # This tells us how well the model generalizes to unseen data within training set
        val_loss = validate(encoder, decoder, val_loader, criterion, device, 'Validating')
        
        # Store validation loss in history
        history['val_loss'].append(val_loss)
        
        # Track current learning rate
        # The scheduler may have changed it
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # Update learning rate based on validation loss
        # If validation loss hasn't improved for 'patience' epochs, reduce LR
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  LR:         {current_lr:.2e}")
        
        # Save best model (based on validation loss)
        # We save the model that performs best on validation set
        # This is the model we'll use for final test evaluation
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Create checkpoint filename with experiment name
            checkpoint_path = save_dir / f'{experiment_name}_best.pth'
            
            # Save model state and training info
            # We save everything needed to resume training or evaluate later
            torch.save({
                'epoch': epoch,  # Which epoch this was saved at
                'decoder_state_dict': decoder.state_dict(),  # Model weights
                'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
                'train_loss': train_loss,  # Training loss at this epoch
                'val_loss': val_loss,  # Validation loss at this epoch
                'config': config,  # Full configuration for reproducibility
                'experiment_name': experiment_name  # Experiment identifier
            }, checkpoint_path)
            
            print(f"  [SAVED] Best model: {checkpoint_path.name} (val_loss: {val_loss:.6f})")
        
        # Save periodic checkpoints (every 10 epochs)
        # These let us track training progress and resume if interrupted
        if epoch % 10 == 0:
            checkpoint_path = save_dir / f'{experiment_name}_epoch{epoch}.pth'
            
            # Save checkpoint with same format as best model
            torch.save({
                'epoch': epoch,
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
                'experiment_name': experiment_name
            }, checkpoint_path)
            
            print(f"  [SAVED] Checkpoint: {checkpoint_path.name}")
    
    # Training complete - evaluate on test set
    print(f"\n{'='*60}")
    print(f"Evaluating on Test Set (held-out 100 images)")
    print(f"{'='*60}")
    
    # Load best model for test evaluation
    checkpoint_path = save_dir / f'{experiment_name}_best.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    # Evaluate on test set
    test_loss = validate(encoder, decoder, test_loader, criterion, device, 'Testing')
    history['test_loss'] = test_loss
    
    print(f"\nTest Set Results:")
    print(f"  Test Loss: {test_loss:.6f}")
    
    # Calculate total training time
    elapsed_time = time.time() - start_time
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Experiment: {experiment_name}")
    print(f"  Total Time: {elapsed_time/60:.2f} minutes")
    print(f"  Best Val Loss: {best_val_loss:.6f}")
    print(f"  Test Loss: {test_loss:.6f}")
    print(f"  Saved to: {save_dir}")
    print(f"{'='*60}\n")
    
    # Return training history for plotting and analysis
    return history