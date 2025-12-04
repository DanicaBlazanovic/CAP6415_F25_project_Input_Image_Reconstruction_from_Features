"""
Image Reconstruction Model Training Script
==========================================

This script trains image reconstruction models on the DIV2K dataset using
feature extraction from various pre-trained architectures (ResNet, VGG, ViT, PVT)
combined with different decoder architectures.

Key Features:
- Single model training (ResNet34, VGG16, ViT, PVT)
- Ensemble model training (multiple fusion strategies)
- Batch training of all model configurations
- Full reproducibility with seed setting and version logging
- Perceptual loss (MSE + LPIPS) optimization
- Early stopping with validation monitoring
- Comprehensive training history and checkpoint saving

Training Process:
1. Load DIV2K dataset (640 train, 160 val, 100 test)
2. Extract features from frozen pre-trained encoder
3. Train decoder to reconstruct original images
4. Monitor validation loss for early stopping
5. Save best checkpoint and training history

Usage Examples:
    # Train single model
    python train.py --arch vgg16 --layer block1 --decoder transposed_conv
    
    # Train ensemble model
    python train.py --arch ensemble --fusion attention --decoder transposed_conv
    
    # Train all configurations (batch mode)
    python train.py --mode all

Author: Danica Blazanovic, Abbas Khan
Course: CAP6415 - Computer Vision, Fall 2025
Institution: Florida Atlantic University
"""

import sys
import os
from pathlib import Path
import random
import numpy as np
import torch
import argparse
import time
import json
import pandas as pd
from datetime import datetime

# ============================================================================
# PATH CONFIGURATION - REPRODUCIBILITY FIX
# ============================================================================

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent

# Determine the correct source directory based on script location
# Case 1: Script is in scripts/ folder (scripts/train.py, src/ is sibling)
# Case 2: Script is in src/ folder itself (src/train.py)
if SCRIPT_DIR.name == 'scripts':
    # Script is in scripts/, look for src/ as sibling
    SRC_DIR = SCRIPT_DIR.parent / 'src'
    if SRC_DIR.exists():
        sys.path.insert(0, str(SRC_DIR))
        print(f"[PATH] Added to Python path: {SRC_DIR}")
    else:
        print(f"[WARNING] src/ directory not found as sibling to scripts/")
        sys.path.insert(0, str(SCRIPT_DIR))
elif SCRIPT_DIR.name == 'src':
    # Script is already in src/, add current directory
    sys.path.insert(0, str(SCRIPT_DIR))
    print(f"[PATH] Added to Python path: {SCRIPT_DIR}")
else:
    # Fallback: add script's directory
    sys.path.insert(0, str(SCRIPT_DIR))
    print(f"[PATH] Added to Python path: {SCRIPT_DIR}")

# Now import project modules
try:
    from models import SingleArchModel, EnsembleModel
    from dataset import get_dataloaders
    from training import train_model
except ImportError as e:
    print(f"\n[ERROR] Failed to import required modules: {e}")
    print(f"[ERROR] Script directory: {SCRIPT_DIR}")
    print(f"[ERROR] Python path: {sys.path[:3]}")
    print(f"\n[HELP] Make sure the following files exist in the src/ folder:")
    print("  - models.py")
    print("  - dataset.py")
    print("  - trainingl.py")
    sys.exit(1)


# ============================================================================
# REPRODUCIBILITY CONFIGURATION
# ============================================================================
# Set random seeds for reproducibility across different runs
# This ensures that model initialization, data shuffling, and stochastic
# operations produce identical results given the same input
RANDOM_SEED = 42

def set_random_seeds(seed=RANDOM_SEED):
    """
    Set random seeds for all libraries to ensure reproducibility.
    
    Training neural networks involves many sources of randomness:
    - Model parameter initialization
    - Data shuffling during training
    - Dropout operations
    - Data augmentation (random crops, flips)
    - GPU operations (some CUDA operations are non-deterministic by default)
    
    This function sets seeds for all these sources to enable exact reproduction
    of training results across different runs.
    
    Args:
        seed (int): Random seed value (default: 42)
    
    Note:
        Setting deterministic operations may reduce training speed by 10-20%
        but is essential for reproducibility in research settings.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        # Make CuDNN deterministic (reduces performance but ensures reproducibility)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"[REPRODUCIBILITY] Random seed set to: {seed}")
    print(f"[REPRODUCIBILITY] CuDNN deterministic mode: {'ON' if torch.cuda.is_available() else 'N/A (no CUDA)'}")


def log_environment_info():
    """
    Log software versions and hardware configuration for reproducibility.
    
    Records critical information needed to reproduce training results:
    - Software versions (Python, PyTorch, NumPy)
    - Hardware (GPU model, CUDA version)
    - Training configuration (seed, batch size, etc.)
    
    This information is saved with training results and should be reported
    in any publications or documentation.
    
    Returns:
        dict: Dictionary containing version and hardware information
    """
    env_info = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
        'numpy_version': np.__version__,
        'cuda_available': torch.cuda.is_available(),
        'random_seed': RANDOM_SEED
    }
    
    if torch.cuda.is_available():
        env_info['cuda_version'] = torch.version.cuda
        env_info['cudnn_version'] = torch.backends.cudnn.version()
        env_info['gpu_name'] = torch.cuda.get_device_name(0)
        env_info['gpu_count'] = torch.cuda.device_count()
        env_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print("\n" + "="*80)
    print("ENVIRONMENT INFORMATION (for reproducibility)")
    print("="*80)
    for key, value in env_info.items():
        print(f"{key}: {value}")
    print("="*80 + "\n")
    
    return env_info


# ============================================================================
# BASE CONFIGURATION
# ============================================================================
# These hyperparameters control the training process
# Carefully tuned based on baseline experiments and computational constraints

BASE_CONFIG = {
    'output_size': 224,         # Target image size (224x224 pixels)
                                 # Matches ImageNet pre-training resolution
    
    'epochs': 30,                # Maximum training epochs
                                 # 30 epochs sufficient for convergence based on validation
                                 # Can increase to 100 for potential marginal improvements
                                 # Early stopping (patience=15) will halt training if no improvement
    
    'lr': 0.0001,                # Learning rate (Adam optimizer)
                                 # 1e-4 is standard for fine-tuning tasks
                                 # Lower than typical (1e-3) since we're training decoders only
                                 # Scheduler reduces this by 0.5× when validation stalls
    
    'batch_size': 4,             # Images per training batch
                                 # Small batch size due to:
                                 # - Large feature maps (especially VGG16 block1: 112×112×64)
                                 # - Memory constraints (48GB GPU)
                                 # - Stable gradients for perceptual loss
                                 # Effective batch size = 4 × gradient accumulation (if used)
    
    'num_workers': 2,            # Parallel data loading workers
                                 # Balance between I/O speed and CPU overhead
                                 # 2 workers sufficient for our dataset size
                                 # More workers can cause slowdown on some systems
    
    'patience': 15,              # Early stopping patience (epochs)
                                 # Stop training if validation loss doesn't improve for 15 epochs
                                 # Prevents overfitting and saves computation time
                                 # Value chosen based on typical convergence patterns
    
    'mse_weight': 0.5,           # Weight for MSE loss component
                                 # Combined loss = 0.5×MSE + 0.5×LPIPS
                                 # Equal weighting balances pixel accuracy and perceptual quality
                                 # MSE: Ensures numerical fidelity to ground truth
    
    'lpips_weight': 0.5,         # Weight for LPIPS (perceptual) loss component
                                 # LPIPS: Ensures perceptual similarity using deep features
                                 # Equal weighting found optimal in Run 1 experiments
                                 # Higher LPIPS weight → more perceptually pleasing but lower PSNR
    
    'limit': None,               # Limit number of training images (None = use all 640)
                                 # Useful for quick debugging: set to 100 for fast test runs
                                 # None = full dataset (640 train, 160 val, 100 test)
    
    'save_dir': 'results'        # Root directory for all outputs
                                 # Structure: results/single/ or results/ensemble/
                                 # Each experiment gets subdirectories for checkpoints, history, etc.
}

# ============================================================================
# DECODER VARIANTS
# ============================================================================
# Four decoder architectures tested in this study
# Each has different inductive biases and computational characteristics

DECODER_VARIANTS = [
    'frequency_aware',      # Explicit low/high frequency decomposition with spatial-channel attention
                            # Most complex, highest parameter count
                            # Hypothesis: Explicit frequency modeling helps reconstruction
    
    'wavelet',              # Multi-resolution wavelet sub-band prediction
                            # Medium complexity, uses wavelet transform
                            # Hypothesis: Wavelet decomposition matches network's hierarchical features
    
    'transposed_conv',      # Simple stride-2 transposed convolutions
                            # Simplest architecture, fewest parameters
                            # SURPRISING FINDING: Often performs best! (see Top 10 models)
                            # Supports "simplicity is better" hypothesis
    
    'attention'             # Transformer blocks with self-attention mechanisms
                            # High complexity, global context aggregation
                            # Hypothesis: Attention helps resolve spatial ambiguities
]

# ============================================================================
# SINGLE ARCHITECTURE CONFIGURATIONS
# ============================================================================
# Layer selections based on baseline experiments showing shallow layers
# provide best reconstruction due to higher spatial resolution

SINGLE_ARCHITECTURES = {
    # ResNet34: Residual network with skip connections
    # Testing early (layer1) and mid (layer2) layers
    # Deeper layers (layer3, layer4) showed poor reconstruction in baselines
    'resnet34_layer1': {
        'architecture': 'resnet34',
        'resnet_layer': 'layer1',       # 64 channels, 56×56 spatial, 3,136 locations
        'decoder_type': 'frequency_aware'  # Default (overridden during experiments)
    },
    'resnet34_layer2': {
        'architecture': 'resnet34',
        'resnet_layer': 'layer2',       # 128 channels, 28×28 spatial, 784 locations
        'decoder_type': 'frequency_aware'
    },
    
    # VGG16: Simple sequential architecture with aggressive pooling
    # Block1 has highest spatial resolution → best reconstruction (14.45 dB PSNR in baselines)
    # This is our best single-architecture baseline
    'vgg16_block1': {
        'architecture': 'vgg16',
        'vgg_block': 'block1',          # 64 channels, 112×112 spatial, 12,544 locations (BEST!)
        'decoder_type': 'frequency_aware'  # Highest resolution = best reconstruction potential
    },
    'vgg16_block3': {
        'architecture': 'vgg16',
        'vgg_block': 'block3',          # 256 channels, 28×28 spatial, 784 locations
        'decoder_type': 'frequency_aware'
    },
    
    # Vision Transformer (ViT): Patch-based self-attention architecture
    # Maintains constant 14×14 spatial resolution through all blocks
    # Global attention provides different information encoding than CNNs
    'vit_small_block1': {
        'architecture': 'vit_small_patch16_224',
        'vit_block': 'block1',          # 384 channels, 14×14 tokens, 196 spatial locations
        'decoder_type': 'frequency_aware'  # Early block preserves more spatial info
    },
    'vit_small_block3': {
        'architecture': 'vit_small_patch16_224',
        'vit_block': 'block3',          # 384 channels, 14×14 tokens, 196 spatial locations
        'decoder_type': 'frequency_aware'  # Deeper block has more semantic abstraction
    },
    
    # Pyramid Vision Transformer (PVT): Hierarchical vision transformer
    # Combines CNN-style spatial pyramids with transformer attention
    # Progressive spatial downsampling like CNNs
    'pvt_v2_b2_stage1': {
        'architecture': 'pvt_v2_b2',
        'pvt_stage': 'stage1',          # 64 channels, 56×56 spatial, 3,136 locations
        'decoder_type': 'frequency_aware'  # Similar resolution to ResNet34 layer1
    },
    'pvt_v2_b2_stage2': {
        'architecture': 'pvt_v2_b2',
        'pvt_stage': 'stage2',          # 128 channels, 28×28 spatial, 784 locations
        'decoder_type': 'frequency_aware'  # Similar resolution to ResNet34 layer2
    }
}

# ============================================================================
# ENSEMBLE CONFIGURATIONS
# ============================================================================
# Multi-architecture ensembles that fuse features from multiple networks
# All use shallow layers (layer1/block1/stage1) based on baseline findings
# that spatial resolution matters more than semantic depth

ENSEMBLE_CONFIGS = {
    # Attention-based fusion: Learned attention weights for each architecture
    # Most sophisticated fusion: learns which architecture to trust for each spatial location
    'ensemble_all_attention': {
        'architectures': ['resnet34', 'vgg16', 'vit_small_patch16_224', 'pvt_v2_b2'],
        'resnet_layer': 'layer1',   # 64 ch, 56×56
        'vgg_block': 'block1',      # 64 ch, 112×112 (provides highest resolution input)
        'vit_block': 'block1',      # 384 ch, 14×14
        'pvt_stage': 'stage1',      # 64 ch, 56×56
        'fusion_strategy': 'attention',
        'fusion_channels': 256,     # Intermediate dimension after fusion
                                    # All features projected to 256-d before fusion
                                    # Balances expressiveness vs. computational cost
        'decoder_type': 'frequency_aware'  # Will be overridden per experiment
    },
    
    # Concatenation fusion: Simple channel-wise concatenation
    # Simplest fusion: just stack all features and let decoder learn combinations
    # Surprisingly effective despite simplicity
    'ensemble_all_concat': {
        'architectures': ['resnet34', 'vgg16', 'vit_small_patch16_224', 'pvt_v2_b2'],
        'resnet_layer': 'layer1',
        'vgg_block': 'block1',
        'vit_block': 'block1',
        'pvt_stage': 'stage1',
        'fusion_strategy': 'concat',
        'fusion_channels': 256,
        'decoder_type': 'frequency_aware'
    },
    
    # Weighted fusion: Learnable scalar weights for each architecture
    # Middle ground: each architecture gets a single learned weight
    # More constrained than attention but more flexible than concat
    'ensemble_all_weighted': {
        'architectures': ['resnet34', 'vgg16', 'vit_small_patch16_224', 'pvt_v2_b2'],
        'resnet_layer': 'layer1',
        'vgg_block': 'block1',
        'vit_block': 'block1',
        'pvt_stage': 'stage1',
        'fusion_strategy': 'weighted',
        'fusion_channels': 256,
        'decoder_type': 'frequency_aware'
    }
}


# ============================================================================
# SINGLE TRAINING FUNCTION
# ============================================================================

def train_single_experiment(architecture, layer, decoder, fusion=None):
    """
    Train a single model configuration on the DIV2K dataset.
    
    This function implements the complete training pipeline:
    1. Load DIV2K dataset (640 train, 160 val splits)
    2. Initialize model (frozen encoder + trainable decoder)
    3. Train with MSE + LPIPS loss for specified epochs
    4. Monitor validation loss for early stopping
    5. Save best checkpoint and training history
    
    The training process uses:
    - Adam optimizer with ReduceLROnPlateau scheduling
    - Combined MSE (pixel accuracy) + LPIPS (perceptual quality) loss
    - Early stopping with patience=15 to prevent overfitting
    - Checkpoint saving for best validation loss
    
    Args:
        architecture (str): Architecture name. Options:
                          - 'resnet34': ResNet-34 with residual connections
                          - 'vgg16': VGG-16 sequential CNN
                          - 'vit_small_patch16_224': Vision Transformer
                          - 'pvt_v2_b2': Pyramid Vision Transformer
                          - 'ensemble': Multi-architecture ensemble
        
        layer (str): Layer/block/stage to extract features from. Options:
                    - ResNet: 'layer1', 'layer2'
                    - VGG: 'block1', 'block3'
                    - ViT: 'block1', 'block3'
                    - PVT: 'stage1', 'stage2'
        
        decoder (str): Decoder architecture type. Options:
                      - 'frequency_aware': Explicit frequency decomposition
                      - 'wavelet': Wavelet-based multi-resolution
                      - 'transposed_conv': Simple transposed convolutions (often best!)
                      - 'attention': Transformer-based attention decoder
        
        fusion (str, optional): Fusion strategy for ensemble. Required if
                               architecture='ensemble'. Options:
                               - 'attention': Attention-based fusion
                               - 'concat': Concatenation fusion
                               - 'weighted': Weighted scalar fusion
    
    Returns:
        None: Model checkpoint and training history saved to disk
    
    Outputs:
        Saved to results/single/ or results/ensemble/:
        - checkpoints_{exp_name}/{exp_name}_best.pth: Best model weights
        - training_history_{exp_name}.json: Loss curves and metrics
        - training_config_{exp_name}.json: Complete configuration
    
    Example:
        >>> train_single_experiment('vgg16', 'block1', 'transposed_conv')
        >>> train_single_experiment('ensemble', None, 'simple', fusion='attention')
    
    Training Time:
        - VGG16 block1: ~2-3 hours (largest feature maps)
        - ResNet34 layer1: ~1-2 hours
        - ViT block1: ~1-2 hours
        - Ensemble: ~3-4 hours (4 encoders + fusion)
    """
    
    print("\n" + "="*80)
    print("SINGLE TRAINING MODE")
    print("="*80)
    print(f"Architecture: {architecture}")
    print(f"Layer: {layer}")
    print(f"Decoder: {decoder}")
    if fusion:
        print(f"Fusion: {fusion}")
    print("="*80 + "\n")
    
    # Detect and report compute device
    device = get_device()
    
    # Load DIV2K dataset with train/val/test splits
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BASE_CONFIG['batch_size'],
        num_workers=BASE_CONFIG['num_workers'],
        limit=BASE_CONFIG['limit']
    )
    print(f"Train: {len(train_loader.dataset)} images | "
          f"Val: {len(val_loader.dataset)} images | "
          f"Test: {len(test_loader.dataset)} images")
    print(f"Batch size: {BASE_CONFIG['batch_size']} | "
          f"Batches per epoch: {len(train_loader)}")
    
    # Build experiment configuration
    if architecture == 'ensemble':
        # Ensemble model requires fusion strategy
        if fusion is None:
            print("[ERROR] Fusion strategy required for ensemble (attention/concat/weighted)")
            print("Example: --arch ensemble --fusion attention --decoder transposed_conv")
            return
        
        # Find matching ensemble configuration
        ensemble_key = f'ensemble_all_{fusion}'
        if ensemble_key not in ENSEMBLE_CONFIGS:
            print(f"[ERROR] Unknown ensemble configuration: {ensemble_key}")
            print(f"Available: {list(ENSEMBLE_CONFIGS.keys())}")
            return
        
        # Merge base config with ensemble config
        exp_config = {**BASE_CONFIG, **ENSEMBLE_CONFIGS[ensemble_key]}
        exp_config['decoder_type'] = decoder
        exp_config['exp_name'] = f"{ensemble_key}_{decoder}"
        
        print(f"\n{'#'*80}")
        print(f"EXPERIMENT: {exp_config['exp_name']}")
        print(f"Architectures: {', '.join(exp_config['architectures'])}")
        print(f"Fusion strategy: {fusion}")
        print(f"Decoder: {decoder}")
        print(f"{'#'*80}\n")
        
        # Create ensemble model (multiple frozen encoders + fusion + decoder)
        model = EnsembleModel(exp_config)
        
    else:
        # Single architecture model
        # Determine which parameter name to use for the layer specification
        # Different architectures use different naming conventions
        if architecture == 'resnet34':
            layer_param = 'resnet_layer'
        elif architecture == 'vgg16':
            layer_param = 'vgg_block'
        elif architecture == 'vit_small_patch16_224':
            layer_param = 'vit_block'
        elif architecture == 'pvt_v2_b2':
            layer_param = 'pvt_stage'
        else:
            print(f"[ERROR] Unknown architecture: {architecture}")
            print("Valid options: resnet34, vgg16, vit_small_patch16_224, pvt_v2_b2, ensemble")
            return
        
        # Build configuration dictionary
        exp_config = BASE_CONFIG.copy()
        exp_config['architecture'] = architecture
        exp_config[layer_param] = layer
        exp_config['decoder_type'] = decoder
        exp_config['exp_name'] = f"{architecture}_{layer}_{decoder}"
        
        print(f"\n{'#'*80}")
        print(f"EXPERIMENT: {exp_config['exp_name']}")
        print(f"Architecture: {architecture}")
        print(f"Layer: {layer}")
        print(f"Decoder: {decoder}")
        print(f"{'#'*80}\n")
        
        # Create single architecture model (frozen encoder + trainable decoder)
        model = SingleArchModel(exp_config)
    
    # Display model statistics
    print_model_info(model)
    
    # Start training timer
    train_start = time.time()
    
    # Main training loop
    # train_model handles:
    # - Forward/backward passes
    # - Loss computation (MSE + LPIPS)
    # - Optimizer updates
    # - Learning rate scheduling
    # - Early stopping
    # - Checkpoint saving
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"Max epochs: {exp_config['epochs']}")
    print(f"Learning rate: {exp_config['lr']}")
    print(f"Early stopping patience: {exp_config['patience']} epochs")
    print(f"Loss: {exp_config['mse_weight']}×MSE + {exp_config['lpips_weight']}×LPIPS")
    print("="*80 + "\n")
    
    history = train_model(model, train_loader, val_loader, exp_config, device)
    
    # Calculate training time
    train_time = time.time() - train_start
    
    # Print training summary
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE: {exp_config['exp_name']}")
    print(f"{'='*80}")
    print(f"Training time: {train_time/60:.2f} minutes ({train_time/3600:.2f} hours)")
    print(f"Best epoch: {history['best_epoch']}/{exp_config['epochs']}")
    print(f"Best validation loss: {history['best_val_loss']:.6f}")
    print(f"Final training loss: {history['train_losses'][-1]:.6f}")
    
    # Determine checkpoint location based on model type
    model_type = 'ensemble' if architecture == 'ensemble' else 'single'
    checkpoint_path = f"results/{model_type}/checkpoints_{exp_config['exp_name']}/{exp_config['exp_name']}_best.pth"
    print(f"Best checkpoint saved: {checkpoint_path}")
    print(f"Training history saved: results/{model_type}/training_history_{exp_config['exp_name']}.json")
    print(f"{'='*80}\n")
    
    # Memory cleanup
    # Important for batch training to prevent OOM errors
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# BATCH TRAINING FUNCTION
# ============================================================================

def train_all_experiments():
    """
    Train all model configurations in batch mode.
    
    This function implements comprehensive batch training of all model
    configurations defined in SINGLE_ARCHITECTURES and ENSEMBLE_CONFIGS,
    combined with all decoder variants.
    
    Training is organized in two phases:
    
    Phase 1: Single Architecture Models
    - 8 architectures × 4 decoders = 32 models
    - Includes: ResNet34 (2 layers), VGG16 (2 blocks), ViT (2 blocks), PVT (2 stages)
    - Each with 4 decoder types: frequency_aware, wavelet, transposed_conv, attention
    
    Phase 2: Ensemble Models
    - 3 fusion strategies × 4 decoders = 12 models
    - Fusion strategies: attention, concat, weighted
    - Each with 4 decoder types
    
    Total: 44 model training runs
    
    For each model:
    1. Initialize architecture and decoder
    2. Train with MSE + LPIPS loss
    3. Monitor validation loss for early stopping
    4. Save best checkpoint
    5. Record training metrics
    6. Cleanup memory
    
    After all training:
    - Generate training summary CSV with all metrics
    - Save complete training results as JSON
    - Display top 10 models by validation loss
    
    Returns:
        None: All results saved to results/training_summary/
    
    Outputs:
        - training_results_{timestamp}.json: Complete training log
        - training_results_{timestamp}.csv: Metrics table
        - Individual checkpoints in results/single/ and results/ensemble/
    
    Estimated Time:
        - Single models: ~1.5 hours each × 32 = ~48 hours
        - Ensemble models: ~3 hours each × 12 = ~36 hours
        - Total: ~84 hours (~3.5 days) on NVIDIA RTX 6000 Ada
    
    Note:
        This function is designed for unattended batch execution.
        Consider using screen/tmux to keep process running if SSH session disconnects.
        Monitor GPU temperature and utilization to ensure stable operation.
    """
    
    print("\n" + "="*80)
    print("BATCH TRAINING MODE")
    print("="*80)
    print(f"Single architectures: {len(SINGLE_ARCHITECTURES)}")
    print(f"Decoder variants: {len(DECODER_VARIANTS)}")
    print(f"Total single models: {len(SINGLE_ARCHITECTURES)} × {len(DECODER_VARIANTS)} = {len(SINGLE_ARCHITECTURES) * len(DECODER_VARIANTS)}")
    print(f"\nEnsemble configurations: {len(ENSEMBLE_CONFIGS)}")
    print(f"Total ensemble models: {len(ENSEMBLE_CONFIGS)} × {len(DECODER_VARIANTS)} = {len(ENSEMBLE_CONFIGS) * len(DECODER_VARIANTS)}")
    print(f"\nGrand total: {(len(SINGLE_ARCHITECTURES) + len(ENSEMBLE_CONFIGS)) * len(DECODER_VARIANTS)} training runs")
    print("="*80 + "\n")
    
    # Start overall timer
    total_start = time.time()
    
    # Detect compute device
    device = get_device()
    
    # Load dataset once (reused for all training runs)
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BASE_CONFIG['batch_size'],
        num_workers=BASE_CONFIG['num_workers'],
        limit=BASE_CONFIG['limit']
    )
    print(f"Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}")
    
    # Initialize results storage
    training_results = []
    
    # ========================================================================
    # PHASE 1: TRAIN SINGLE ARCHITECTURE MODELS
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 1: SINGLE ARCHITECTURE MODELS")
    print("="*80)
    print(f"Training {len(SINGLE_ARCHITECTURES) * len(DECODER_VARIANTS)} models...")
    print("="*80 + "\n")
    
    phase1_success = 0
    phase1_failed = 0
    
    for arch_name, arch_config in SINGLE_ARCHITECTURES.items():
        for decoder in DECODER_VARIANTS:
            # Build experiment configuration
            exp_config = {**BASE_CONFIG, **arch_config}
            exp_config['decoder_type'] = decoder
            exp_config['exp_name'] = f"{arch_name}_{decoder}"
            
            print(f"\n{'#'*80}")
            print(f"TRAINING: {exp_config['exp_name']}")
            print(f"  Architecture: {arch_config['architecture']}")
            print(f"  Layer: {list(arch_config.values())[1]}")  # Extract layer name
            print(f"  Decoder: {decoder}")
            print(f"  Progress: {phase1_success + phase1_failed + 1}/{len(SINGLE_ARCHITECTURES) * len(DECODER_VARIANTS)}")
            print(f"{'#'*80}\n")
            
            try:
                # Initialize model
                model = SingleArchModel(exp_config)
                print_model_info(model)
                
                # Train model
                train_start = time.time()
                history = train_model(model, train_loader, val_loader, exp_config, device)
                train_time = time.time() - train_start
                
                # Record training results
                # Extract layer name (second value in architecture config dict)
                result = {
                    'exp_name': exp_config['exp_name'],
                    'architecture': arch_config['architecture'],
                    'layer': list(arch_config.values())[1],
                    'decoder': decoder,
                    'train_time_minutes': train_time / 60,
                    'train_time_hours': train_time / 3600,
                    'best_val_loss': history['best_val_loss'],
                    'best_epoch': history['best_epoch'],
                    'total_epochs': exp_config['epochs'],
                    'stopped_early': history['best_epoch'] < exp_config['epochs']
                }
                training_results.append(result)
                
                print(f"\n[SUCCESS] {exp_config['exp_name']}")
                print(f"  Val Loss: {history['best_val_loss']:.6f}")
                print(f"  Best Epoch: {history['best_epoch']}/{exp_config['epochs']}")
                print(f"  Time: {train_time/60:.2f} minutes")
                phase1_success += 1
                
                # Memory cleanup to prevent OOM in batch mode
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                # Catch and log errors without stopping batch process
                print(f"\n[ERROR] {exp_config['exp_name']}: {str(e)}\n")
                training_results.append({
                    'exp_name': exp_config['exp_name'],
                    'architecture': arch_config['architecture'],
                    'layer': list(arch_config.values())[1],
                    'decoder': decoder,
                    'error': str(e)
                })
                phase1_failed += 1
    
    print(f"\n[PHASE 1 COMPLETE]")
    print(f"  Successful: {phase1_success}/{len(SINGLE_ARCHITECTURES) * len(DECODER_VARIANTS)}")
    print(f"  Failed: {phase1_failed}/{len(SINGLE_ARCHITECTURES) * len(DECODER_VARIANTS)}")
    print(f"  Success rate: {phase1_success/(phase1_success+phase1_failed)*100:.1f}%")
    
    # ========================================================================
    # PHASE 2: TRAIN ENSEMBLE MODELS
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 2: ENSEMBLE MODELS")
    print("="*80)
    print(f"Training {len(ENSEMBLE_CONFIGS) * len(DECODER_VARIANTS)} models...")
    print("="*80 + "\n")
    
    phase2_success = 0
    phase2_failed = 0
    
    for ensemble_name, ensemble_config in ENSEMBLE_CONFIGS.items():
        for decoder in DECODER_VARIANTS:
            # Build experiment configuration
            exp_config = {**BASE_CONFIG, **ensemble_config}
            exp_config['decoder_type'] = decoder
            exp_config['exp_name'] = f"{ensemble_name}_{decoder}"
            
            print(f"\n{'#'*80}")
            print(f"TRAINING: {exp_config['exp_name']}")
            print(f"  Architectures: {', '.join(ensemble_config['architectures'])}")
            print(f"  Fusion: {ensemble_config['fusion_strategy']}")
            print(f"  Decoder: {decoder}")
            print(f"  Progress: {phase2_success + phase2_failed + 1}/{len(ENSEMBLE_CONFIGS) * len(DECODER_VARIANTS)}")
            print(f"{'#'*80}\n")
            
            try:
                # Initialize ensemble model
                model = EnsembleModel(exp_config)
                print_model_info(model)
                
                # Train model
                train_start = time.time()
                history = train_model(model, train_loader, val_loader, exp_config, device)
                train_time = time.time() - train_start
                
                # Record training results
                result = {
                    'exp_name': exp_config['exp_name'],
                    'architectures': ', '.join(ensemble_config['architectures']),
                    'fusion': ensemble_config['fusion_strategy'],
                    'decoder': decoder,
                    'train_time_minutes': train_time / 60,
                    'train_time_hours': train_time / 3600,
                    'best_val_loss': history['best_val_loss'],
                    'best_epoch': history['best_epoch'],
                    'total_epochs': exp_config['epochs'],
                    'stopped_early': history['best_epoch'] < exp_config['epochs']
                }
                training_results.append(result)
                
                print(f"\n[SUCCESS] {exp_config['exp_name']}")
                print(f"  Val Loss: {history['best_val_loss']:.6f}")
                print(f"  Best Epoch: {history['best_epoch']}/{exp_config['epochs']}")
                print(f"  Time: {train_time/60:.2f} minutes")
                phase2_success += 1
                
                # Memory cleanup
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                # Catch and log errors without stopping batch process
                print(f"\n[ERROR] {exp_config['exp_name']}: {str(e)}\n")
                training_results.append({
                    'exp_name': exp_config['exp_name'],
                    'architectures': ', '.join(ensemble_config['architectures']),
                    'fusion': ensemble_config['fusion_strategy'],
                    'decoder': decoder,
                    'error': str(e)
                })
                phase2_failed += 1
    
    print(f"\n[PHASE 2 COMPLETE]")
    print(f"  Successful: {phase2_success}/{len(ENSEMBLE_CONFIGS) * len(DECODER_VARIANTS)}")
    print(f"  Failed: {phase2_failed}/{len(ENSEMBLE_CONFIGS) * len(DECODER_VARIANTS)}")
    print(f"  Success rate: {phase2_success/(phase2_success+phase2_failed)*100:.1f}%")
    
    # ========================================================================
    # SAVE TRAINING SUMMARY
    # ========================================================================
    total_time = time.time() - total_start
    
    print("\n" + "="*80)
    print("GENERATING TRAINING SUMMARY")
    print("="*80)
    
    # Create output directory
    report_dir = Path('results') / 'training_summary'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for file naming
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Save complete training results as JSON
    with open(report_dir / f'training_results_{timestamp}.json', 'w') as f:
        json.dump(training_results, f, indent=2)
    print(f"[SAVED] Complete training log: training_results_{timestamp}.json")
    
    # Create DataFrame from successful training runs
    df_results = pd.DataFrame([r for r in training_results if 'error' not in r])
    
    if not df_results.empty:
        # Sort by validation loss (best models first)
        df_results = df_results.sort_values('best_val_loss', ascending=True)
        
        # Save as CSV
        df_results.to_csv(report_dir / f'training_results_{timestamp}.csv', index=False)
        print(f"[SAVED] Training metrics CSV: training_results_{timestamp}.csv")
        
        # Display top 10 models
        print("\n" + "="*80)
        print("TOP 10 MODELS BY VALIDATION LOSS")
        print("="*80)
        display_cols = ['exp_name', 'best_val_loss', 'best_epoch', 'train_time_minutes']
        print(df_results[display_cols].head(10).to_string(index=False))
    else:
        print("\n[WARNING] No successful training runs to summarize")
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"BATCH TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/3600/24:.2f} days)")
    print(f"Total experiments: {len(training_results)}")
    print(f"Successful: {phase1_success + phase2_success}")
    print(f"Failed: {phase1_failed + phase2_failed}")
    print(f"Success rate: {(phase1_success + phase2_success)/len(training_results)*100:.1f}%")
    print(f"\nResults saved in: {report_dir}")
    print(f"{'='*80}\n")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_device():
    """
    Detect and return the best available compute device.
    
    Priority order:
    1. CUDA (NVIDIA GPU) - fastest for deep learning, supports large models
    2. MPS (Apple Silicon) - fast on M1/M2/M3 Macs, good for development
    3. CPU - slowest fallback, avoid for training if possible
    
    For training, GPU is strongly recommended:
    - CPU training: ~100× slower than GPU
    - Single epoch on CPU: ~10-20 hours
    - Single epoch on GPU: ~5-10 minutes
    
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    
    Side Effects:
        Prints device information including:
        - Device type
        - GPU name (if available)
        - GPU memory (if available)
        - Warning if using CPU
    """
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[DEVICE] CUDA GPU: {gpu_name}")
        print(f"[DEVICE] GPU Memory: {gpu_memory:.1f} GB")
        print(f"[DEVICE] CUDA Version: {torch.version.cuda}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("[DEVICE] Apple Silicon MPS")
        print("[DEVICE] Note: MPS may be slower than CUDA for some operations")
        print("[DEVICE] Recommendation: Use CUDA GPU for batch training")
    else:
        device = 'cpu'
        print("[DEVICE] CPU")
        print("[DEVICE] WARNING: Training on CPU will be VERY slow (~100× slower than GPU)")
        print("[DEVICE] Recommendation: Use a GPU for training")
        print("[DEVICE] Estimated time per epoch on CPU: 10-20 hours")
        print("[DEVICE] Estimated time per epoch on GPU: 5-10 minutes")
    
    return device


def print_model_info(model):
    """
    Print detailed model parameter information.
    
    Displays:
    - Total parameters (frozen + trainable)
    - Trainable parameters (decoder weights)
    - Frozen parameters (encoder weights)
    
    This information is important for:
    - Understanding model complexity
    - Comparing different decoder architectures
    - Debugging parameter freezing issues
    - Estimating memory requirements
    
    Args:
        model: PyTorch model (SingleArchModel or EnsembleModel)
    
    Side Effects:
        Prints parameter counts to console
    
    Note:
        Frozen encoders typically have 5M-80M parameters depending on architecture
        Trainable decoders typically have 0.2M-35M parameters depending on type
        Total memory usage ≈ (total_params × 4 bytes) for FP32
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print("\n" + "-"*80)
    print("MODEL INFORMATION")
    print("-"*80)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"Frozen parameters:    {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    print(f"Estimated memory:     {total_params * 4 / 1e9:.2f} GB (FP32)")
    print("-"*80 + "\n")


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_arguments():
    """
    Parse and validate command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments with attributes:
                           - mode: 'all' or 'single'
                           - arch: Architecture name
                           - layer: Layer/block/stage name
                           - decoder: Decoder type
                           - fusion: Fusion strategy (ensemble only)
    
    Examples:
        $ python train.py --mode all
        $ python train.py --arch vgg16 --layer block1 --decoder transposed_conv
        $ python train.py --arch ensemble --fusion attention --decoder transposed_conv
    """
    parser = argparse.ArgumentParser(
        description='Train image reconstruction models on DIV2K dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single model
  python train.py --arch vgg16 --layer block1 --decoder transposed_conv
  
  # Train ensemble model
  python train.py --arch ensemble --fusion attention --decoder transposed_conv
  
  # Train all configurations (batch mode)
  python train.py --mode all
  
Available Architectures:
  Single:
    - resnet34 (layers: layer1, layer2)
    - vgg16 (layers: block1, block3)
    - vit_small_patch16_224 (layers: block1, block3)
    - pvt_v2_b2 (layers: stage1, stage2)
  Ensemble:
    - ensemble (requires --fusion: attention/concat/weighted)

Available Decoders:
  - frequency_aware: Frequency decomposition with spatial-channel attention
  - wavelet: Wavelet-based multi-resolution reconstruction
  - transposed_conv: Simple transposed convolution (often best!)
  - attention: Transformer-based with self-attention

Available Fusion Strategies (ensemble only):
  - attention: Learned attention weights for each architecture
  - concat: Simple channel-wise concatenation
  - weighted: Learned scalar weights for each architecture

Training Configuration:
  - Epochs: 30 (early stopping with patience=15)
  - Learning rate: 0.0001 (Adam optimizer with ReduceLROnPlateau)
  - Loss: 0.5×MSE + 0.5×LPIPS (combined pixel + perceptual loss)
  - Batch size: 4 (limited by GPU memory for large feature maps)
  - Dataset: DIV2K (640 train, 160 val, 100 test)

Expected Training Time (NVIDIA RTX 6000 Ada):
  - Single model: 1-3 hours depending on architecture
  - Ensemble model: 3-4 hours (4 encoders + fusion)
  - Batch mode (all): ~84 hours (~3.5 days)

For More Information:
  See README.md for detailed documentation and baseline results.
        """
    )
    
    # Training mode
    parser.add_argument('--mode', type=str, choices=['all', 'single'],
                        help='Training mode: "all" (batch train all models), '
                             '"single" (train one model)')
    
    # Model specification
    parser.add_argument('--arch', type=str,
                        choices=['resnet34', 'vgg16', 'vit_small_patch16_224', 
                                'pvt_v2_b2', 'ensemble'],
                        help='Architecture name (required for single mode)')
    
    parser.add_argument('--layer', type=str,
                        help='Layer/block/stage name (e.g., layer1, block1, stage1)')
    
    parser.add_argument('--decoder', type=str,
                        choices=['frequency_aware', 'wavelet', 'transposed_conv', 'attention'],
                        help='Decoder type (required for single mode)')
    
    parser.add_argument('--fusion', type=str,
                        choices=['attention', 'concat', 'weighted'],
                        help='Fusion strategy (required for ensemble architecture)')
    
    return parser.parse_args()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Set random seeds for reproducibility
    # This must be done before any model creation or data loading
    set_random_seeds(RANDOM_SEED)
    
    # Log environment information for reproducibility
    env_info = log_environment_info()
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # If no arguments provided, show helpful usage message
    if len(sys.argv) == 1:
        print("\n" + "="*80)
        print("IMAGE RECONSTRUCTION MODEL TRAINING")
        print("="*80)
        print("\nNo arguments provided. Choose a training mode:")
        print("\n1. SINGLE MODEL TRAINING:")
        print("   Train one specific model configuration")
        print("   Example: python train.py --arch vgg16 --layer block1 --decoder transposed_conv")
        print("\n2. ENSEMBLE MODEL TRAINING:")
        print("   Train a multi-architecture ensemble model")
        print("   Example: python train.py --arch ensemble --fusion attention --decoder transposed_conv")
        print("\n3. BATCH TRAINING:")
        print("   Train all model configurations (44 models, ~84 hours)")
        print("   Example: python train.py --mode all")
        print("\nFor detailed help and options:")
        print("   python train.py --help")
        print("="*80 + "\n")
        sys.exit(0)
    
    # Route to appropriate training function based on mode
    if args.mode == 'all':
        # Batch training mode: train all configurations
        print("\n[MODE] Batch training - will train all configurations")
        print("[WARNING] This will take approximately 84 hours (~3.5 days)")
        print("[RECOMMENDATION] Use screen/tmux to keep process running")
        
        # Confirm batch training
        response = input("\nProceed with batch training? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Batch training cancelled.")
            sys.exit(0)
        
        train_all_experiments()
        
    elif args.mode == 'single' or (args.arch and args.decoder):
        # Single training mode: train one specific model
        print("\n[MODE] Single training")
        
        # Validate required arguments
        if not args.arch or not args.decoder:
            print("[ERROR] Single training requires --arch and --decoder arguments")
            print("Example: python train.py --arch vgg16 --layer block1 --decoder transposed_conv")
            sys.exit(1)
        
        # Handle ensemble vs single architecture
        if args.arch == 'ensemble':
            # Ensemble requires fusion strategy
            if not args.fusion:
                print("[ERROR] Ensemble architecture requires --fusion argument")
                print("Options: attention, concat, weighted")
                print("Example: python train.py --arch ensemble --fusion attention --decoder transposed_conv")
                sys.exit(1)
            train_single_experiment(args.arch, args.layer, args.decoder, args.fusion)
        else:
            # Single architecture requires layer
            if not args.layer:
                print("[ERROR] Single architecture requires --layer argument")
                print(f"For {args.arch}, valid layers are:")
                if args.arch == 'resnet34':
                    print("  - layer1, layer2")
                elif args.arch == 'vgg16':
                    print("  - block1, block3")
                elif args.arch == 'vit_small_patch16_224':
                    print("  - block1, block3")
                elif args.arch == 'pvt_v2_b2':
                    print("  - stage1, stage2")
                sys.exit(1)
            train_single_experiment(args.arch, args.layer, args.decoder)
    
    else:
        # Invalid argument combination
        print("[ERROR] Invalid arguments provided")
        print("Use --help for usage information")
        print("Quick examples:")
        print("  python train.py --mode all")
        print("  python train.py --arch vgg16 --layer block1 --decoder transposed_conv")
        sys.exit(1)
    
    print("\n[COMPLETE] Training finished successfully")
    print("[NOTE] Use evaluate.py to evaluate trained models on test set")
