"""
Image Reconstruction Model Evaluation Script
=============================================

This script evaluates trained image reconstruction models on the DIV2K test set.
It supports both single-architecture models and multi-architecture ensemble models
with various decoder types.

Key Features:
- Single model evaluation (ResNet34, VGG16, ViT, PVT)
- Ensemble model evaluation (multiple fusion strategies)
- Batch evaluation of all trained models
- Comprehensive comparison reports with CSV/JSON outputs
- Full reproducibility with seed setting and version logging

Usage Examples:
    # Evaluate single model
    python evaluate.py --arch vgg16 --layer block1 --decoder transposed_conv
    
    # Evaluate ensemble
    python evaluate.py --arch ensemble --fusion attention --decoder simple
    
    # Evaluate all experiments
    python evaluate.py --mode all

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

# Add the src directory to Python path (assumes src/ is sibling to scripts/)
# Project structure: project_root/scripts/evaluate.py and project_root/src/
SRC_DIR = SCRIPT_DIR.parent / 'src'
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
else:
    # If running from src/ directory itself
    sys.path.insert(0, str(SCRIPT_DIR))

# Import project modules
from models import SingleArchModel, EnsembleModel
from dataset import get_dataloaders
from training import load_and_evaluate


# ============================================================================
# REPRODUCIBILITY CONFIGURATION
# ============================================================================
# Set random seeds for reproducibility across different runs
# This ensures that any stochastic operations (dropout, data shuffling, etc.)
# produce the same results given the same input
RANDOM_SEED = 42

def set_random_seeds(seed=RANDOM_SEED):
    """
    Set random seeds for all libraries to ensure reproducibility.
    
    This function sets seeds for:
    - Python's random module
    - NumPy's random number generator
    - PyTorch's random number generators (CPU and CUDA)
    - PyTorch's deterministic operations
    
    Args:
        seed (int): Random seed value (default: 42)
    
    Note:
        Setting deterministic operations may impact performance but ensures
        exact reproducibility of results across different runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        # Make CuDNN deterministic (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set Python hash seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"[REPRODUCIBILITY] Random seed set to: {seed}")


def log_environment_info():
    """
    Log software versions and hardware configuration for reproducibility.
    
    Returns:
        dict: Dictionary containing version information
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
# These are the default hyperparameters used across all experiments
# Modifying these values will change experiment behavior and results

BASE_CONFIG = {
    'output_size': 224,        # Target image size (224x224 pixels)
                                # Chosen to match ImageNet pre-training resolution
    
    'batch_size': 4,            # Number of images processed simultaneously
                                # Smaller batch = less memory, more stable gradients
                                # Chosen based on GPU memory constraints (48GB RTX 6000)
    
    'num_workers': 2,           # Number of parallel data loading processes
                                # Balance between I/O speed and CPU usage
                                # Too many workers can cause slowdown
    
    'limit': None,              # Limit number of test images (None = use all)
                                # Useful for quick testing: set to 100 for fast runs
    
    'save_dir': 'results'       # Root directory for saving all results
}

# ============================================================================
# SINGLE ARCHITECTURE CONFIGURATIONS
# ============================================================================
# These configurations define which layers/blocks to extract features from
# for each architecture. Layer selection is based on baseline experiments
# showing that shallow layers (layer1/block1/stage1) provide best reconstruction.

SINGLE_ARCHITECTURES = {
    # ResNet34: Residual network with skip connections
    # Testing early (layer1) and mid (layer2) layers
    'resnet34_layer1': {
        'architecture': 'resnet34',
        'resnet_layer': 'layer1',       # 64 channels, 56x56 spatial
        'decoder_type': 'frequency_aware'  # Default decoder (will be overridden)
    },
    'resnet34_layer2': {
        'architecture': 'resnet34',
        'resnet_layer': 'layer2',       # 128 channels, 28x28 spatial
        'decoder_type': 'frequency_aware'
    },
    
    # VGG16: Simple sequential architecture with aggressive pooling
    # Block1 has highest spatial resolution (112x112) → best reconstruction
    'vgg16_block1': {
        'architecture': 'vgg16',
        'vgg_block': 'block1',          # 64 channels, 112x112 spatial (BEST)
        'decoder_type': 'frequency_aware'
    },
    'vgg16_block3': {
        'architecture': 'vgg16',
        'vgg_block': 'block3',          # 256 channels, 28x28 spatial
        'decoder_type': 'frequency_aware'
    },
    
    # Vision Transformer (ViT): Patch-based self-attention architecture
    # Maintains constant 14x14 spatial resolution through all blocks
    'vit_small_block1': {
        'architecture': 'vit_small_patch16_224',
        'vit_block': 'block1',          # 384 channels, 14x14 tokens
        'decoder_type': 'frequency_aware'
    },
    'vit_small_block3': {
        'architecture': 'vit_small_patch16_224',
        'vit_block': 'block3',          # 384 channels, 14x14 tokens
        'decoder_type': 'frequency_aware'
    },
    
    # Pyramid Vision Transformer (PVT): Hierarchical vision transformer
    # Combines benefits of CNNs (spatial hierarchy) and transformers (attention)
    'pvt_v2_b2_stage1': {
        'architecture': 'pvt_v2_b2',
        'pvt_stage': 'stage1',          # 64 channels, 56x56 spatial
        'decoder_type': 'frequency_aware'
    },
    'pvt_v2_b2_stage2': {
        'architecture': 'pvt_v2_b2',
        'pvt_stage': 'stage2',          # 128 channels, 28x28 spatial
        'decoder_type': 'frequency_aware'
    }
}

# ============================================================================
# DECODER VARIANTS
# ============================================================================
# Different decoder architectures tested in this study:
# - frequency_aware: Explicit low/high frequency separation with attention
# - wavelet: Multi-resolution wavelet sub-band prediction
# - transposed_conv: Simple stride-2 transposed convolutions (surprisingly best!)
# - attention: Transformer blocks with self-attention mechanisms

DECODER_VARIANTS = ['frequency_aware', 'wavelet', 'transposed_conv', 'attention']

# ============================================================================
# ENSEMBLE CONFIGURATIONS
# ============================================================================
# Multi-architecture ensembles that fuse features from multiple networks
# All ensembles use shallow layers (layer1/block1/stage1) based on baseline
# findings that spatial resolution is more important than semantic depth

ENSEMBLE_CONFIGS = {
    # Attention-based fusion: Learnable attention weights for each architecture
    'ensemble_all_attention': {
        'architectures': ['resnet34', 'vgg16', 'vit_small_patch16_224', 'pvt_v2_b2'],
        'resnet_layer': 'layer1',   # 64 ch, 56x56
        'vgg_block': 'block1',      # 64 ch, 112x112 (highest resolution)
        'vit_block': 'block1',      # 384 ch, 14x14
        'pvt_stage': 'stage1',      # 64 ch, 56x56
        'fusion_strategy': 'attention',
        'fusion_channels': 256,     # Intermediate feature dimension after fusion
        'decoder_type': 'frequency_aware'  # Will be overridden per experiment
    },
    
    # Concatenation fusion: Simple channel-wise concatenation
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
# SINGLE EVALUATION FUNCTION
# ============================================================================

def evaluate_single_experiment(architecture, layer, decoder, fusion=None):
    """
    Evaluate a single trained model on the test set.
    
    This function loads a trained model checkpoint and evaluates it on the
    DIV2K test set, computing PSNR, SSIM, and LPIPS metrics. Results are
    saved to the appropriate directory structure.
    
    Args:
        architecture (str): Architecture name. Options:
                          - 'resnet34': ResNet-34 with residual connections
                          - 'vgg16': VGG-16 sequential architecture
                          - 'vit_small_patch16_224': Vision Transformer
                          - 'pvt_v2_b2': Pyramid Vision Transformer
                          - 'ensemble': Multi-architecture ensemble
        
        layer (str): Layer/block/stage to extract features from. Options:
                    - ResNet: 'layer1', 'layer2', 'layer3', 'layer4'
                    - VGG: 'block1', 'block2', 'block3', 'block4', 'block5'
                    - ViT: 'block0', 'block1', ..., 'block11'
                    - PVT: 'stage1', 'stage2', 'stage3', 'stage4'
        
        decoder (str): Decoder architecture. Options:
                      - 'frequency_aware': Frequency decomposition decoder
                      - 'wavelet': Wavelet-based multi-resolution decoder
                      - 'transposed_conv': Simple transposed convolution decoder
                      - 'attention': Transformer-based attention decoder
        
        fusion (str, optional): Fusion strategy for ensemble. Required if
                               architecture='ensemble'. Options:
                               - 'attention': Attention-based fusion
                               - 'concat': Concatenation fusion
                               - 'weighted': Weighted scalar fusion
    
    Returns:
        None: Results are printed to console and saved to disk
    
    Example:
        >>> evaluate_single_experiment('vgg16', 'block1', 'transposed_conv')
        >>> evaluate_single_experiment('ensemble', None, 'simple', fusion='attention')
    """
    
    print("\n" + "="*80)
    print("SINGLE EVALUATION MODE")
    print("="*80)
    print(f"Architecture: {architecture}")
    print(f"Layer: {layer}")
    print(f"Decoder: {decoder}")
    if fusion:
        print(f"Fusion: {fusion}")
    print("="*80 + "\n")
    
    # Get device (CUDA/MPS/CPU)
    device = get_device()
    
    # Load test data
    # Note: We only load test_loader here since we're evaluating, not training
    print("\nLoading test dataset...")
    _, _, test_loader = get_dataloaders(
        batch_size=BASE_CONFIG['batch_size'],
        num_workers=BASE_CONFIG['num_workers'],
        limit=BASE_CONFIG['limit']
    )
    print(f"Test set size: {len(test_loader.dataset)} images")
    print(f"Batch size: {BASE_CONFIG['batch_size']}")
    print(f"Number of batches: {len(test_loader)}")
    
    # Build experiment configuration dictionary
    if architecture == 'ensemble':
        # Ensemble model requires fusion strategy
        if fusion is None:
            print("[ERROR] Fusion strategy required for ensemble (attention/concat/weighted)")
            print("Example: --arch ensemble --fusion attention --decoder simple")
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
        
    else:
        # Single architecture model
        # Determine which parameter name to use for the layer
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
        
        # Build configuration
        exp_config = BASE_CONFIG.copy()
        exp_config['architecture'] = architecture
        exp_config[layer_param] = layer
        exp_config['decoder_type'] = decoder
        exp_config['exp_name'] = f"{architecture}_{layer}_{decoder}"
    
    print(f"\n{'#'*80}")
    print(f"EVALUATING: {exp_config['exp_name']}")
    print(f"{'#'*80}\n")
    
    # Perform evaluation
    # The load_and_evaluate function will:
    # 1. Load the trained model checkpoint
    # 2. Run inference on test set
    # 3. Compute metrics (PSNR, SSIM, LPIPS)
    # 4. Save results and visualizations
    eval_start = time.time()
    metrics = load_and_evaluate(exp_config, test_loader, device)
    eval_time = time.time() - eval_start
    
    # Check if evaluation succeeded
    if metrics is None:
        print(f"[ERROR] Evaluation failed for {exp_config['exp_name']}")
        print("Possible reasons:")
        print("  - Checkpoint file not found")
        print("  - Model architecture mismatch")
        print("  - Out of memory")
        return
    
    # Print results summary
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE: {exp_config['exp_name']}")
    print(f"{'='*80}")
    print(f"Evaluation time: {eval_time/60:.2f} minutes")
    print(f"\nMetrics (higher PSNR/SSIM is better, lower LPIPS is better):")
    print(f"  PSNR: {metrics['psnr_mean']:.4f} dB (std: {metrics['psnr_std']:.4f})")
    print(f"  SSIM: {metrics['ssim_mean']:.4f} (std: {metrics['ssim_std']:.4f})")
    
    # Interpret PSNR score
    if metrics['psnr_mean'] > 17.0:
        quality = "Excellent"
    elif metrics['psnr_mean'] > 15.0:
        quality = "Good"
    elif metrics['psnr_mean'] > 13.0:
        quality = "Fair"
    else:
        quality = "Poor"
    print(f"  Quality: {quality}")
    
    result_dir = 'ensemble' if architecture == 'ensemble' else 'single'
    print(f"\nResults saved in: results/{result_dir}/evaluation_{exp_config['exp_name']}/")
    print(f"{'='*80}\n")


# ============================================================================
# BATCH EVALUATION FUNCTION
# ============================================================================

def evaluate_all_experiments():
    """
    Evaluate all trained models and generate comprehensive comparison report.
    
    This function runs evaluation on all single-architecture and ensemble models,
    then generates comparison reports showing:
    - Top 10 models by PSNR
    - Decoder architecture comparison
    - Single architecture comparison
    - Fusion strategy comparison (for ensembles)
    - Summary statistics
    
    The evaluation is organized in phases:
    1. Single architecture models (ResNet, VGG, ViT, PVT with all decoders)
    2. Ensemble models (all fusion strategies with all decoders)
    3. Report generation (CSV files, JSON summaries, console output)
    
    Results are saved to: results/comparison_report/
    
    Returns:
        None: Results are printed and saved to disk
    
    Note:
        This function can take several hours to complete depending on:
        - Number of trained models available
        - GPU speed
        - Test set size
        - Number of decoder variants
    """
    
    print("\n" + "="*80)
    print("BATCH EVALUATION MODE")
    print("="*80)
    print(f"Single architectures: {len(SINGLE_ARCHITECTURES)}")
    print(f"Decoder variants: {len(DECODER_VARIANTS)}")
    print(f"Total single models: {len(SINGLE_ARCHITECTURES)} × {len(DECODER_VARIANTS)} = {len(SINGLE_ARCHITECTURES) * len(DECODER_VARIANTS)}")
    print(f"\nEnsemble configurations: {len(ENSEMBLE_CONFIGS)}")
    print(f"Total ensemble models: {len(ENSEMBLE_CONFIGS)} × {len(DECODER_VARIANTS)} = {len(ENSEMBLE_CONFIGS) * len(DECODER_VARIANTS)}")
    print(f"\nGrand total: {(len(SINGLE_ARCHITECTURES) + len(ENSEMBLE_CONFIGS)) * len(DECODER_VARIANTS)} model evaluations")
    print("="*80 + "\n")
    
    # Start timing
    total_start = time.time()
    
    # Get device
    device = get_device()
    
    # Load test data once (reused for all evaluations)
    print("\nLoading test dataset...")
    _, _, test_loader = get_dataloaders(
        batch_size=BASE_CONFIG['batch_size'],
        num_workers=BASE_CONFIG['num_workers'],
        limit=BASE_CONFIG['limit']
    )
    print(f"Test: {len(test_loader.dataset)} images")
    
    # Initialize results storage
    all_results = {
        'single_models': [],    # Results for single-architecture models
        'ensemble_models': []   # Results for ensemble models
    }
    
    # ========================================================================
    # PHASE 1: EVALUATE SINGLE ARCHITECTURE MODELS
    # ========================================================================
    # Evaluate each combination of (architecture, layer, decoder)
    # E.g., resnet34_layer1_simple, vgg16_block1_attention, etc.
    
    print("\n" + "="*80)
    print("PHASE 1: SINGLE ARCHITECTURE MODELS")
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
            print(f"EVALUATING: {exp_config['exp_name']}")
            print(f"  Architecture: {arch_config['architecture']}")
            print(f"  Layer: {list(arch_config.values())[1]}")  # Extract layer name
            print(f"  Decoder: {decoder}")
            print(f"{'#'*80}\n")
            
            try:
                # Run evaluation
                eval_start = time.time()
                metrics = load_and_evaluate(exp_config, test_loader, device)
                eval_time = time.time() - eval_start
                
                # Check if checkpoint was found
                if metrics is None:
                    print(f"[WARNING] No checkpoint found for {exp_config['exp_name']}, skipping...")
                    phase1_failed += 1
                    continue
                
                # Record results
                # Extract layer name from config (second value in dict)
                result = {
                    'exp_name': exp_config['exp_name'],
                    'architecture': arch_config['architecture'],
                    'layer': list(arch_config.values())[1],  # e.g., 'layer1', 'block1'
                    'decoder': decoder,
                    'eval_time_minutes': eval_time / 60,
                    **metrics  # Unpack PSNR, SSIM, LPIPS metrics
                }
                all_results['single_models'].append(result)
                
                print(f"\n[SUCCESS] {exp_config['exp_name']}")
                print(f"  PSNR: {metrics['psnr_mean']:.2f} dB | SSIM: {metrics['ssim_mean']:.4f}")
                phase1_success += 1
                
            except Exception as e:
                # Catch any errors during evaluation
                print(f"\n[ERROR] {exp_config['exp_name']}: {str(e)}\n")
                all_results['single_models'].append({
                    'exp_name': exp_config['exp_name'],
                    'error': str(e)
                })
                phase1_failed += 1
    
    print(f"\n[PHASE 1 COMPLETE] Success: {phase1_success}, Failed: {phase1_failed}")
    
    # ========================================================================
    # PHASE 2: EVALUATE ENSEMBLE MODELS
    # ========================================================================
    # Evaluate each combination of (ensemble_config, decoder)
    # E.g., ensemble_all_attention_simple, ensemble_all_concat_wavelet, etc.
    
    print("\n" + "="*80)
    print("PHASE 2: ENSEMBLE MODELS")
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
            print(f"EVALUATING: {exp_config['exp_name']}")
            print(f"  Architectures: {', '.join(ensemble_config['architectures'])}")
            print(f"  Fusion: {ensemble_config['fusion_strategy']}")
            print(f"  Decoder: {decoder}")
            print(f"{'#'*80}\n")
            
            try:
                # Run evaluation
                eval_start = time.time()
                metrics = load_and_evaluate(exp_config, test_loader, device)
                eval_time = time.time() - eval_start
                
                # Check if checkpoint was found
                if metrics is None:
                    print(f"[WARNING] No checkpoint found for {exp_config['exp_name']}, skipping...")
                    phase2_failed += 1
                    continue
                
                # Record results
                result = {
                    'exp_name': exp_config['exp_name'],
                    'architectures': ensemble_config['architectures'],
                    'fusion': ensemble_config['fusion_strategy'],
                    'decoder': decoder,
                    'eval_time_minutes': eval_time / 60,
                    **metrics  # Unpack PSNR, SSIM, LPIPS metrics
                }
                all_results['ensemble_models'].append(result)
                
                print(f"\n[SUCCESS] {exp_config['exp_name']}")
                print(f"  PSNR: {metrics['psnr_mean']:.2f} dB | SSIM: {metrics['ssim_mean']:.4f}")
                phase2_success += 1
                
            except Exception as e:
                # Catch any errors during evaluation
                print(f"\n[ERROR] {exp_config['exp_name']}: {str(e)}\n")
                all_results['ensemble_models'].append({
                    'exp_name': exp_config['exp_name'],
                    'error': str(e)
                })
                phase2_failed += 1
    
    print(f"\n[PHASE 2 COMPLETE] Success: {phase2_success}, Failed: {phase2_failed}")
    
    # ========================================================================
    # PHASE 3: GENERATE COMPARISON REPORT
    # ========================================================================
    # Create comprehensive reports comparing all models
    
    total_time = time.time() - total_start
    
    print("\n" + "="*80)
    print("PHASE 3: GENERATING COMPARISON REPORT")
    print("="*80 + "\n")
    
    generate_comparison_report(all_results, total_time)
    
    print(f"\n{'='*80}")
    print(f"ALL EVALUATIONS COMPLETE")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Total successful: {phase1_success + phase2_success}")
    print(f"Total failed: {phase1_failed + phase2_failed}")
    print(f"{'='*80}\n")


def generate_comparison_report(results, total_time):
    """
    Generate comprehensive comparison report from evaluation results.
    
    This function processes raw evaluation results and creates:
    1. CSV files with all metrics for single and ensemble models
    2. Ranked tables showing top performers
    3. Comparison tables for decoders, architectures, and fusion strategies
    4. JSON summary with aggregate statistics
    
    Args:
        results (dict): Dictionary containing:
                       - 'single_models': List of single model results
                       - 'ensemble_models': List of ensemble model results
        total_time (float): Total evaluation time in seconds
    
    Outputs:
        Files saved to results/comparison_report/:
        - all_results_{timestamp}.json: Raw results
        - single_models_{timestamp}.csv: All single model metrics
        - ensemble_models_{timestamp}.csv: All ensemble model metrics
        - all_models_{timestamp}.csv: Combined rankings
        - decoder_comparison_{timestamp}.csv: Average performance by decoder
        - architecture_comparison_{timestamp}.csv: Average performance by architecture
        - fusion_comparison_{timestamp}.csv: Average performance by fusion strategy
        - summary_{timestamp}.json: Aggregate statistics
    
    Console Output:
        - Top 10 single models
        - All ensemble models
        - Top 10 overall models
        - Decoder comparison statistics
        - Architecture comparison statistics
        - Fusion strategy comparison statistics
        - Summary statistics
    """
    
    # Create output directory
    report_dir = Path('results') / 'comparison_report'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for file naming
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Save raw results as JSON for future reference
    print("Saving raw results...")
    with open(report_dir / f'all_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # ========================================================================
    # SINGLE MODELS ANALYSIS
    # ========================================================================
    # Filter out failed experiments (those with 'error' key)
    single_success = [r for r in results['single_models'] if 'error' not in r]
    
    if single_success:
        print("Processing single model results...")
        df_single = pd.DataFrame(single_success)
        
        # Sort by PSNR (primary metric for reconstruction quality)
        df_single = df_single.sort_values('psnr_mean', ascending=False)
        
        # Save to CSV
        df_single.to_csv(report_dir / f'single_models_{timestamp}.csv', index=False)
        
        # Display top 10 single models
        print("\n" + "="*80)
        print("TOP 10 SINGLE MODELS (by PSNR)")
        print("="*80)
        print(df_single[['exp_name', 'psnr_mean', 'ssim_mean', 'best_val_loss']].head(10).to_string(index=False))
    
    # ========================================================================
    # ENSEMBLE MODELS ANALYSIS
    # ========================================================================
    ensemble_success = [r for r in results['ensemble_models'] if 'error' not in r]
    
    if ensemble_success:
        print("\nProcessing ensemble model results...")
        df_ensemble = pd.DataFrame(ensemble_success)
        df_ensemble = df_ensemble.sort_values('psnr_mean', ascending=False)
        df_ensemble.to_csv(report_dir / f'ensemble_models_{timestamp}.csv', index=False)
        
        print("\n" + "="*80)
        print("ALL ENSEMBLE MODELS (by PSNR)")
        print("="*80)
        print(df_ensemble[['exp_name', 'psnr_mean', 'ssim_mean', 'best_val_loss']].to_string(index=False))
    
    # ========================================================================
    # OVERALL COMPARISON (SINGLE + ENSEMBLE)
    # ========================================================================
    all_success = single_success + ensemble_success
    
    if all_success:
        print("\nGenerating overall rankings...")
        df_all = pd.DataFrame(all_success)
        df_all = df_all.sort_values('psnr_mean', ascending=False)
        df_all.to_csv(report_dir / f'all_models_{timestamp}.csv', index=False)
        
        print("\n" + "="*80)
        print("TOP 10 OVERALL MODELS (by PSNR)")
        print("="*80)
        print(df_all[['exp_name', 'psnr_mean', 'ssim_mean', 'best_val_loss']].head(10).to_string(index=False))
    
    # ========================================================================
    # DECODER COMPARISON
    # ========================================================================
    # Compare average performance across different decoder types
    if single_success:
        print("\nAnalyzing decoder performance...")
        print("\n" + "="*80)
        print("DECODER COMPARISON (Single Models)")
        print("="*80)
        print("Shows average PSNR and SSIM for each decoder type")
        print("Higher PSNR/SSIM indicates better reconstruction quality")
        print("-"*80)
        
        decoder_comparison = df_single.groupby('decoder').agg({
            'psnr_mean': ['mean', 'std', 'count'],
            'ssim_mean': ['mean', 'std']
        }).round(4)
        print(decoder_comparison)
        decoder_comparison.to_csv(report_dir / f'decoder_comparison_{timestamp}.csv')
    
    # ========================================================================
    # ARCHITECTURE COMPARISON
    # ========================================================================
    # Compare average performance across different base architectures
    if single_success:
        print("\nAnalyzing architecture performance...")
        print("\n" + "="*80)
        print("ARCHITECTURE COMPARISON (Single Models)")
        print("="*80)
        print("Shows average PSNR and SSIM for each architecture")
        print("Helps identify which architecture best preserves spatial information")
        print("-"*80)
        
        arch_comparison = df_single.groupby('architecture').agg({
            'psnr_mean': ['mean', 'std', 'count'],
            'ssim_mean': ['mean', 'std']
        }).round(4)
        print(arch_comparison)
        arch_comparison.to_csv(report_dir / f'architecture_comparison_{timestamp}.csv')
    
    # ========================================================================
    # FUSION STRATEGY COMPARISON
    # ========================================================================
    # Compare average performance across different ensemble fusion methods
    if ensemble_success:
        print("\nAnalyzing fusion strategies...")
        print("\n" + "="*80)
        print("FUSION STRATEGY COMPARISON (Ensemble Models)")
        print("="*80)
        print("Shows average PSNR and SSIM for each fusion method")
        print("  - attention: Learned attention weights")
        print("  - concat: Simple concatenation")
        print("  - weighted: Learned scalar weights")
        print("-"*80)
        
        fusion_comparison = df_ensemble.groupby('fusion').agg({
            'psnr_mean': ['mean', 'std', 'count'],
            'ssim_mean': ['mean', 'std']
        }).round(4)
        print(fusion_comparison)
        fusion_comparison.to_csv(report_dir / f'fusion_comparison_{timestamp}.csv')
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    print("\nGenerating summary statistics...")
    
    summary = {
        'timestamp': timestamp,
        'evaluation_date': datetime.now().isoformat(),
        'total_time_hours': round(total_time / 3600, 2),
        'total_experiments': len(results['single_models']) + len(results['ensemble_models']),
        
        'single_models': {
            'total': len(results['single_models']),
            'successful': len(single_success),
            'failed': len(results['single_models']) - len(single_success),
            'success_rate': len(single_success) / len(results['single_models']) if results['single_models'] else 0
        },
        
        'ensemble_models': {
            'total': len(results['ensemble_models']),
            'successful': len(ensemble_success),
            'failed': len(results['ensemble_models']) - len(ensemble_success),
            'success_rate': len(ensemble_success) / len(results['ensemble_models']) if results['ensemble_models'] else 0
        },
        
        'random_seed': RANDOM_SEED,
        'batch_size': BASE_CONFIG['batch_size'],
        'test_set_size': BASE_CONFIG['limit'] if BASE_CONFIG['limit'] else 'full'
    }
    
    # Add best model information
    if all_success:
        best_model = df_all.iloc[0]
        summary['best_overall'] = {
            'name': best_model['exp_name'],
            'psnr': float(best_model['psnr_mean']),
            'ssim': float(best_model['ssim_mean']),
            'type': 'ensemble' if 'ensemble' in best_model['exp_name'] else 'single'
        }
        
        # Add best single model if different from overall best
        if single_success:
            best_single = df_single.iloc[0]
            if best_single['exp_name'] != best_model['exp_name']:
                summary['best_single'] = {
                    'name': best_single['exp_name'],
                    'psnr': float(best_single['psnr_mean']),
                    'ssim': float(best_single['ssim_mean'])
                }
    
    # Save summary
    with open(report_dir / f'summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Display summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(json.dumps(summary, indent=2))
    
    print(f"\n[SAVED] All comparison reports in: {report_dir}")
    print(f"[SAVED] Summary file: summary_{timestamp}.json")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_device():
    """
    Detect and return the best available compute device.
    
    Priority order:
    1. CUDA (NVIDIA GPU) - fastest for deep learning
    2. MPS (Apple Silicon) - fast on M1/M2/M3 Macs
    3. CPU - slowest fallback
    
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    
    Side Effects:
        Prints device information to console
    """
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[DEVICE] CUDA GPU: {gpu_name}")
        print(f"[DEVICE] GPU Memory: {gpu_memory:.1f} GB")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("[DEVICE] Apple Silicon MPS")
        print("[DEVICE] Note: MPS may be slower than CUDA for some operations")
    else:
        device = 'cpu'
        print("[DEVICE] CPU (Warning: This will be very slow)")
        print("[DEVICE] Consider using a GPU for faster evaluation")
    
    return device


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
        $ python evaluate.py --mode all
        $ python evaluate.py --arch vgg16 --layer block1 --decoder transposed_conv
        $ python evaluate.py --arch ensemble --fusion attention --decoder simple
    """
    parser = argparse.ArgumentParser(
        description='Evaluate trained image reconstruction models on DIV2K test set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single model
  python evaluate.py --arch vgg16 --layer block1 --decoder transposed_conv
  
  # Evaluate ensemble model
  python evaluate.py --arch ensemble --fusion attention --decoder simple
  
  # Evaluate all trained models (generates comparison report)
  python evaluate.py --mode all
  
Available Architectures:
  Single:
    - resnet34 (layers: layer1, layer2)
    - vgg16 (layers: block1, block3)
    - vit_small_patch16_224 (layers: block1, block3)
    - pvt_v2_b2 (layers: stage1, stage2)
  Ensemble:
    - ensemble (requires --fusion: attention/concat/weighted)

Available Decoders:
  - frequency_aware: Frequency decomposition with attention
  - wavelet: Wavelet-based multi-resolution
  - transposed_conv: Simple transposed convolution (often best!)
  - attention: Transformer-based with self-attention

Available Fusion Strategies (ensemble only):
  - attention: Learned attention weights for each architecture
  - concat: Simple channel-wise concatenation
  - weighted: Learned scalar weights for each architecture

Metrics:
  - PSNR (dB): Peak Signal-to-Noise Ratio (higher is better)
               >17 dB: Excellent, 15-17 dB: Good, 13-15 dB: Fair, <13 dB: Poor
  - SSIM: Structural Similarity Index (higher is better, range 0-1)
               >0.55: Excellent, 0.45-0.55: Good, 0.35-0.45: Fair, <0.35: Poor
  - LPIPS: Learned Perceptual Image Patch Similarity (lower is better)

For More Information:
  See README.md for detailed documentation and baseline results.
        """
    )
    
    # Evaluation mode
    parser.add_argument('--mode', type=str, choices=['all', 'single'],
                        help='Evaluation mode: "all" (batch evaluate all models), '
                             '"single" (evaluate one model)')
    
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
    set_random_seeds(RANDOM_SEED)
    
    # Log environment information
    env_info = log_environment_info()
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # If no arguments provided, show helpful usage message
    if len(sys.argv) == 1:
        print("\n" + "="*80)
        print("IMAGE RECONSTRUCTION MODEL EVALUATION")
        print("="*80)
        print("\nNo arguments provided. Choose an evaluation mode:")
        print("\n1. SINGLE MODEL EVALUATION:")
        print("   Evaluate one specific trained model")
        print("   Example: python evaluate.py --arch vgg16 --layer block1 --decoder transposed_conv")
        print("\n2. ENSEMBLE MODEL EVALUATION:")
        print("   Evaluate a multi-architecture ensemble model")
        print("   Example: python evaluate.py --arch ensemble --fusion attention --decoder simple")
        print("\n3. BATCH EVALUATION:")
        print("   Evaluate all trained models and generate comparison report")
        print("   Example: python evaluate.py --mode all")
        print("\nFor detailed help and options:")
        print("   python evaluate.py --help")
        print("="*80 + "\n")
        sys.exit(0)
    
    # Route to appropriate evaluation function based on mode
    if args.mode == 'all':
        # Batch evaluation mode: evaluate all trained models
        print("\n[MODE] Batch evaluation - will evaluate all trained models")
        evaluate_all_experiments()
        
    elif args.mode == 'single' or (args.arch and args.decoder):
        # Single evaluation mode: evaluate one specific model
        print("\n[MODE] Single evaluation")
        
        # Validate required arguments
        if not args.arch or not args.decoder:
            print("[ERROR] Single evaluation requires --arch and --decoder arguments")
            print("Example: python evaluate.py --arch vgg16 --layer block1 --decoder transposed_conv")
            sys.exit(1)
        
        # Handle ensemble vs single architecture
        if args.arch == 'ensemble':
            # Ensemble requires fusion strategy
            if not args.fusion:
                print("[ERROR] Ensemble architecture requires --fusion argument")
                print("Options: attention, concat, weighted")
                print("Example: python evaluate.py --arch ensemble --fusion attention --decoder simple")
                sys.exit(1)
            evaluate_single_experiment(args.arch, args.layer, args.decoder, args.fusion)
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
            evaluate_single_experiment(args.arch, args.layer, args.decoder)
    
    else:
        # Invalid argument combination
        print("[ERROR] Invalid arguments provided")
        print("Use --help for usage information")
        print("Quick examples:")
        print("  python evaluate.py --mode all")
        print("  python evaluate.py --arch vgg16 --layer block1 --decoder transposed_conv")
        sys.exit(1)
    
    print("\n[COMPLETE] Evaluation finished successfully")
