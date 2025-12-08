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
- Dual sorting by PSNR and SSIM metrics
- Full reproducibility with seed setting and version logging

Usage Examples:
    # Evaluate single model
    python evaluate.py --arch vgg16 --layer block1 --decoder transposed_conv
    
    # Evaluate ensemble
    python evaluate.py --arch ensemble --fusion attention --decoder transposed_conv
    
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
RANDOM_SEED = 42

def set_random_seeds(seed=RANDOM_SEED):
    """
    Set random seeds for all libraries to ensure reproducibility.
    
    Args:
        seed (int): Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
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

BASE_CONFIG = {
    'output_size': 224,
    'batch_size': 4,
    'num_workers': 2,
    'limit': None,
    'save_dir': 'results'
}

# ============================================================================
# SINGLE ARCHITECTURE CONFIGURATIONS
# ============================================================================

SINGLE_ARCHITECTURES = {
    'resnet34_layer1': {
        'architecture': 'resnet34',
        'resnet_layer': 'layer1',
        'decoder_type': 'frequency_aware'
    },
    'resnet34_layer2': {
        'architecture': 'resnet34',
        'resnet_layer': 'layer2',
        'decoder_type': 'frequency_aware'
    },
    'vgg16_block1': {
        'architecture': 'vgg16',
        'vgg_block': 'block1',
        'decoder_type': 'frequency_aware'
    },
    'vgg16_block3': {
        'architecture': 'vgg16',
        'vgg_block': 'block3',
        'decoder_type': 'frequency_aware'
    },
    'vit_small_block1': {
        'architecture': 'vit_small_patch16_224',
        'vit_block': 'block1',
        'decoder_type': 'frequency_aware'
    },
    'vit_small_block3': {
        'architecture': 'vit_small_patch16_224',
        'vit_block': 'block3',
        'decoder_type': 'frequency_aware'
    },
    'pvt_v2_b2_stage1': {
        'architecture': 'pvt_v2_b2',
        'pvt_stage': 'stage1',
        'decoder_type': 'frequency_aware'
    },
    'pvt_v2_b2_stage2': {
        'architecture': 'pvt_v2_b2',
        'pvt_stage': 'stage2',
        'decoder_type': 'frequency_aware'
    }
}

DECODER_VARIANTS = ['frequency_aware', 'wavelet', 'transposed_conv', 'attention']

# ============================================================================
# ENSEMBLE CONFIGURATIONS
# ============================================================================

ENSEMBLE_CONFIGS = {
    'ensemble_all_attention': {
        'architectures': ['resnet34', 'vgg16', 'vit_small_patch16_224', 'pvt_v2_b2'],
        'resnet_layer': 'layer1',
        'vgg_block': 'block1',
        'vit_block': 'block1',
        'pvt_stage': 'stage1',
        'fusion_strategy': 'attention',
        'fusion_channels': 256,
        'decoder_type': 'frequency_aware'
    },
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
# PARAMETER COUNTING UTILITY
# ============================================================================

def count_decoder_parameters(model, device):
    """
    Count the number of trainable parameters in the decoder.
    
    Args:
        model: PyTorch model (SingleArchModel or EnsembleModel)
        device: Device to load model on
    
    Returns:
        int: Number of trainable decoder parameters
    """
    try:
        model = model.to(device)
        decoder_params = 0
        for name, param in model.named_parameters():
            if 'decoder' in name and param.requires_grad:
                decoder_params += param.numel()
        return decoder_params
    except Exception as e:
        print(f"[WARNING] Could not count parameters: {str(e)}")
        return None


# ============================================================================
# SINGLE EVALUATION FUNCTION
# ============================================================================

def evaluate_single_experiment(architecture, layer, decoder, fusion=None):
    """
    Evaluate a single trained model on the test set.
    
    Args:
        architecture (str): Architecture name
        layer (str): Layer/block/stage to extract features from
        decoder (str): Decoder architecture
        fusion (str, optional): Fusion strategy for ensemble
    
    Returns:
        None: Results are printed to console and saved to disk
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
    
    device = get_device()
    
    print("\nLoading test dataset...")
    _, _, test_loader = get_dataloaders(
        batch_size=BASE_CONFIG['batch_size'],
        num_workers=BASE_CONFIG['num_workers'],
        limit=BASE_CONFIG['limit']
    )
    print(f"Test set size: {len(test_loader.dataset)} images")
    
    if architecture == 'ensemble':
        if fusion is None:
            print("[ERROR] Fusion strategy required for ensemble (attention/concat/weighted)")
            return
        
        ensemble_key = f'ensemble_all_{fusion}'
        if ensemble_key not in ENSEMBLE_CONFIGS:
            print(f"[ERROR] Unknown ensemble configuration: {ensemble_key}")
            return
        
        exp_config = {**BASE_CONFIG, **ENSEMBLE_CONFIGS[ensemble_key]}
        exp_config['decoder_type'] = decoder
        exp_config['exp_name'] = f"{ensemble_key}_{decoder}"
        
    else:
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
            return
        
        exp_config = BASE_CONFIG.copy()
        exp_config['architecture'] = architecture
        exp_config[layer_param] = layer
        exp_config['decoder_type'] = decoder
        exp_config['exp_name'] = f"{architecture}_{layer}_{decoder}"
    
    print(f"\n{'#'*80}")
    print(f"EVALUATING: {exp_config['exp_name']}")
    print(f"{'#'*80}\n")
    
    eval_start = time.time()
    metrics = load_and_evaluate(exp_config, test_loader, device)
    eval_time = time.time() - eval_start
    
    if metrics is None:
        print(f"[ERROR] Evaluation failed for {exp_config['exp_name']}")
        return
    
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE: {exp_config['exp_name']}")
    print(f"{'='*80}")
    print(f"Evaluation time: {eval_time/60:.2f} minutes")
    print(f"\nMetrics:")
    print(f"  PSNR: {metrics['psnr_mean']:.4f} dB (std: {metrics['psnr_std']:.4f})")
    print(f"  SSIM: {metrics['ssim_mean']:.4f} (std: {metrics['ssim_std']:.4f})")
    
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
    
    Returns:
        None: Results are printed and saved to disk
    """
    
    print("\n" + "="*80)
    print("BATCH EVALUATION MODE")
    print("="*80)
    print(f"Single architectures: {len(SINGLE_ARCHITECTURES)}")
    print(f"Decoder variants: {len(DECODER_VARIANTS)}")
    print(f"Total single models: {len(SINGLE_ARCHITECTURES) * len(DECODER_VARIANTS)}")
    print(f"\nEnsemble configurations: {len(ENSEMBLE_CONFIGS)}")
    print(f"Total ensemble models: {len(ENSEMBLE_CONFIGS) * len(DECODER_VARIANTS)}")
    print(f"\nGrand total: {(len(SINGLE_ARCHITECTURES) + len(ENSEMBLE_CONFIGS)) * len(DECODER_VARIANTS)} evaluations")
    print("="*80 + "\n")
    
    total_start = time.time()
    device = get_device()
    
    print("\nLoading test dataset...")
    _, _, test_loader = get_dataloaders(
        batch_size=BASE_CONFIG['batch_size'],
        num_workers=BASE_CONFIG['num_workers'],
        limit=BASE_CONFIG['limit']
    )
    print(f"Test: {len(test_loader.dataset)} images")
    
    all_results = {
        'single_models': [],
        'ensemble_models': []
    }
    
    # ========================================================================
    # PHASE 1: EVALUATE SINGLE ARCHITECTURE MODELS
    # ========================================================================
    
    print("\n" + "="*80)
    print("PHASE 1: SINGLE ARCHITECTURE MODELS")
    print("="*80 + "\n")
    
    phase1_success = 0
    phase1_failed = 0
    
    for arch_name, arch_config in SINGLE_ARCHITECTURES.items():
        for decoder in DECODER_VARIANTS:
            exp_config = {**BASE_CONFIG, **arch_config}
            exp_config['decoder_type'] = decoder
            exp_config['exp_name'] = f"{arch_name}_{decoder}"
            
            print(f"\n{'#'*80}")
            print(f"EVALUATING: {exp_config['exp_name']}")
            print(f"{'#'*80}\n")
            
            try:
                eval_start = time.time()
                metrics = load_and_evaluate(exp_config, test_loader, device)
                eval_time = time.time() - eval_start
                
                if metrics is None:
                    print(f"[WARNING] No checkpoint found for {exp_config['exp_name']}, skipping...")
                    phase1_failed += 1
                    continue
                
                # Count decoder parameters
                print("\nCounting decoder parameters...")
                try:
                    model = SingleArchModel(**exp_config)
                    decoder_params = count_decoder_parameters(model, device)
                    del model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                except Exception as e:
                    print(f"[WARNING] Could not count parameters: {str(e)}")
                    decoder_params = None
                
                result = {
                    'exp_name': exp_config['exp_name'],
                    'architecture': arch_config['architecture'],
                    'layer': list(arch_config.values())[1],
                    'decoder': decoder,
                    'decoder_params': decoder_params,
                    'eval_time_minutes': eval_time / 60,
                    **metrics
                }
                all_results['single_models'].append(result)
                
                print(f"\n[SUCCESS] {exp_config['exp_name']}")
                print(f"  PSNR: {metrics['psnr_mean']:.4f} dB | SSIM: {metrics['ssim_mean']:.4f}")
                if decoder_params:
                    print(f"  Decoder Parameters: {decoder_params:,}")
                phase1_success += 1
                
            except Exception as e:
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
    
    print("\n" + "="*80)
    print("PHASE 2: ENSEMBLE MODELS")
    print("="*80 + "\n")
    
    phase2_success = 0
    phase2_failed = 0
    
    for ensemble_name, ensemble_config in ENSEMBLE_CONFIGS.items():
        for decoder in DECODER_VARIANTS:
            exp_config = {**BASE_CONFIG, **ensemble_config}
            exp_config['decoder_type'] = decoder
            exp_config['exp_name'] = f"{ensemble_name}_{decoder}"
            
            print(f"\n{'#'*80}")
            print(f"EVALUATING: {exp_config['exp_name']}")
            print(f"{'#'*80}\n")
            
            try:
                eval_start = time.time()
                metrics = load_and_evaluate(exp_config, test_loader, device)
                eval_time = time.time() - eval_start
                
                if metrics is None:
                    print(f"[WARNING] No checkpoint found for {exp_config['exp_name']}, skipping...")
                    phase2_failed += 1
                    continue
                
                # Count decoder parameters
                print("\nCounting decoder parameters...")
                try:
                    model = EnsembleModel(**exp_config)
                    decoder_params = count_decoder_parameters(model, device)
                    del model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                except Exception as e:
                    print(f"[WARNING] Could not count parameters: {str(e)}")
                    decoder_params = None
                
                result = {
                    'exp_name': exp_config['exp_name'],
                    'architectures': ensemble_config['architectures'],
                    'fusion': ensemble_config['fusion_strategy'],
                    'decoder': decoder,
                    'decoder_params': decoder_params,
                    'eval_time_minutes': eval_time / 60,
                    **metrics
                }
                all_results['ensemble_models'].append(result)
                
                print(f"\n[SUCCESS] {exp_config['exp_name']}")
                print(f"  PSNR: {metrics['psnr_mean']:.4f} dB | SSIM: {metrics['ssim_mean']:.4f}")
                if decoder_params:
                    print(f"  Decoder Parameters: {decoder_params:,}")
                phase2_success += 1
                
            except Exception as e:
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
    Includes dual sorting by PSNR and SSIM.
    
    Args:
        results (dict): Dictionary containing evaluation results
        total_time (float): Total evaluation time in seconds
    """
    
    report_dir = Path('results') / 'comparison_report'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    print("Saving raw results...")
    with open(report_dir / f'all_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # ========================================================================
    # SINGLE MODELS ANALYSIS
    # ========================================================================
    
    single_success = [r for r in results['single_models'] if 'error' not in r]
    
    if single_success:
        print("Processing single model results...")
        df_single = pd.DataFrame(single_success)
        
        # Sort by PSNR (primary)
        df_single_psnr = df_single.sort_values('psnr_mean', ascending=False).copy()
        df_single_psnr.to_csv(report_dir / f'single_models_by_psnr_{timestamp}.csv', index=False)
        
        # Sort by SSIM
        df_single_ssim = df_single.sort_values('ssim_mean', ascending=False).copy()
        df_single_ssim.to_csv(report_dir / f'single_models_by_ssim_{timestamp}.csv', index=False)
        
        # Sort by combined score (normalized PSNR + SSIM)
        df_single['combined_score'] = (
            df_single['psnr_mean'] / df_single['psnr_mean'].max() * 0.5 +
            df_single['ssim_mean'] / df_single['ssim_mean'].max() * 0.5
        )
        df_single_combined = df_single.sort_values('combined_score', ascending=False).copy()
        df_single_combined.to_csv(report_dir / f'single_models_by_combined_{timestamp}.csv', index=False)
        
        print("\n" + "="*80)
        print("TOP 10 SINGLE MODELS (sorted by PSNR)")
        print("="*80)
        display_cols = ['exp_name', 'psnr_mean', 'ssim_mean']
        if 'decoder_params' in df_single_psnr.columns:
            display_cols.append('decoder_params')
        print(df_single_psnr[display_cols].head(10).to_string(index=False))
        
        print("\n" + "="*80)
        print("TOP 10 SINGLE MODELS (sorted by SSIM)")
        print("="*80)
        print(df_single_ssim[display_cols].head(10).to_string(index=False))
        
        print("\n" + "="*80)
        print("TOP 10 SINGLE MODELS (sorted by Combined Score)")
        print("="*80)
        display_cols_combined = ['exp_name', 'psnr_mean', 'ssim_mean', 'combined_score']
        if 'decoder_params' in df_single_combined.columns:
            display_cols_combined.append('decoder_params')
        print(df_single_combined[display_cols_combined].head(10).to_string(index=False))
    
    # ========================================================================
    # ENSEMBLE MODELS ANALYSIS
    # ========================================================================
    
    ensemble_success = [r for r in results['ensemble_models'] if 'error' not in r]
    
    if ensemble_success:
        print("\nProcessing ensemble model results...")
        df_ensemble = pd.DataFrame(ensemble_success)
        
        # Sort by PSNR
        df_ensemble_psnr = df_ensemble.sort_values('psnr_mean', ascending=False).copy()
        df_ensemble_psnr.to_csv(report_dir / f'ensemble_models_by_psnr_{timestamp}.csv', index=False)
        
        # Sort by SSIM
        df_ensemble_ssim = df_ensemble.sort_values('ssim_mean', ascending=False).copy()
        df_ensemble_ssim.to_csv(report_dir / f'ensemble_models_by_ssim_{timestamp}.csv', index=False)
        
        # Sort by combined score
        df_ensemble['combined_score'] = (
            df_ensemble['psnr_mean'] / df_ensemble['psnr_mean'].max() * 0.5 +
            df_ensemble['ssim_mean'] / df_ensemble['ssim_mean'].max() * 0.5
        )
        df_ensemble_combined = df_ensemble.sort_values('combined_score', ascending=False).copy()
        df_ensemble_combined.to_csv(report_dir / f'ensemble_models_by_combined_{timestamp}.csv', index=False)
        
        print("\n" + "="*80)
        print("ENSEMBLE MODELS (sorted by PSNR)")
        print("="*80)
        display_cols = ['exp_name', 'psnr_mean', 'ssim_mean']
        if 'decoder_params' in df_ensemble_psnr.columns:
            display_cols.append('decoder_params')
        print(df_ensemble_psnr[display_cols].to_string(index=False))
        
        print("\n" + "="*80)
        print("ENSEMBLE MODELS (sorted by SSIM)")
        print("="*80)
        print(df_ensemble_ssim[display_cols].to_string(index=False))
        
        print("\n" + "="*80)
        print("ENSEMBLE MODELS (sorted by Combined Score)")
        print("="*80)
        display_cols_combined = ['exp_name', 'psnr_mean', 'ssim_mean', 'combined_score']
        if 'decoder_params' in df_ensemble_combined.columns:
            display_cols_combined.append('decoder_params')
        print(df_ensemble_combined[display_cols_combined].to_string(index=False))
    
    # ========================================================================
    # OVERALL COMPARISON (SINGLE + ENSEMBLE)
    # ========================================================================
    
    all_success = single_success + ensemble_success
    
    if all_success:
        print("\nGenerating overall rankings...")
        df_all = pd.DataFrame(all_success)
        
        # Calculate combined score
        df_all['combined_score'] = (
            df_all['psnr_mean'] / df_all['psnr_mean'].max() * 0.5 +
            df_all['ssim_mean'] / df_all['ssim_mean'].max() * 0.5
        )
        
        # Sort by PSNR
        df_all_psnr = df_all.sort_values('psnr_mean', ascending=False).copy()
        df_all_psnr.to_csv(report_dir / f'all_models_by_psnr_{timestamp}.csv', index=False)
        
        # Sort by SSIM
        df_all_ssim = df_all.sort_values('ssim_mean', ascending=False).copy()
        df_all_ssim.to_csv(report_dir / f'all_models_by_ssim_{timestamp}.csv', index=False)
        
        # Sort by combined score
        df_all_combined = df_all.sort_values('combined_score', ascending=False).copy()
        df_all_combined.to_csv(report_dir / f'all_models_by_combined_{timestamp}.csv', index=False)
        
        print("\n" + "="*80)
        print("TOP 10 OVERALL MODELS (sorted by PSNR)")
        print("="*80)
        display_cols = ['exp_name', 'psnr_mean', 'ssim_mean']
        if 'decoder_params' in df_all_psnr.columns:
            display_cols.append('decoder_params')
        print(df_all_psnr[display_cols].head(10).to_string(index=False))
        
        print("\n" + "="*80)
        print("TOP 10 OVERALL MODELS (sorted by SSIM)")
        print("="*80)
        print(df_all_ssim[display_cols].head(10).to_string(index=False))
        
        print("\n" + "="*80)
        print("TOP 10 OVERALL MODELS (sorted by Combined Score)")
        print("="*80)
        display_cols_combined = ['exp_name', 'psnr_mean', 'ssim_mean', 'combined_score']
        if 'decoder_params' in df_all_combined.columns:
            display_cols_combined.append('decoder_params')
        print(df_all_combined[display_cols_combined].head(10).to_string(index=False))
    
    # ========================================================================
    # DECODER COMPARISON
    # ========================================================================
    
    if single_success:
        print("\n" + "="*80)
        print("DECODER COMPARISON (Single Models)")
        print("="*80)
        
        decoder_comparison = df_single.groupby('decoder').agg({
            'psnr_mean': ['mean', 'std', 'max', 'min', 'count'],
            'ssim_mean': ['mean', 'std', 'max', 'min']
        }).round(4)
        
        decoder_comparison.columns = ['_'.join(col).strip() for col in decoder_comparison.columns.values]
        decoder_comparison = decoder_comparison.sort_values('psnr_mean_mean', ascending=False)
        
        print(decoder_comparison.to_string())
        decoder_comparison.to_csv(report_dir / f'decoder_comparison_{timestamp}.csv')
    
    # ========================================================================
    # DECODER PARAMETERS COMPARISON
    # ========================================================================
    
    if single_success and 'decoder_params' in df_single.columns:
        print("\n" + "="*80)
        print("DECODER PARAMETERS COMPARISON")
        print("="*80)
        
        df_params = df_single[df_single['decoder_params'].notna()]
        if len(df_params) > 0:
            decoder_params_comparison = df_params.groupby('decoder').agg({
                'decoder_params': ['mean', 'std', 'min', 'max', 'count']
            }).round(0)
            
            decoder_params_comparison.columns = ['_'.join(col).strip() for col in decoder_params_comparison.columns.values]
            print(decoder_params_comparison.to_string())
            decoder_params_comparison.to_csv(report_dir / f'decoder_params_comparison_{timestamp}.csv')
    
    # ========================================================================
    # ARCHITECTURE COMPARISON
    # ========================================================================
    
    if single_success:
        print("\n" + "="*80)
        print("ARCHITECTURE COMPARISON (Single Models)")
        print("="*80)
        
        arch_comparison = df_single.groupby('architecture').agg({
            'psnr_mean': ['mean', 'std', 'max', 'min', 'count'],
            'ssim_mean': ['mean', 'std', 'max', 'min']
        }).round(4)
        
        arch_comparison.columns = ['_'.join(col).strip() for col in arch_comparison.columns.values]
        arch_comparison = arch_comparison.sort_values('psnr_mean_mean', ascending=False)
        
        print(arch_comparison.to_string())
        arch_comparison.to_csv(report_dir / f'architecture_comparison_{timestamp}.csv')
    
    # ========================================================================
    # FUSION STRATEGY COMPARISON
    # ========================================================================
    
    if ensemble_success:
        print("\n" + "="*80)
        print("FUSION STRATEGY COMPARISON (Ensemble Models)")
        print("="*80)
        
        fusion_comparison = df_ensemble.groupby('fusion').agg({
            'psnr_mean': ['mean', 'std', 'max', 'min', 'count'],
            'ssim_mean': ['mean', 'std', 'max', 'min']
        }).round(4)
        
        fusion_comparison.columns = ['_'.join(col).strip() for col in fusion_comparison.columns.values]
        fusion_comparison = fusion_comparison.sort_values('psnr_mean_mean', ascending=False)
        
        print(fusion_comparison.to_string())
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
    
    # Add best model information for each metric
    if all_success:
        # Best by PSNR
        best_psnr = df_all_psnr.iloc[0]
        summary['best_by_psnr'] = {
            'name': best_psnr['exp_name'],
            'psnr': float(best_psnr['psnr_mean']),
            'ssim': float(best_psnr['ssim_mean']),
            'type': 'ensemble' if 'ensemble' in best_psnr['exp_name'] else 'single'
        }
        if 'decoder_params' in best_psnr and pd.notna(best_psnr['decoder_params']):
            summary['best_by_psnr']['decoder_params'] = int(best_psnr['decoder_params'])
        
        # Best by SSIM
        best_ssim = df_all_ssim.iloc[0]
        summary['best_by_ssim'] = {
            'name': best_ssim['exp_name'],
            'psnr': float(best_ssim['psnr_mean']),
            'ssim': float(best_ssim['ssim_mean']),
            'type': 'ensemble' if 'ensemble' in best_ssim['exp_name'] else 'single'
        }
        if 'decoder_params' in best_ssim and pd.notna(best_ssim['decoder_params']):
            summary['best_by_ssim']['decoder_params'] = int(best_ssim['decoder_params'])
        
        # Best by combined score
        best_combined = df_all_combined.iloc[0]
        summary['best_by_combined_score'] = {
            'name': best_combined['exp_name'],
            'psnr': float(best_combined['psnr_mean']),
            'ssim': float(best_combined['ssim_mean']),
            'combined_score': float(best_combined['combined_score']),
            'type': 'ensemble' if 'ensemble' in best_combined['exp_name'] else 'single'
        }
        if 'decoder_params' in best_combined and pd.notna(best_combined['decoder_params']):
            summary['best_by_combined_score']['decoder_params'] = int(best_combined['decoder_params'])
    
    # Save summary
    with open(report_dir / f'summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(json.dumps(summary, indent=2))
    
    print(f"\n[SAVED] All comparison reports in: {report_dir}")
    print(f"[SAVED] Files generated:")
    print(f"  - all_results_{timestamp}.json")
    print(f"  - *_by_psnr_{timestamp}.csv")
    print(f"  - *_by_ssim_{timestamp}.csv")
    print(f"  - *_by_combined_{timestamp}.csv")
    print(f"  - decoder_comparison_{timestamp}.csv")
    print(f"  - architecture_comparison_{timestamp}.csv")
    print(f"  - fusion_comparison_{timestamp}.csv")
    print(f"  - summary_{timestamp}.json")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_device():
    """
    Detect and return the best available compute device.
    
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
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
    else:
        device = 'cpu'
        print("[DEVICE] CPU (Warning: This will be very slow)")
    
    return device


def parse_arguments():
    """
    Parse and validate command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
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

The comparison report includes:
  - Rankings by PSNR (primary quality metric)
  - Rankings by SSIM (structural similarity)
  - Rankings by combined score (normalized PSNR + SSIM)
  - Decoder architecture comparison
  - Base architecture comparison
  - Fusion strategy comparison
        """
    )
    
    parser.add_argument('--mode', type=str, choices=['all', 'single'],
                        help='Evaluation mode')
    
    parser.add_argument('--arch', type=str,
                        choices=['resnet34', 'vgg16', 'vit_small_patch16_224', 
                                'pvt_v2_b2', 'ensemble'],
                        help='Architecture name')
    
    parser.add_argument('--layer', type=str,
                        help='Layer/block/stage name')
    
    parser.add_argument('--decoder', type=str,
                        choices=['frequency_aware', 'wavelet', 'transposed_conv', 'attention'],
                        help='Decoder type')
    
    parser.add_argument('--fusion', type=str,
                        choices=['attention', 'concat', 'weighted'],
                        help='Fusion strategy (for ensemble)')
    
    return parser.parse_args()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    set_random_seeds(RANDOM_SEED)
    env_info = log_environment_info()
    args = parse_arguments()
    
    if len(sys.argv) == 1:
        print("\n" + "="*80)
        print("IMAGE RECONSTRUCTION MODEL EVALUATION")
        print("="*80)
        print("\nNo arguments provided. Choose an evaluation mode:")
        print("\n1. SINGLE MODEL EVALUATION:")
        print("   python evaluate.py --arch vgg16 --layer block1 --decoder transposed_conv")
        print("\n2. ENSEMBLE MODEL EVALUATION:")
        print("   python evaluate.py --arch ensemble --fusion attention --decoder simple")
        print("\n3. BATCH EVALUATION:")
        print("   python evaluate.py --mode all")
        print("\nFor detailed help:")
        print("   python evaluate.py --help")
        print("="*80 + "\n")
        sys.exit(0)
    
    if args.mode == 'all':
        print("\n[MODE] Batch evaluation - will evaluate all trained models")
        evaluate_all_experiments()
        
    elif args.mode == 'single' or (args.arch and args.decoder):
        print("\n[MODE] Single evaluation")
        
        if not args.arch or not args.decoder:
            print("[ERROR] Single evaluation requires --arch and --decoder arguments")
            sys.exit(1)
        
        if args.arch == 'ensemble':
            if not args.fusion:
                print("[ERROR] Ensemble architecture requires --fusion argument")
                sys.exit(1)
            evaluate_single_experiment(args.arch, args.layer, args.decoder, args.fusion)
        else:
            if not args.layer:
                print("[ERROR] Single architecture requires --layer argument")
                sys.exit(1)
            evaluate_single_experiment(args.arch, args.layer, args.decoder)
    
    else:
        print("[ERROR] Invalid arguments provided")
        print("Use --help for usage information")
        sys.exit(1)
    
    print("\n[COMPLETE] Evaluation finished successfully")

