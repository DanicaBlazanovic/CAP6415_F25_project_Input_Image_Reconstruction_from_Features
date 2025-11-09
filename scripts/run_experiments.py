"""
Batch Experiment Runner for Feature Inversion Project

This script automates running multiple experiments to compare:
- Different network layers (how depth affects reconstruction)
- Different architectures (ResNet vs VGG vs ViT)
- Different decoder types (simple vs attention)

The script handles:
- Data loading (once, shared across experiments)
- Model creation and training
- Evaluation and metrics calculation
- Visualization generation
- Results aggregation and comparison

Usage:
    # Quick test with limited data
    python scripts/run_experiments.py --limit 100 --epochs 5
    
    # Run all 4 ResNet34 layers with attention decoder
    python scripts/run_experiments.py --architecture resnet34 --decoder attention
    
    # Run specific layers only
    python scripts/run_experiments.py --architecture resnet34 --layers layer1 layer2
    
    # Full training on entire dataset
    python scripts/run_experiments.py --limit none --epochs 30
    
    # See all options
    python scripts/run_experiments.py --help

Author: CAP6415 Computer Vision - Fall 2025
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add src directory to Python path so we can import our modules
# This allows us to run the script from project root
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import our custom modules
import torch
from models import FeatureExtractor, create_decoder
from dataset import get_dataloaders
from train import train_decoder
from evaluate import MetricsCalculator, save_metrics, print_metrics
from utils import (get_device, save_comparison_grid, plot_training_history, 
                   count_parameters, set_seed, save_model_info)


def run_single_experiment(config, train_loader, val_loader, test_loader, device):
    """
    Run a single complete experiment from start to finish.
    
    This is the core function that executes one experiment.
    An experiment consists of:
    1. Creating encoder (feature extractor) and decoder
    2. Training the decoder to invert features
    3. Evaluating reconstruction quality with metrics
    4. Generating and saving visualizations
    5. Saving all results and checkpoints
    
    The experiment follows the pipeline:
        Images -> Encoder -> Features -> Decoder -> Reconstructed Images
        Then compute: PSNR, SSIM, LPIPS to measure quality
    
    Args:
        config: Configuration dictionary containing:
            - architecture: 'resnet34', 'vgg16', or 'vit_base_patch16_224'
            - layer_name: Which layer to extract from (e.g., 'layer3')
            - decoder_type: 'simple' or 'attention'
            - img_size: Input image size (typically 224)
            - batch_size: Batch size for training
            - num_epochs: Number of training epochs
            - lr: Learning rate
            - weight_decay: L2 regularization strength
        
        train_loader: PyTorch DataLoader with training images (640 from DIV2K_train_HR)
        val_loader: PyTorch DataLoader with validation images (160 from DIV2K_train_HR)
        test_loader: PyTorch DataLoader with test images (100 from DIV2K_test_HR)
        device: torch.device object (cuda/mps/cpu)
        
    Returns:
        results: Dictionary with evaluation metrics
    """
    # Create unique experiment identifier
    experiment_name = f"{config['architecture']}_{config['layer_name']}_{config['decoder_type']}"
    
    # Print experiment header
    print("\n" + "="*80)
    print(f"EXPERIMENT: {experiment_name}")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Create Encoder (Feature Extractor)
    # ========================================================================
    print(f"\n[1/5] Creating encoder...")
    
    encoder = FeatureExtractor(
        architecture=config['architecture'],
        layer_name=config['layer_name']
    )
    
    print(f"  Architecture: {config['architecture']}")
    print(f"  Layer: {config['layer_name']}")
    print(f"  Parameters: {count_parameters(encoder):,} (frozen)")
    
    # Determine feature shape
    dummy_input = torch.randn(1, 3, config['img_size'], config['img_size'])
    with torch.no_grad():
        dummy_features = encoder(dummy_input)
        feat_shape = dummy_features.shape
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Feature shape: {feat_shape}")
    
    # ========================================================================
    # STEP 2: Create Decoder
    # ========================================================================
    print(f"\n[2/5] Creating decoder...")
    
    decoder = create_decoder(
        decoder_type = config['decoder_type'],
        input_channels = feat_shape[1],
        input_size = feat_shape[2],
        output_size = config['img_size']
    )
    
    print(f"  Decoder type: {config['decoder_type']}")
    print(f"  Parameters: {count_parameters(decoder):,} (trainable)")
    print(f"  Input: {feat_shape}")
    print(f"  Output: [B, 3, {config['img_size']}, {config['img_size']}]")
    
    # Save model documentation
    save_model_info(
        encoder, 
        decoder, 
        config, 
        save_path=f"results/{config['architecture']}/model_info_{experiment_name}.txt"
    )
    
    # ========================================================================
    # STEP 3: Train Decoder
    # ========================================================================
    print(f"\n[3/5] Training decoder...")
    
    start_time = time.time()
    
    history = train_decoder(
        encoder = encoder,
        decoder = decoder,
        train_loader = train_loader,
        val_loader = val_loader,
        test_loader = test_loader,
        config = config,
        device = device,
        save_dir = Path(f'results/{config["architecture"]}/')
    )
    
    training_time = time.time() - start_time
    print(f"\n  Training completed in {training_time/60:.2f} minutes")
    
    # Generate training curve plot
    plot_training_history(history, config)
    
    # ========================================================================
    # STEP 4: Evaluate Model
    # ========================================================================
    print(f"\n[4/5] Evaluating model...")
    
    # Load best model
    checkpoint_path = f"results/{config['architecture']}/checkpoints/{experiment_name}_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    print(f"  Loaded best model from epoch {checkpoint['epoch']}")
    print(f"    Training loss: {checkpoint['train_loss']:.6f}")
    print(f"    Validation loss: {checkpoint['val_loss']:.6f}")
    
    # Compute metrics on validation set
    metrics_calc = MetricsCalculator(device=device)
    
    results = metrics_calc.evaluate_model(
        encoder = encoder,
        decoder = decoder,
        dataloader = val_loader,
        device = device,
        max_batches = None
    )
    
    print_metrics(results, f"Results: {experiment_name}")
    save_metrics(results, config, save_dir='results')
    
    # ========================================================================
    # STEP 5: Generate Visualizations
    # ========================================================================
    print(f"\n[5/5] Generating visualizations...")
    
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    
    # Get test images from validation set
    test_images = next(iter(val_loader))[:8].to(device)
    
    with torch.no_grad():
        features = encoder(test_images)
        reconstructed = decoder(features)
    
    save_comparison_grid(
        test_images, 
        reconstructed, 
        config, 
        num_images=8
    )
    
    # ========================================================================
    # Experiment Complete
    # ========================================================================
    print(f"\n[COMPLETE] Experiment {experiment_name} finished!")
    print(f"  Training time: {training_time/60:.2f} minutes")
    print(f"  PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
    print(f"  SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    print(f"  LPIPS: {results['lpips_mean']:.4f} ± {results['lpips_std']:.4f}")
    print(f"\n  Files saved to: results/{config['architecture']}/")
    print(f"    - Checkpoint: checkpoints/{experiment_name}_best.pth")
    print(f"    - Training plot: figures/{experiment_name}_training.png")
    print(f"    - Reconstructions: figures/{experiment_name}_reconstruction.png")
    print(f"    - Metrics: metrics/{experiment_name}_metrics.csv")
    
    results['training_time_minutes'] = training_time / 60
    
    return results


def run_batch_experiments(args):
    """
    Run a batch of experiments based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Detect best available device
    device = get_device()
    
    # ========================================================================
    # Load Data (Once, Shared Across Experiments)
    # ========================================================================
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir = args.data_dir,
        img_size = args.img_size,
        batch_size = args.batch_size,
        num_workers = 'auto',
        limit = args.limit
    )
    
    print(f"\n  Dataset: {args.data_dir}")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Train batches per epoch: {len(train_loader)}")
    
    # ========================================================================
    # Prepare Experiment Configurations
    # ========================================================================
    experiments = []
    
    for layer in args.layers:
        experiments.append({
            'architecture': args.architecture,
            'layer_name': layer,
            'decoder_type': args.decoder,
            'img_size': args.img_size,
            'batch_size': args.batch_size,
            'num_epochs': args.epochs,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'device': device
        })
    
    print(f"\n  Total experiments to run: {len(experiments)}")
    print(f"\n  Experiments:")
    for i, exp in enumerate(experiments, 1):
        print(f"    {i}. {exp['architecture']}-{exp['layer_name']}-{exp['decoder_type']}")
    
    estimated_time = len(experiments) * 12
    print(f"\n  Estimated total time: {estimated_time:.0f} minutes ({estimated_time/60:.1f} hours)")
    
    # ========================================================================
    # Confirmation Prompt
    # ========================================================================
    if not args.no_confirm:
        response = input("\nProceed with experiments? (y/n): ")
        if response.lower() != 'y':
            print("Experiments cancelled.")
            return
    
    # ========================================================================
    # Run All Experiments
    # ========================================================================
    all_results = {}
    start_time = time.time()
    
    for i, config in enumerate(experiments, 1):
        print("\n" + "="*80)
        print(f"RUNNING EXPERIMENT {i}/{len(experiments)}")
        print("="*80)
        
        experiment_name = f"{config['architecture']}_{config['layer_name']}_{config['decoder_type']}"
        
        try:
            results = run_single_experiment(config, train_loader, val_loader, test_loader, device)
            all_results[experiment_name] = results
            
        except Exception as e:
            print(f"\n[ERROR] Experiment {experiment_name} failed with error:")
            print(f"  {type(e).__name__}: {str(e)}")
            print(f"  Skipping to next experiment...")
            
            # Save error log
            error_log_path = f"results/{config['architecture']}/error_{experiment_name}.txt"
            Path(error_log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(error_log_path, 'w') as f:
                f.write(f"Experiment: {experiment_name}\n")
                f.write(f"Error: {type(e).__name__}\n")
                f.write(f"Message: {str(e)}\n")
            
            continue
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    print(f"\nTotal time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
    print(f"Successful experiments: {len(all_results)}/{len(experiments)}")
    
    if all_results:
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        print(f"\n{'Experiment':<40} {'PSNR (dB)':<15} {'SSIM':<10} {'LPIPS':<10} {'Time (min)':<12}")
        print("-"*90)
        
        for exp_name, results in all_results.items():
            print(f"{exp_name:<40} "
                  f"{results['psnr_mean']:>6.2f} ± {results['psnr_std']:<5.2f} "
                  f"{results['ssim_mean']:<10.4f} "
                  f"{results['lpips_mean']:<10.4f} "
                  f"{results['training_time_minutes']:<12.2f}")
        
        print("\n" + "="*80)
        
        # Save combined results
        import pandas as pd
        df = pd.DataFrame(all_results).T
        
        results_path = f"results/{args.architecture}/all_experiments_summary.csv"
        Path(results_path).parent.mkdir(parents = True, exist_ok = True)
        df.to_csv(results_path)
        print(f"\nCombined results saved to: {results_path}")
        
        # Print best experiment
        best_exp = max(all_results.items(), key=lambda x: x[1]['psnr_mean'])
        print(f"\nBest experiment (by PSNR): {best_exp[0]}")
        print(f"  PSNR: {best_exp[1]['psnr_mean']:.2f} dB")
        print(f"  SSIM: {best_exp[1]['ssim_mean']:.4f}")


def main():
    """
    Main function: Parse arguments and run experiments.
    """
    parser = argparse.ArgumentParser(
        description = 'Run feature inversion experiments systematically',
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
Examples:
  # Quick test with 100 images, 5 epochs
  python scripts/run_experiments.py --limit 100 --epochs 5
  
  # Run all 4 ResNet34 layers (full dataset, 30 epochs)
  python scripts/run_experiments.py --architecture resnet34 --epochs 30 --limit none
  
  # Run specific layers only
  python scripts/run_experiments.py --layers layer1 layer2
  
  # Skip confirmation prompt
  python scripts/run_experiments.py --no-confirm

For more information, see the project README.
        """
    )
    
    # Architecture arguments
    parser.add_argument(
        '--architecture', 
        type = str, 
        default = 'resnet34',
        choices = ['resnet34', 'vgg16', 'vit_base_patch16_224'],
        help = 'Architecture to use (default: resnet34)'
    )
    
    parser.add_argument(
        '--layers', 
        nargs = '+',
        default = ['layer1', 'layer2', 'layer3', 'layer4'],
        help = 'Layers to extract from (default: all 4 ResNet layers)'
    )
    
    parser.add_argument(
        '--decoder', 
        type = str, 
        default = 'attention',
        choices = ['simple', 'attention'],
        help = 'Decoder type (default: attention)'
    )
    
    # Data arguments
    parser.add_argument(
        '--data-dir', 
        type = str, 
        default = 'data',
        help = 'Path to data directory containing DIV2K_train_HR and DIV2K_test_HR (default: data)'
    )
    
    parser.add_argument(
        '--img-size', 
        type = int, 
        default = 224,
        help = 'Image size for training (default: 224)'
    )
    
    parser.add_argument(
        '--batch-size', 
        type = int, 
        default = 8,
        help ='Batch size for training (default: 8)'
    )
    
    parser.add_argument(
        '--limit', 
        type = str, 
        default = '100',
        help = 'Limit number of training images. Use "none" for full dataset (default: 100)'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs', 
        type = int, 
        default = 20,
        help = 'Number of training epochs (default: 20)'
    )
    
    parser.add_argument(
        '--lr', 
        type = float, 
        default = 1e-3,
        help = 'Learning rate (default: 0.001)'
    )
    
    parser.add_argument(
        '--weight-decay', 
        type = float, 
        default = 1e-5,
        help = 'Weight decay for L2 regularization (default: 0.00001)'
    )
    
    # Other arguments
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help = 'Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--no-confirm', 
        action = 'store_true',
        help = 'Skip confirmation prompt before starting'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process limit argument
    if args.limit.lower() == 'none':
        args.limit = None
    else:
        args.limit = int(args.limit)
    
    # Validate layers for architecture
    if args.architecture == 'resnet34':
        valid_layers = ['layer1', 'layer2', 'layer3', 'layer4']
    elif args.architecture == 'vgg16':
        valid_layers = ['block1', 'block2', 'block3', 'block4', 'block5']
    elif args.architecture.startswith('vit'):
        valid_layers = [f'block{i}' for i in range(12)]
    
    invalid_layers = [l for l in args.layers if l not in valid_layers]
    if invalid_layers:
        print(f"Error: Invalid layers for {args.architecture}: {invalid_layers}")
        print(f"Valid layers: {valid_layers}")
        return
    
    # Run experiments
    run_batch_experiments(args)


if __name__ == '__main__':
    main()