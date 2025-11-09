"""
Generate reconstruction visualizations from trained models.

Usage:
    python scripts/generate_reconstructions.py --architecture resnet34
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
from models import FeatureExtractor, create_decoder
from dataset import get_dataloaders
from utils import save_comparison_grid


def generate_reconstructions(architecture, layers, device='auto'):
    """Generate reconstruction visualizations for all trained layers."""
    
    # Auto-detect device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"\nUsing device: {device}")
    
    # Load validation data (just need a few images)
    _, val_loader = get_dataloaders(
        data_dir='data/DIV2K_train_HR',
        img_size=224,
        batch_size=8,
        num_workers='auto',
        limit=None
    )
    
    # Get test images
    test_images = next(iter(val_loader))[:8].to(device)
    
    print(f"\n{'='*80}")
    print(f"GENERATING RECONSTRUCTIONS FOR {architecture.upper()}")
    print(f"{'='*80}\n")
    
    for layer_name in layers:
        print(f"\nProcessing {layer_name}...")
        
        # Create encoder
        encoder = FeatureExtractor(architecture=architecture, layer_name=layer_name)
        encoder.to(device)
        encoder.eval()
        
        # Get feature shape
        with torch.no_grad():
            dummy_features = encoder(torch.randn(1, 3, 224, 224).to(device))
            feat_shape = dummy_features.shape
        
        # Create decoder
        decoder = create_decoder(
            decoder_type='attention',
            input_channels=feat_shape[1],
            input_size=feat_shape[2],
            output_size=224
        )
        decoder.to(device)
        decoder.eval()
        
        # Load trained weights
        checkpoint_path = f"results/{architecture}/checkpoints/{architecture}_{layer_name}_attention_best.pth"
        
        if not Path(checkpoint_path).exists():
            print(f"  ⚠️  Checkpoint not found: {checkpoint_path}")
            continue
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print(f"  ✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        # Reconstruct
        with torch.no_grad():
            features = encoder(test_images)
            reconstructed = decoder(features)
        
        # Save comparison
        config = {
            'architecture': architecture,
            'layer_name': layer_name,
            'decoder_type': 'attention'
        }
        
        save_comparison_grid(test_images, reconstructed, config, num_images=8)
        print(f"  ✓ Saved reconstruction visualization")
    
    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate reconstruction visualizations')
    
    parser.add_argument('--architecture', type=str, default='resnet34',
                       choices=['resnet34', 'vgg16', 'vit_base_patch16_224'],
                       help='Architecture to generate for')
    
    parser.add_argument('--layers', nargs='+', 
                       default=['layer1', 'layer2', 'layer3', 'layer4'],
                       help='Layers to generate for')
    
    args = parser.parse_args()
    
    generate_reconstructions(args.architecture, args.layers)


if __name__ == '__main__':
    main()