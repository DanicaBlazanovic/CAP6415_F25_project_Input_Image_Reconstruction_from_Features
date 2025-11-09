"""
Results Visualization Script

Generates all comparison plots and saves them with timestamped filenames.
Automatically creates visualizations from experiment results.

Usage:
    python scripts/visualize_results.py --architecture resnet34
    python scripts/visualize_results.py --architecture vgg16
    python scripts/visualize_results.py --help
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def create_timestamp():
    """
    Create timestamp string for unique filenames.
    
    Format: YYYYMMDD_HHMMSS
    Example: 20241105_153045
    
    This ensures visualizations don't overwrite each other.
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def load_results(architecture):
    """
    Load experiment results from CSV file.
    
    Args:
        architecture: Architecture name (e.g., 'resnet34')
        
    Returns:
        df: Pandas DataFrame with results
    """
    # Path to results file
    results_path = f'results/{architecture}/all_experiments_summary.csv'
    
    # Check if file exists
    if not Path(results_path).exists():
        print(f"Error: Results file not found at {results_path}")
        print("Run experiments first!")
        sys.exit(1)
    
    # Load CSV
    df = pd.read_csv(results_path, index_col=0)
    
    # FIXED: Extract layer names with correct method
    # Pattern: resnet34_layer1_attention → extract "layer1"
    df['layer'] = df.index.str.split('_').str[1]
    
    print(f"Loaded {len(df)} experiments from {results_path}")
    print(f"Layers: {df['layer'].tolist()}")
    
    return df


def plot_quality_metrics(df, architecture, timestamp, save_dir):
    """
    Create bar charts comparing PSNR, SSIM, and LPIPS across layers.
    
    This shows which layer achieves best reconstruction quality.
    
    Args:
        df: Results DataFrame
        architecture: Architecture name
        timestamp: Timestamp string for filename
        save_dir: Directory to save figure
    """
    # Create figure with 3 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Subplot 1: PSNR (higher is better)
    axes[0].bar(df['layer'], df['psnr_mean'], color='steelblue', alpha=0.8, edgecolor='black')
    axes[0].errorbar(df['layer'], df['psnr_mean'], yerr=df['psnr_std'], 
                     fmt='none', ecolor='black', capsize=5, linewidth=2)
    axes[0].set_xlabel('Layer', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    axes[0].set_title('Peak Signal-to-Noise Ratio\n(Higher = Better)', 
                      fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(df.iterrows()):
        axes[0].text(i, row['psnr_mean'] + 0.1, f"{row['psnr_mean']:.2f}", 
                     ha='center', fontsize=10, fontweight='bold')
    
    # Subplot 2: SSIM (higher is better)
    axes[1].bar(df['layer'], df['ssim_mean'], color='coral', alpha=0.8, edgecolor='black')
    axes[1].errorbar(df['layer'], df['ssim_mean'], yerr=df['ssim_std'], 
                     fmt='none', ecolor='black', capsize=5, linewidth=2)
    axes[1].set_xlabel('Layer', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('SSIM', fontsize=12, fontweight='bold')
    axes[1].set_title('Structural Similarity Index\n(Higher = Better)', 
                      fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (idx, row) in enumerate(df.iterrows()):
        axes[1].text(i, row['ssim_mean'] + 0.01, f"{row['ssim_mean']:.3f}", 
                     ha='center', fontsize=10, fontweight='bold')
    
    # Subplot 3: LPIPS (lower is better - use inverted colors)
    axes[2].bar(df['layer'], df['lpips_mean'], color='lightcoral', alpha=0.8, edgecolor='black')
    axes[2].errorbar(df['layer'], df['lpips_mean'], yerr=df['lpips_std'], 
                     fmt='none', ecolor='black', capsize=5, linewidth=2)
    axes[2].set_xlabel('Layer', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('LPIPS', fontsize=12, fontweight='bold')
    axes[2].set_title('Learned Perceptual Similarity\n(Lower = Better)', 
                      fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (idx, row) in enumerate(df.iterrows()):
        axes[2].text(i, row['lpips_mean'] + 0.02, f"{row['lpips_mean']:.3f}", 
                     ha='center', fontsize=10, fontweight='bold')
    
    # Overall title
    fig.suptitle(f'{architecture.upper()} - Quality Metrics Comparison', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save with timestamp
    filename = f"{architecture}_quality_metrics_{timestamp}.png"
    save_path = save_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] Quality metrics: {save_path}")


def plot_efficiency_analysis(df, architecture, timestamp, save_dir):
    """
    Create scatter plot showing parameter efficiency.
    
    Shows relationship between model size (parameters) and quality (PSNR).
    Reveals which layer is most parameter-efficient.
    
    Args:
        df: Results DataFrame
        architecture: Architecture name
        timestamp: Timestamp string
        save_dir: Save directory
    """
    # Parameter counts for each architecture/layer
    # ResNet34 parameter counts from model_info files
    params_resnet = {
        'layer1': 250_147,
        'layer2': 974_563,
        'layer3': 3_865_187,
        'layer4': 15_413_603
    }
    
    # VGG16 parameter counts (approximate, update with actual values)
    params_vgg = {
        'block1': 200_000,
        'block2': 500_000,
        'block3': 1_000_000,
        'block4': 2_000_000,
        'block5': 4_000_000
    }
    
    # Map parameters to dataframe based on architecture
    if 'resnet' in architecture:
        df['params'] = df['layer'].map(params_resnet)
    elif 'vgg' in architecture:
        df['params'] = df['layer'].map(params_vgg)
    else:
        # For ViT or unknown architectures, use placeholder
        df['params'] = 1_000_000
    
    df['params_millions'] = df['params'] / 1_000_000
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot with size based on SSIM
    scatter = ax.scatter(df['params_millions'], df['psnr_mean'], 
                        s=df['ssim_mean'] * 1000,  # Size proportional to SSIM
                        c=df['ssim_mean'], cmap='RdYlGn', 
                        alpha=0.7, edgecolors='black', linewidth=3)
    
    # Add layer labels
    for idx, row in df.iterrows():
        ax.annotate(row['layer'], 
                   (row['params_millions'], row['psnr_mean']),
                   fontsize=14, fontweight='bold',
                   ha='center', va='center')
    
    # Colorbar for SSIM
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('SSIM (Structural Similarity)', fontsize=12, fontweight='bold')
    
    # Axis labels
    ax.set_xlabel('Decoder Parameters (Millions)', fontsize=14, fontweight='bold')
    ax.set_ylabel('PSNR (dB)', fontsize=14, fontweight='bold')
    ax.set_title(f'{architecture.upper()} - Parameter Efficiency Analysis\n' + 
                 'Bubble size = SSIM | Color = SSIM | Top-left = Most efficient', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add efficiency annotation
    best_layer = df.loc[df['psnr_mean'].idxmax()]
    worst_layer = df.loc[df['psnr_mean'].idxmin()]
    
    efficiency_text = (
        f"Most Efficient: {best_layer['layer']}\n"
        f"  {best_layer['params']/1e6:.1f}M params → {best_layer['psnr_mean']:.2f} dB\n\n"
        f"Least Efficient: {worst_layer['layer']}\n"
        f"  {worst_layer['params']/1e6:.1f}M params → {worst_layer['psnr_mean']:.2f} dB"
    )
    
    ax.text(0.98, 0.02, efficiency_text,
           transform=ax.transAxes, fontsize=11,
           verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    
    # Save with timestamp
    filename = f"{architecture}_efficiency_analysis_{timestamp}.png"
    save_path = save_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] Efficiency analysis: {save_path}")


def plot_time_quality_tradeoff(df, architecture, timestamp, save_dir):
    """
    Create dual-axis bar plot showing training time vs quality.
    
    Shows the tradeoff between training time and reconstruction quality.
    
    Args:
        df: Results DataFrame
        architecture: Architecture name
        timestamp: Timestamp string
        save_dir: Save directory
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # X-axis positions
    x = np.arange(len(df))
    width = 0.35
    
    # Bar plot for training time (left axis)
    bars1 = ax.bar(x - width/2, df['training_time_minutes'], width, 
                   label='Training Time', color='skyblue', alpha=0.8, edgecolor='black')
    
    # Add value labels on time bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}m',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Create second y-axis for PSNR (right axis)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, df['psnr_mean'], width, 
                    label='PSNR', color='salmon', alpha=0.8, edgecolor='black')
    
    # Add value labels on PSNR bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}dB',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Configure axes
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Training Time (minutes)', fontsize=14, fontweight='bold', color='skyblue')
    ax2.set_ylabel('PSNR (dB)', fontsize=14, fontweight='bold', color='salmon')
    ax.set_title(f'{architecture.upper()} - Training Time vs Quality Tradeoff', 
                 fontsize=16, fontweight='bold')
    
    # Set x-tick labels
    ax.set_xticks(x)
    ax.set_xticklabels(df['layer'], fontsize=12)
    
    # Color y-tick labels to match bars
    ax.tick_params(axis='y', labelcolor='skyblue', labelsize=12)
    ax2.tick_params(axis='y', labelcolor='salmon', labelsize=12)
    
    # Add legends
    ax.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save with timestamp
    filename = f"{architecture}_time_quality_tradeoff_{timestamp}.png"
    save_path = save_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] Time-quality tradeoff: {save_path}")


def plot_metrics_heatmap(df, architecture, timestamp, save_dir):
    """
    Create heatmap showing normalized metrics across all layers.
    
    All metrics are normalized to 0-1 scale where higher = better.
    Provides quick visual comparison across all metrics simultaneously.
    
    Args:
        df: Results DataFrame
        architecture: Architecture name
        timestamp: Timestamp string
        save_dir: Save directory
    """
    # Prepare data for heatmap
    heatmap_data = df[['psnr_mean', 'ssim_mean', 'lpips_mean']].copy()
    
    # Invert LPIPS (lower is better → higher is better)
    heatmap_data['lpips_inverted'] = 1 - heatmap_data['lpips_mean']
    heatmap_data = heatmap_data.drop('lpips_mean', axis=1)
    
    # Set index to layer names
    heatmap_data.index = df['layer']
    
    # Rename columns for display
    heatmap_data.columns = ['PSNR', 'SSIM', 'Perceptual Quality']
    
    # Normalize to 0-1 scale for better visualization
    scaler = MinMaxScaler()
    heatmap_normalized = pd.DataFrame(
        scaler.fit_transform(heatmap_data),
        columns=heatmap_data.columns,
        index=heatmap_data.index
    )
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use RdYlGn colormap (Red-Yellow-Green: bad-medium-good)
    sns.heatmap(heatmap_normalized.T, annot=True, fmt='.3f', 
                cmap='RdYlGn', cbar_kws={'label': 'Normalized Score (0-1)'},
                linewidths=3, linecolor='white', ax=ax,
                vmin=0, vmax=1, center=0.5,
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    
    ax.set_title(f'{architecture.upper()} - Normalized Metrics Heatmap\n' + 
                 '(All metrics scaled to 0-1, higher = better)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=14, fontweight='bold')
    
    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    
    plt.tight_layout()
    
    # Save with timestamp
    filename = f"{architecture}_metrics_heatmap_{timestamp}.png"
    save_path = save_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] Metrics heatmap: {save_path}")


def generate_summary_table(df, architecture, timestamp, save_dir):
    """
    Generate formatted summary table and save as image.
    
    Creates a professional table showing all key metrics.
    
    Args:
        df: Results DataFrame
        architecture: Architecture name
        timestamp: Timestamp string
        save_dir: Save directory
    """
    # Select columns for table
    table_data = df[['layer', 'psnr_mean', 'psnr_std', 'ssim_mean', 
                     'ssim_std', 'lpips_mean', 'training_time_minutes']].copy()
    
    # Format columns
    table_data['PSNR (dB)'] = table_data.apply(
        lambda x: f"{x['psnr_mean']:.2f} ± {x['psnr_std']:.2f}", axis=1)
    table_data['SSIM'] = table_data.apply(
        lambda x: f"{x['ssim_mean']:.4f} ± {x['ssim_std']:.4f}", axis=1)
    table_data['LPIPS'] = table_data['lpips_mean'].apply(lambda x: f"{x:.4f}")
    table_data['Time (min)'] = table_data['training_time_minutes'].apply(lambda x: f"{x:.1f}")
    
    # Select final columns
    final_table = table_data[['layer', 'PSNR (dB)', 'SSIM', 'LPIPS', 'Time (min)']]
    final_table.columns = ['Layer', 'PSNR (dB)', 'SSIM', 'LPIPS ↓', 'Time (min)']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, len(df) * 0.8 + 2))
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=final_table.values,
                    colLabels=final_table.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15, 0.25, 0.25, 0.15, 0.15])
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
    # Header styling
    for i in range(len(final_table.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=14)
    
    # Row colors (alternating)
    for i in range(1, len(final_table) + 1):
        for j in range(len(final_table.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    # Add title
    plt.suptitle(f'{architecture.upper()} - Results Summary', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Add note
    note = "↓ Lower is better | Green header = Metric categories"
    plt.figtext(0.5, 0.02, note, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # Save with timestamp
    filename = f"{architecture}_summary_table_{timestamp}.png"
    save_path = save_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] Summary table: {save_path}")


def main():
    """Main function with argument parsing."""
    
    parser = argparse.ArgumentParser(
        description='Generate visualizations from experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate visualizations for ResNet34 results
  python scripts/visualize_results.py --architecture resnet34
  
  # Generate for VGG16
  python scripts/visualize_results.py --architecture vgg16
  
  # Specify custom output directory
  python scripts/visualize_results.py --architecture resnet34 --output results/resnet34/visualizations
        """
    )
    
    parser.add_argument(
        '--architecture', 
        type=str, 
        required=True,
        choices=['resnet34', 'vgg16', 'vit_base_patch16_224'],
        help='Architecture to visualize results for'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: results/{architecture}/visualizations)'
    )
    
    args = parser.parse_args()
    
    # Set up output directory
    if args.output:
        save_dir = Path(args.output)
    else:
        save_dir = Path(f'results/{args.architecture}/visualizations')
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for unique filenames
    timestamp = create_timestamp()
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    print(f"Architecture: {args.architecture}")
    print(f"Output directory: {save_dir}")
    print(f"Timestamp: {timestamp}")
    print("="*80 + "\n")
    
    # Load results
    df = load_results(args.architecture)
    
    # Generate all plots
    print("\nGenerating plots...")
    
    plot_quality_metrics(df, args.architecture, timestamp, save_dir)
    plot_efficiency_analysis(df, args.architecture, timestamp, save_dir)
    plot_time_quality_tradeoff(df, args.architecture, timestamp, save_dir)
    plot_metrics_heatmap(df, args.architecture, timestamp, save_dir)
    generate_summary_table(df, args.architecture, timestamp, save_dir)
    plot_layer_comparison(args.architecture, timestamp, save_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nAll visualizations saved to: {save_dir}")
    print(f"\nGenerated files (timestamped with {timestamp}):")
    print(f"  1. {args.architecture}_quality_metrics_{timestamp}.png")
    print(f"  2. {args.architecture}_efficiency_analysis_{timestamp}.png")
    print(f"  3. {args.architecture}_time_quality_tradeoff_{timestamp}.png")
    print(f"  4. {args.architecture}_metrics_heatmap_{timestamp}.png")
    print(f"  5. {args.architecture}_summary_table_{timestamp}.png")
    print("\n" + "="*80 + "\n")


def plot_layer_comparison(architecture, timestamp, save_dir):
    """
    Create vertical comparison of reconstructions from all layers.
    
    Args:
        architecture: Architecture name
        timestamp: Timestamp string
        save_dir: Save directory
    """
    from PIL import Image, ImageDraw, ImageFont
    
    # Find all reconstruction images
    figures_dir = Path(f'results/{architecture}/figures')
    reconstruction_files = sorted(figures_dir.glob(f'{architecture}_layer*_reconstruction.png'))
    
    if not reconstruction_files:
        print(f"[WARNING] No reconstruction images found in {figures_dir}")
        return
    
    print(f"Found {len(reconstruction_files)} reconstruction images")
    
    # Load images
    images = [Image.open(f) for f in reconstruction_files]
    
    # Get dimensions
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    
    max_width = max(widths)
    label_height = 60  # Height for each label
    total_height = sum(heights) + len(images) * label_height + 100  # Title space
    
    # Create new blank image (white background)
    combined = Image.new('RGB', (max_width, total_height), 'white')
    draw = ImageDraw.Draw(combined)
    
    # Try to use a nice font, fall back to default if not available
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
    
    # Add main title
    title = f'{architecture.upper()} - Layer-by-Layer Reconstruction Comparison'
    title_bbox = draw.textbbox((0, 0), title, font=font_title)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(((max_width - title_width) // 2, 20), title, fill='black', font=font_title)
    
    # Paste images vertically with labels
    y_offset = 100  # Start after title
    
    for i, (img, filepath) in enumerate(zip(images, reconstruction_files)):
        # Extract layer name
        layer_name = filepath.stem.split('_')[1].upper()  # e.g., "LAYER1"
        
        # Draw label above image
        label_text = f'{layer_name} (Original top | Reconstructed bottom)'
        draw.text((20, y_offset + 10), label_text, fill='black', font=font_label)
        y_offset += label_height
        
        # Paste image
        combined.paste(img, (0, y_offset))
        y_offset += img.height + 20  # 20px gap between images
    
    # Save directly using PIL (no matplotlib needed)
    filename = f"{architecture}_layer_comparison_{timestamp}.png"
    save_path = save_dir / filename
    combined.save(save_path, dpi=(300, 300))
    
    print(f"[SAVED] Layer comparison: {save_path}")


if __name__ == '__main__':
    main()