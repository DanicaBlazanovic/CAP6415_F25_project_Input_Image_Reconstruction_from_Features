# Input Image Reconstruction from Features: Cross-Architecture Analysis


**Team Members:** Danica Blazanovic, Abbas Khan

---

## Table of Contents

- [Abstract](#abstrackt)
- [Top 10 Models by PSNR Performance](#Top-10-Models-by-PSNR-Performance)
- [Key Observations](#key-observations)
- [Dataset](#dataset)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)


---

## Abstract
This work presents a comprehensive investigation into reconstructing RGB images from intermediate feature representations of frozen pre-trained deep neural networks. We systematically evaluate 44 model configurations comprising 8 encoder architectures (ResNet-34, VGG-16, Vision Transformer, Pyramid Vision Transformer at various layers) with 4 decoder strategies and 3 ensemble fusion approaches on the DIV2K dataset.
Our decoder architectures include: (1) Transposed Convolution Decoder - sequential stride-2 transposed convolutions with batch normalization, (2) Frequency-Aware Decoder - explicit low/high-frequency separation with spatial-channel attention, (3) Wavelet Frequency Decoder - multi-resolution wavelet sub-band prediction, and (4) Attention Decoder - transformer blocks with self-attention mechanisms. Ensemble models fuse features from multiple architectures using attention-based, concatenation-based, or weighted strategies.
Contrary to expectations that complex domain-specific decoders would excel, our results demonstrate that the simple Transposed Convolution Decoder consistently outperforms all alternatives, achieving the highest reconstruction quality across every architecture. This reveals that unconstrained data-driven learning of upsampling patterns outperforms hand-crafted inductive biases for this inverse problem.
The best single model, VGG-16 Block1 + Transposed Convolution Decoder, achieves 17.35 dB PSNR and 0.560 SSIM. Multi-architecture ensemble fusion marginally improves to 17.64 dB PSNR and 0.586 SSIM (+1.7% gain, 4× cost), revealing significant feature redundancy. Critically, VGG-16 block1's high spatial resolution (112×112) proves more valuable than deeper semantic features or architectural diversity.
Our findings demonstrate that architectural simplicity combined with high-resolution feature extraction is optimal for neural reconstruction, with important implications for interpretability, visualization, and compression applications requiring intermediate feature inversion.

---

## Top 10 Models by PSNR Performance

The following table presents the best-performing model configurations ranked by Peak Signal-to-Noise Ratio (PSNR) on the DIV2K test set. Notably, **9 out of 10 top models utilize the Transposed Convolution or Wavelet decoder**, with ensemble architectures dominating the leaderboard. The best single-architecture model (**VGG-16 Block1 + Transposed Conv**) ranks 3rd overall, achieving competitive performance at a fraction of the computational cost. **VGG-16's high spatial resolution features (112×112) combined with the simple transposed convolution decoder proves remarkably effective**, outperforming all other single architectures and nearly matching ensemble performance. Ensemble models show consistent but marginal improvements (+0.3 dB PSNR) over the best single model, highlighting the trade-off between architectural complexity and reconstruction quality.

| Rank | Model Configuration | PSNR (dB) | SSIM | Val Loss | Type |
|:----:|---------------------|:---------:|:----:|:--------:|:----:|
| 1 | **Ensemble All Weighted + Transposed Conv** | **17.64** | 0.586 | 0.478 | Ensemble |
| 2 | **Ensemble All Concat + Transposed Conv** | **17.50** | 0.584 | 0.473 | Ensemble |
| 3 | **VGG-16 Block1 + Transposed Conv** | **17.35** | 0.560 | 0.502 | **Single** |
| 4 | Ensemble All Attention + Transposed Conv | 17.30 | 0.570 | 0.478 | Ensemble |
| 5 | Ensemble All Concat + Frequency-Aware | 17.27 | 0.562 | 0.474 | Ensemble |
| 6 | Ensemble All Weighted + Frequency-Aware | 17.25 | 0.573 | 0.477 | Ensemble |
| 7 | VGG-16 Block1 + Wavelet | 17.24 | 0.572 | 0.482 | Single |
| 8 | Ensemble All Attention + Wavelet | 17.24 | 0.562 | 0.480 | Ensemble |
| 9 | Ensemble All Concat + Wavelet | 17.21 | **0.591** | 0.479 | Ensemble |
| 10 | Ensemble All Weighted + Wavelet | 17.14 | 0.576 | 0.481 | Ensemble |

### Key Observations

-  **Best Overall**: Ensemble weighted fusion with transposed convolution decoder (**17.64 dB PSNR**)
-  **Best Single Model**: VGG-16 Block1 + Transposed Conv (**17.35 dB**) — only **0.29 dB behind best ensemble**
-  **VGG-16 Dominance**: VGG-16 Block1 with transposed convolution significantly outperforms ResNet, ViT, and PVT single architectures, demonstrating that **high spatial resolution (112×112) is more critical than architectural sophistication**
-  **Decoder Dominance**: Transposed convolution appears in **4 of top 4 models**; wavelet decoder in **4 of remaining 6**
-  **Ensemble vs. Single**: Minimal **+1.7% PSNR improvement** from ensembling despite **4× computational cost**
-  **SSIM Leader**: Ensemble concat + wavelet achieves highest SSIM (**0.591**) despite ranking 9th in PSNR










## Hardware and Training Configuration

**Computing Environment:**
- **GPU:** NVIDIA A100 (40GB) via Google Colab Pro for VGG16 block1; standard Colab GPU (16GB) for all other experiments
- **CPU:** Mac M1 for local development and testing
- **Precision:** FP32 (no mixed precision)
- **Framework:** PyTorch 2.0+ with cuDNN backend
- **Batch size:** 8 (standard layers), 1 (VGG16 block1 due to memory constraints)

**Note on timing comparisons:** Reported training times are machine-specific and depend on hardware, precision settings, dataloader configuration, and cuDNN optimization. These should be interpreted as relative comparisons within this study rather than absolute performance benchmarks. Times may vary significantly on different hardware configurations.

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or Apple Silicon (MPS)
- 16GB+ RAM for most experiments
- 40GB+ GPU RAM for VGG16 block1 (use Google Colab Pro with A100)
- ~5GB disk space for dataset and models

### Setup
```bash
# Clone repository
git clone https://github.com/DanicaBlazanovic/CAP6415_F25_project_Input_Image_Reconstruction_from_Features.git
cd CAP6415_F25_project_Input_Image_Reconstruction_from_Features
 scripts/download_dataset.py


# Create conda environment
conda create -n cv_final python=3.10
conda activate cv_final

# Install dependencies
pip install -r requirements.txt

# Download DIV2K dataset
python scripts/download_dataset.py
```

### Dependencies

See `requirements.txt` for complete list. Key dependencies:
```txt
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
numpy>=1.23.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
lpips>=0.1.4
pandas>=1.5.0
seaborn>=0.12.0
```

---

## Usage

### 1. Run Complete Experiment Suite
```bash
# ResNet34 - all layers (recommended: standard GPU)
python scripts/run_experiments.py --architecture resnet34 --epochs 30 --limit none --no-confirm

# VGG16 - blocks 2-5 (standard GPU)
python scripts/run_experiments.py --architecture vgg16 --layers block2 block3 block4 block5 --epochs 30 --limit none --no-confirm

# VGG16 block1 - requires 40GB GPU (use Google Colab Pro)
python scripts/run_experiments.py --architecture vgg16 --layers block1 --epochs 30 --limit none --batch-size 1 --no-confirm

# ViT - selected blocks
python scripts/run_experiments.py --architecture vit_base_patch16_224 --layers block0 block1 block5 block8 block11 --epochs 30 --limit none --no-confirm

# Quick test (100 images, 5 epochs)
python scripts/run_experiments.py --architecture resnet34 --epochs 5 --limit 100
```

**Key Arguments:**
- `--architecture`: Choose from `resnet34`, `vgg16`, `vit_base_patch16_224`
- `--layers`: Specify layers (default: all layers for that architecture)
- `--epochs`: Number of training epochs (default: 20)
- `--limit`: Number of images to use (`none` for full dataset)
- `--batch-size`: Batch size (default: 8, use 1 for VGG16 block1)
- `--decoder`: Decoder type (`simple` or `attention`, default: `attention`)
- `--no-confirm`: Skip confirmation prompt

### 2. Generate Visualizations
```bash
# Generate reconstruction visualizations
python scripts/generate_reconstructions.py --architecture resnet34

# Generate all analysis plots
python scripts/visualize_results.py --architecture resnet34

# For all architectures
for arch in resnet34 vgg16 vit_base_patch16_224; do
    python scripts/generate_reconstructions.py --architecture $arch
    python scripts/visualize_results.py --architecture $arch
done
```

### 3. Analyze Results (Interactive)
```bash
# Launch Jupyter Lab
jupyter lab

# Open notebooks/analyze_results.ipynb
```

---

## Project Structure
```
CV_Final_Project/
├── data/
│   ├── DIV2K_train_HR/          # Training dataset (800 images)
│   └── DIV2K_test_HR/           # Test dataset (100 images from DIV2K_valid_HR)
│
├── src/
│   ├── models.py                 # Encoder and decoder architectures
│   ├── dataset.py                # Data loading and preprocessing
│   ├── train.py                  # Training loop implementation
│   ├── evaluate.py               # Metrics calculation (PSNR, SSIM, LPIPS)
│   └── utils.py                  # Helper functions and visualization
│
├── scripts/
│   ├── train.py                  # Main experiment runner
│   ├── evaluate.py               # Evaluate 
│
│
├── results/
│   ├── resnet34/
│   │   ├── checkpoints/          # Trained model weights (.pth files)
│   │   ├── figures/              # Reconstruction comparison images
│   │   ├── metrics/              # CSV files with numerical results
│   │   └── all_experiments_summary.csv  # Combined results
│   │
│   ├── vgg16/
│   │   ├── checkpoints/          # Baseline MSE-trained models
│   │   ├── figures/              # Standard reconstruction visualizations
│   │   ├── figures_perceptual/   # Perceptual loss experiment visualizations
│   │   ├── figures_adversarial/  # GAN-based experiment visualizations
│   │   ├── metrics/              # CSV files with numerical results
│   │   └── ...                   # Same structure for VGG16
│   │
│   └── vit_base_patch16_224/
│       └── ...                   # Same structure for ViTor ViT
│
├── requirements.txt              # Python dependencies
├── .gitignore                 # Git ignore patterns
└── README.md                    # This file
```



## Contact

**Danica Blazanovic** - dblazanovic2015@fau.edu  
**Abbas Khan** - abbaskhan2024@fau.edu

**Course:** CAP6415 - Computer Vision  
**Institution:** Florida Atlantic University  
**Semester:** Fall 2025

---

## License

This project is for academic purposes as part of CAP6415 coursework.

---

## Acknowledgments

- DIV2K dataset creators for high-quality training data
- PyTorch and timm library maintainers for excellent deep learning tools
- LPIPS authors for the perceptual similarity metric implementation

---

**Last Updated:** December 2, 2025
