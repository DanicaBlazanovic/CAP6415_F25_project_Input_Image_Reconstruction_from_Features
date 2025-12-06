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

This work investigates how accurately RGB images can be reconstructed from intermediate feature maps of frozen, pre-trained deep neural networks, using a feature inversion method with learned decoders trained using mean-squared error (MSE) and learned perceptual image patch similarity (LPIPS). We first establish a baseline in which an attention decoder reconstructs input images from ResNet-34 Layer1 features on DIV2K, achieving 13.53 dB PSNR, 0.376 SSIM, and 0.282 LPIPS from 56×56 feature maps. Building on this, we systematically evaluate 44 model configurations, combining 8 encoder architectures (ResNet-34, VGG-16, Vision Transformer, and Pyramid Vision Transformer at multiple depths) with 4 decoder families and 3 multi-encoder ensemble fusion strategies on the DIV2K benchmark. Decoder variants include a stride-2 transposed convolution decoder, a frequency-aware decoder with explicit low/high-frequency separation and spatial–channel attention, a wavelet-based multi-resolution decoder, and a transformer-style attention decoder, while ensemble models fuse heterogeneous features from ResNet-34, VGG-16, ViT, and PVT-v2 via weighted, concatenation, or attention mechanisms. Across all architectures and ensembles, a plain transposed convolution decoder consistently outperforms more specialized designs, with the best single model (VGG-16 Block1 + Transposed Conv) reaching 17.36 dB PSNR and 0.547 SSIM and the strongest ensemble (attention fusion + transposed conv) only marginally improving to 17.58 dB PSNR and 0.581 SSIM at roughly 4× computational cost. Relative to the ResNet-34 baseline model, this yields a gain of 3.83 dB PSNR, a 45% SSIM increase, and improved LPIPS, indicating that learned upsampling from high-resolution early features is substantially more effective than hand-crafted frequency priors, deeper semantic features, or cross-architecture ensembles for feature inversion. These findings suggest that architecturally simple decoders paired with shallow, high-spatial-resolution features provide a strong and computationally efficient solution for inverting deep representations, with implications for interpretability, visualization, and compression systems that rely on intermediate network activations.



This document presents our main experimental setup and results; for baseline comparisons and additional methodology details, see [METHODOLOGY_BASELINES.md](METHODOLOGY_BASELINES.md).


---

## Top 10 Models by PSNR Performance

The following table presents the best-performing model configurations ranked by Peak Signal-to-Noise Ratio (PSNR) on the DIV2K test set. Notably, **9 out of 10 top models utilize the Transposed Convolution or Wavelet decoder**, with ensemble architectures dominating the leaderboard. The best single-architecture model (**VGG-16 Block1 + Transposed Conv**) ranks 3rd overall, achieving competitive performance at a fraction of the computational cost. **VGG-16's high spatial resolution features (112×112) combined with the simple transposed convolution decoder proves remarkably effective**, outperforming all other single architectures and nearly matching ensemble performance. Ensemble models show consistent but marginal improvements (+0.3 dB PSNR) over the best single model, highlighting the trade-off between architectural complexity and reconstruction quality.


| Rank | Model Configuration | PSNR (dB) | SSIM | Type |
|:----:|---------------------|:---------:|:----:|:----:|
| 1 | **Ensemble (ResNet34 + VGG16 + ViT + PVT-v2) + Transposed Conv** | **17.58** | **0.581** | Ensemble |
| 2 | **VGG16 Block1 + Transposed Conv** | **17.36** | **0.547** | **Single (Best)** |
| 3 | Ensemble (ResNet34 + VGG16 + ViT + PVT-v2) + Wavelet | 17.33 | **0.582** | Ensemble |
| 4 | PVT-v2-B2 Stage1 + Wavelet | 15.86 | 0.495 | Single |
| 5 | ResNet34 Layer1 + Wavelet | 15.65 | 0.509 | Single |
| 6 | ViT Small Block1 + Attention | 15.41 | 0.445 | Single |
| — | **ResNet34 Layer1 + Attention (Baseline)** | **13.53** | **0.376** | Single |

### Key Observations
- **Best Overall**: Ensemble (ResNet34 + VGG16 + ViT + PVT-v2) with attention fusion and transposed convolution decoder (**17.58 dB PSNR, 0.581 SSIM**)
- **Best Single Model**: VGG16 Block1 + Transposed Conv (**17.36 dB**) — only **0.22 dB behind best ensemble**
- **VGG16 Dominance**: VGG16 Block1 with transposed convolution significantly outperforms ResNet34 (15.65 dB), ViT (15.41 dB), and PVT-v2 (15.86 dB) single architectures, demonstrating that **high spatial resolution (112×112) is more critical than architectural sophistication**
- **Decoder Dominance**: Transposed convolution decoder appears in **top 3 models**; wavelet decoder achieves **highest SSIM (0.582)**
- **Ensemble vs. Single**: Minimal **+1.3% PSNR improvement** from ensembling despite **4× computational cost**
- **SSIM Leader**: Ensemble concat + wavelet achieves highest SSIM (**0.582**) while maintaining competitive PSNR (17.33 dB)
- **Baseline Improvement**: Best single model (VGG16 Block1 + Transposed Conv) improves **+3.83 dB PSNR (+28%)** and **+0.171 SSIM (+45%)** over ResNet34 baseline (13.53 dB, 0.376 SSIM)
- **Decoder Ranking**: Transposed Conv ≈ Wavelet > Frequency-Aware >> Attention decoder (attention performs at baseline level)
- **Architecture Ranking (Best Single Results)**: VGG16 (17.36 dB) >> PVT-v2 (15.86 dB) > ResNet34 (15.65 dB) > ViT (15.41 dB)

---
## Visual Results

Our best model (VGG-16 Block1 + Transposed Convolution Decoder) achieves high-quality reconstructions across diverse image content:

![Reconstruction Examples](results/model_comparison.png)




*Figure: Qualitative results on DIV2K validation set. Baseline (left) vs. proposed VGG16 Block1 + Transposed Convolution (right). Each shows original, reconstructed, and difference map. Proposed method demonstrates better reconstruction quality with higher PSNR/SSIM and reduced errors.*

## Dataset

**DIV2K Dataset** [9]
- 800 high-resolution training images (~2K resolution, varying dimensions)
- Diverse natural scenes (landscapes, urban, portraits, objects)
- Resized to 224×224 for computational efficiency
- Official split: 800 train (DIV2K_train_HR), 100 validation (DIV2K_valid_HR)
- Dataset source: https://www.kaggle.com/datasets/soumikrakshit/div2k-high-resolution-images

**Data Usage in This Study:**

We split the 800 DIV2K training images into proper training and validation sets, and use the official DIV2K validation set (DIV2K_valid_HR) as our test set with publicly available ground truth:

- **Training:** 640 images (80% of DIV2K_train_HR) - used for model parameter updates
- **Validation:** 160 images (20% of DIV2K_train_HR) - used for learning rate scheduling and checkpoint selection
- **Test:** 100 images (DIV2K_valid_HR) - held-out set for final evaluation and reporting metrics

**Split Strategy:**

- Random seed 42 ensures reproducibility (set via `set_seed(42)` using NumPy)
- No overlap between training, validation, and test sets
- All three architectures (ResNet34, VGG16, ViT) use identical splits for fair comparison
- Validation set monitors training progress and prevents overfitting
- Test set provides unbiased evaluation of final model performance

This standard 640/160/100 (train/val/test) split ensures our reconstruction quality metrics reflect true generalization performance while maintaining fair cross-architecture comparisons.

**Preprocessing:**
1. Resize to 256×256 (bicubic interpolation, PyTorch default)
2. **Training:** Random crop to 224×224 + random horizontal flip (p=0.5)
3. **Validation/Test:** Center crop to 224×224 (deterministic)
4. Normalize with ImageNet statistics: $\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$
5. Denormalize before computing metrics (reverse normalization and clamp to [0,1])

---


## Hardware and Training Configuration

**Computing Environment:**
- **GPU:** NVIDIA RTX 6000 Ada Generation (48GB VRAM)
- **Driver Version:** 550.107.02
- **CUDA Version:** 12.4
- **Precision:** FP32 
- **Framework:** PyTorch 2.0+ with cuDNN backend


**Note on timing comparisons:** Reported training times are machine-specific and depend on hardware, precision settings, dataloader configuration, and cuDNN optimization. These should be interpreted as relative comparisons within this study rather than absolute performance benchmarks. Times may vary significantly on different hardware configurations.


---

## Installation


### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)


**Hardware used in this study:**
- GPU: NVIDIA RTX 6000 Ada Generation (48GB VRAM)
- CUDA: 12.4
- Driver: 550.107.02


### Setup

```bash
# Clone repository
git clone https://github.com/DanicaBlazanovic/CAP6415_F25_project_Input_Image_Reconstruction_from_Features.git
cd CAP6415_F25_project_Input_Image_Reconstruction_from_Features

# Create conda environment
conda create -n cv_final python=3.10
conda activate cv_final

# Install dependencies
pip install -r requirements.txt

# For Downloading and Formatting DIV2K dataset,use
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
# Train single experiment
python train.py --arch vgg16 --layer block1 --decoder transposed_conv

# Train ensemble
python train.py --arch ensemble --fusion attention --decoder transposed_conv

# Train all experiments
python train.py --mode all

# Evaluate single experiment
python evaluate.py --arch vgg16 --layer block1 --decoder transposed_conv

# Evaluate ensemble
python evaluate.py --arch ensemble --fusion attention --decoder transposed_conv

# Evaluate all experiments
python evaluate.py --mode all
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
│   ├── coming soon............
│  
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
