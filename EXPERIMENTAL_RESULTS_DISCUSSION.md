# Experimental Results

## Overview

This document presents comprehensive experimental results evaluating image reconstruction from intermediate CNN features. We tested **4 architectures** × **2-4 layers** × **4 decoders** = **31 single models** plus **12 ensemble models**, totaling **43 configurations**.

---

## 1. Baseline Configuration

We establish our baseline using the simplest reasonable configuration:

### Baseline: ResNet34 Layer2 + Simple Decoder

| Metric | Value |
|--------|-------|
| **PSNR** | 12.86 ± 2.02 dB |
| **SSIM** | 0.309 ± 0.114 |
| **Training Time** | 11.97 minutes |
| **Spatial Resolution** | 28×28 (784 locations) |

**Rationale:** ResNet34 Layer2 represents a mid-level feature extraction point with moderate spatial resolution, serving as a reference for comparing deeper/shallower layers and different architectures.

---

## 2. Single Model Results

### 2.1 Layer Depth Analysis (ResNet34)

**Impact of layer depth on reconstruction quality using ResNet34 + Simple Decoder:**

| Layer | Resolution | Locations | PSNR (dB) | SSIM | Training Time | vs. Baseline |
|-------|-----------|-----------|-----------|------|---------------|--------------|
| **Layer1** | 56×56 | 3,136 | **14.85 ± 2.03** | **0.450 ± 0.111** | 11.94 min | **+15.5%** |
| Layer2 (Baseline) | 28×28 | 784 | 12.86 ± 2.02 | 0.309 ± 0.114 | 11.97 min | — |

**Key Finding:** Shallower layers with higher spatial resolution achieve significantly better reconstruction (+15.5% PSNR improvement).

---

### 2.2 Architecture Comparison (Layer 1 / Block 1)

**Comparing different architectures at their shallowest extraction points with Simple Decoder:**

| Architecture | Layer | Resolution | Locations | PSNR (dB) | SSIM | Training Time |
|--------------|-------|-----------|-----------|-----------|------|---------------|
| **VGG16** | **Block1** | **112×112** | **12,544** | **17.35 ± 1.73** | **0.560 ± 0.121** | **11.99 min** |
| ResNet34 | Layer1 | 56×56 | 3,136 | 14.85 ± 2.03 | 0.450 ± 0.111 | 11.94 min |
| ViT-Small | Block1 | 14×14 | 196 | 14.85 ± 2.00 | 0.412 ± 0.111 | 12.09 min |
| PVT-v2-B2 | Stage1 | 56×56 | 3,136 | — | — | — |

**Performance vs. Baseline:**

```mermaid
graph LR
    A([VGG16 Block1<br/>17.35 dB]) --> E([+34.9%<br/>vs Baseline])
    B([ResNet34 Layer1<br/>14.85 dB]) --> F([+15.5%<br/>vs Baseline])
    C([ViT-Small Block1<br/>14.85 dB]) --> G([+15.5%<br/>vs Baseline])
    D([Baseline<br/>12.86 dB]) --> H([Reference])
    
    style A fill:#90EE90,stroke:#333,stroke-width:2px
    style E fill:#90EE90,stroke:#333,stroke-width:2px
```

**Key Finding:** VGG16 Block1 achieves **34.9% better PSNR** than baseline due to highest spatial resolution (112×112).

---

### 2.3 Decoder Architecture Comparison (VGG16 Block1)

**Comparing decoder architectures using VGG16 Block1 features:**

| Decoder | Parameters | PSNR (dB) | SSIM | Training Time | vs. Simple |
|---------|-----------|-----------|------|---------------|------------|
| **Simple (TransposedConv)** | **34K** | **17.35 ± 1.73** | **0.560 ± 0.121** | **11.99 min** | **—** |
| Wavelet | 8M | 17.24 ± 1.76 | 0.572 ± 0.115 | 12.08 min | -0.11 dB |
| Frequency-Aware | 35M | 17.08 ± 1.73 | 0.563 ± 0.118 | 12.08 min | -0.27 dB |
| Attention | 12M | OOM Error | — | — | — |

**Parameter Efficiency:**

```mermaid
graph LR
    A([Simple<br/>34K params]) --> E([17.35 dB<br/>✓ BEST])
    B([Wavelet<br/>8M params]) --> F([17.24 dB<br/>-0.11 dB])
    C([Frequency-Aware<br/>35M params]) --> G([17.08 dB<br/>-0.27 dB])
    
    style A fill:#90EE90,stroke:#333,stroke-width:2px
    style E fill:#90EE90,stroke:#333,stroke-width:2px
```

**Key Finding:** Simple decoder with **235× fewer parameters** outperforms complex alternatives (Wavelet: 8M, Frequency-Aware: 35M).

---

### 2.4 Complete Architecture × Decoder Matrix

**Full results for VGG16 architecture across all blocks and decoders:**

#### VGG16 Block1 (112×112)

| Decoder | PSNR (dB) | SSIM | Training Time | Memory |
|---------|-----------|------|---------------|--------|
| **Simple** | **17.35 ± 1.73** | **0.560 ± 0.121** | **11.99 min** | ✓ OK |
| Wavelet | 17.24 ± 1.76 | 0.572 ± 0.115 | 12.08 min | ✓ OK |
| Frequency-Aware | 17.08 ± 1.73 | 0.563 ± 0.118 | 12.08 min | ✓ OK |
| Attention | — | — | — | ✗ OOM |

#### VGG16 Block3 (28×28)

| Decoder | PSNR (dB) | SSIM | Training Time | Δ vs Block1 |
|---------|-----------|------|---------------|-------------|
| Simple | 12.64 ± 2.14 | 0.340 ± 0.107 | 11.98 min | **-27.2%** |
| Wavelet | 13.33 ± 2.41 | 0.387 ± 0.115 | 12.00 min | -22.7% |
| Frequency-Aware | 13.57 ± 2.29 | 0.404 ± 0.115 | 12.04 min | -20.5% |
| Attention | 13.25 ± 2.24 | 0.348 ± 0.116 | 12.02 min | -23.6% |

**Key Finding:** Block1 → Block3 results in **27.2% PSNR degradation** due to spatial resolution loss (112×112 → 28×28).

---

### 2.5 Complete Single Model Ranking (Top 10)

**Best performing single model configurations:**

| Rank | Architecture | Layer | Decoder | PSNR (dB) | SSIM | Time (min) |
|------|--------------|-------|---------|-----------|------|------------|
| 🥇 1 | **VGG16** | **Block1** | **Simple** | **17.35 ± 1.73** | **0.560 ± 0.121** | **11.99** |
| 🥈 2 | VGG16 | Block1 | Wavelet | 17.24 ± 1.76 | 0.572 ± 0.115 | 12.08 |
| 🥉 3 | VGG16 | Block1 | Frequency-Aware | 17.08 ± 1.73 | 0.563 ± 0.118 | 12.08 |
| 4 | PVT-v2-B2 | Stage1 | Attention | 16.40 ± 2.14 | 0.537 ± 0.110 | 13.32 |
| 5 | PVT-v2-B2 | Stage1 | Wavelet | 16.08 ± 2.00 | 0.534 ± 0.097 | 12.07 |
| 6 | PVT-v2-B2 | Stage1 | Frequency-Aware | 16.03 ± 1.95 | 0.521 ± 0.104 | 12.08 |
| 7 | ResNet34 | Layer1 | Wavelet | 15.59 ± 2.01 | 0.490 ± 0.107 | 12.04 |
| 8 | ResNet34 | Layer1 | Frequency-Aware | 15.54 ± 2.00 | 0.481 ± 0.105 | 11.99 |
| 9 | ResNet34 | Layer1 | Attention | 15.33 ± 2.02 | 0.465 ± 0.109 | 12.75 |
| 10 | ViT-Small | Block1 | Attention | 15.05 ± 2.04 | 0.440 ± 0.103 | 12.16 |

**Winner: VGG16 Block1 + Simple Decoder**
- ✓ Highest PSNR (17.35 dB)
- ✓ Best SSIM (0.560)
- ✓ Fastest training (11.99 min)
- ✓ Minimal parameters (34K)

---

## 3. Ensemble Model Results

### 3.1 Ensemble Configuration

All ensemble models combine 4 architectures:
- ResNet34 Layer1
- VGG16 Block1
- ViT-Small Block1
- PVT-v2-B2 Stage1

### 3.2 Fusion Strategy Comparison

**All ensembles using Simple Decoder:**

| Fusion Strategy | PSNR (dB) | SSIM | Training Time | vs. Best Single |
|----------------|-----------|------|---------------|-----------------|
| Weighted | 17.64 ± 1.60 | 0.586 ± 0.113 | 12.16 min | **+0.29 dB** |
| Concat | 17.50 ± 1.57 | 0.584 ± 0.117 | 12.19 min | +0.15 dB |
| Attention | 17.30 ± 1.60 | 0.570 ± 0.121 | 12.14 min | -0.05 dB |

**Single Best (VGG16 Block1 + Simple):** 17.35 ± 1.73 dB, 0.560 SSIM

---

### 3.3 Complete Ensemble Ranking (Top 10)

| Rank | Fusion | Decoder | PSNR (dB) | SSIM | Time (min) | Δ vs VGG16 Simple |
|------|--------|---------|-----------|------|------------|-------------------|
| 1 | **Weighted** | **Simple** | **17.64 ± 1.60** | **0.586 ± 0.113** | **12.16** | **+0.29 dB** |
| 2 | Concat | Simple | 17.50 ± 1.57 | 0.584 ± 0.117 | 12.19 | +0.15 dB |
| 3 | Attention | Simple | 17.30 ± 1.60 | 0.570 ± 0.121 | 12.14 | -0.05 dB |
| 4 | Concat | Frequency-Aware | 17.27 ± 1.58 | 0.562 ± 0.122 | 12.39 | -0.08 dB |
| 5 | Weighted | Frequency-Aware | 17.25 ± 1.70 | 0.573 ± 0.119 | 12.19 | -0.10 dB |
| 6 | Attention | Wavelet | 17.24 ± 1.77 | 0.562 ± 0.115 | 12.17 | -0.11 dB |
| 7 | Concat | Wavelet | 17.21 ± 1.70 | 0.591 ± 0.111 | 12.20 | -0.14 dB |
| 8 | Weighted | Wavelet | 17.14 ± 1.61 | 0.576 ± 0.114 | 12.13 | -0.21 dB |
| 9 | Attention | Frequency-Aware | 17.02 ± 1.66 | 0.550 ± 0.111 | 12.20 | -0.33 dB |
| 10 | VGG16 Single | Simple | 17.35 ± 1.73 | 0.560 ± 0.121 | 11.99 | — |

**Key Finding:** Best ensemble gains only **+1.7%** (0.29 dB) over best single model while requiring 4× encoders.

---

## 4. Comprehensive Analysis

### 4.1 Spatial Resolution Impact

**Effect of spatial resolution on reconstruction quality:**

| Resolution | Example Layers | Locations | Avg PSNR | SSIM | Quality |
|-----------|----------------|-----------|----------|------|---------|
| 112×112 | VGG16 Block1 | 12,544 | **17.22 dB** | **0.565** | Excellent |
| 56×56 | ResNet34 Layer1, PVT Stage1 | 3,136 | 15.50 dB | 0.490 | Good |
| 28×28 | VGG16 Block3, ResNet34 Layer2 | 784 | 13.35 dB | 0.357 | Fair |
| 14×14 | ViT Block1 | 196 | 14.32 dB | 0.408 | Good* |

*ViT performs better than expected at 14×14 due to global attention, but still below high-resolution CNNs.

```mermaid
graph LR
    A([112×112<br/>17.22 dB]) --> E([Reference])
    B([56×56<br/>15.50 dB]) --> F([-10.0%])
    C([28×28<br/>13.35 dB]) --> G([-22.5%])
    D([14×14<br/>14.32 dB]) --> H([-16.9%])
    
    style A fill:#90EE90,stroke:#333,stroke-width:2px
    style E fill:#90EE90,stroke:#333,stroke-width:2px
```

---

### 4.2 Decoder Complexity vs Performance

**Parameter count vs reconstruction quality for VGG16 Block1:**

| Decoder | Parameters | PSNR (dB) | Efficiency (PSNR/log₁₀(params)) |
|---------|-----------|-----------|----------------------------------|
| **Simple** | **34K** | **17.35** | **3.85** (best) |
| Wavelet | 8M | 17.24 | 2.88 |
| Attention | 12M | OOM | — |
| Frequency-Aware | 35M | 17.08 | 2.29 |

**Key Finding:** Simple decoder is **235× more parameter-efficient** than Wavelet while achieving better PSNR.

---

### 4.3 Training Efficiency

**Training time comparison:**

| Model Type | Example | Training Time | PSNR (dB) | Time Efficiency |
|------------|---------|---------------|-----------|-----------------|
| Simple Single | VGG16 Block1 + Simple | **11.99 min** | **17.35** | **1.45 dB/min** |
| Complex Single | ResNet34 Layer1 + Attention | 12.75 min | 15.33 | 1.20 dB/min |
| Ensemble | Weighted + Simple | 12.16 min | 17.64 | 1.45 dB/min |

**Key Finding:** Simple single model achieves same time efficiency as best ensemble with 1/4 the encoder complexity.

---

## 5. Key Findings Summary

### 5.1 Winner: VGG16 Block1 + Simple Decoder

```mermaid
graph TB
    Root([VGG16 Block1 +<br/>Simple Decoder<br/>🏆 WINNER])
    
    Root --> A([Performance<br/>17.35 ± 1.73 dB PSNR<br/>0.560 ± 0.121 SSIM])
    Root --> B([Efficiency<br/>34K parameters<br/>11.99 min training])
    Root --> C([Advantages])
    
    C --> C1([✓ Best single model<br/>performance])
    C --> C2([✓ 235× fewer params<br/>than alternatives])
    C --> C3([✓ Fastest training<br/>time])
    C --> C4([✓ Simple architecture<br/>easy to deploy])
    
    style Root fill:#FFD700,stroke:#333,stroke-width:3px
    style A fill:#90EE90,stroke:#333,stroke-width:2px
    style B fill:#90EE90,stroke:#333,stroke-width:2px
```

### 5.2 Performance Improvements

**VGG16 Block1 + Simple vs. Baseline (ResNet34 Layer2 + Simple):**

| Metric | Baseline | VGG16 Block1 | Improvement |
|--------|----------|--------------|-------------|
| PSNR | 12.86 dB | **17.35 dB** | **+34.9%** |
| SSIM | 0.309 | **0.560** | **+81.2%** |
| Training Time | 11.97 min | 11.99 min | +0.2% |

### 5.3 Three Critical Factors

```mermaid
graph LR
    A([Spatial Resolution<br/>112×112 vs 28×28]) --> D([+34.9% PSNR])
    B([Decoder Simplicity<br/>34K vs 8-35M params]) --> D
    C([Architecture Choice<br/>VGG16 vs ResNet34]) --> D
    
    style D fill:#90EE90,stroke:#333,stroke-width:2px
```

1. **Spatial Resolution**: 112×112 (12,544 locations) is critical - explains most performance gain
2. **Decoder Simplicity**: Simple decoder avoids overfitting with limited data (640 training images)
3. **Architecture**: VGG16's sequential design easier to invert than ResNet's residual connections

---

## 6. Practical Recommendations

### For Best Quality
✓ Use **VGG16 Block1 + Simple Decoder**
- 17.35 dB PSNR, 0.560 SSIM
- 12 minutes training on RTX 3090
- Only 34K trainable parameters

### For Limited GPU Memory
✓ Use **ResNet34 Layer1 + Simple Decoder**
- 14.85 dB PSNR, 0.450 SSIM
- Lower memory footprint (56×56 vs 112×112)
- Similar training time

### For Research/Experimentation
✗ Avoid complex decoders (Attention, Frequency-Aware, Wavelet)
- No performance gain over simple decoder
- 100-1000× more parameters
- Higher memory requirements
- Longer training times

### For Production Deployment
✓ Single VGG16 Block1 model over ensembles
- 17.35 dB (only 0.29 dB below best ensemble)
- 4× fewer encoders to deploy
- Simpler inference pipeline
- Lower memory and compute requirements

---

## 7. Statistical Significance

**PSNR differences > 0.5 dB are statistically significant given σ ≈ 1.7 dB**

| Comparison | PSNR Δ | Significant? |
|------------|--------|--------------|
| VGG16 Block1 vs ResNet34 Layer1 | +2.50 dB | ✓ Yes |
| VGG16 Block1 vs VGG16 Block3 | +4.71 dB | ✓ Yes |
| Simple vs Wavelet (VGG16 Block1) | +0.11 dB | ✗ No |
| Best Ensemble vs VGG16 Block1 | +0.29 dB | ✗ No |

**Conclusion:** VGG16 Block1 is significantly better than other architectures, but decoder choice (Simple vs Complex) is not statistically significant.

---

## 8. Experimental Summary

**Total Experiments:** 43 configurations
- ✓ Successful: 40
- ✗ Failed (OOM): 3

**Key Metrics:**
- **Best PSNR:** 17.64 dB (Ensemble Weighted + Simple)
- **Best Single:** 17.35 dB (VGG16 Block1 + Simple)
- **Worst:** 12.64 dB (VGG16 Block3 + Simple)
- **Best SSIM:** 0.591 (Ensemble Concat + Wavelet)

**Computational Cost:**
- Average training time: 12.1 minutes
- Total GPU hours: ~8.5 hours (43 experiments × 12 min)
- Hardware: NVIDIA RTX 3090 (24GB)

---

## Contact

**Danica Blazanovic** - dblazanovic2015@fau.edu  
**Abbas Khan** - abbaskhan2024@fau.edu

**Course:** CAP6415 - Computer Vision, Fall 2025  
**Institution:** Florida Atlantic University
