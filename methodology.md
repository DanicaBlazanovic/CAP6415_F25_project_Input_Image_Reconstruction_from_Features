# Methodology

## Overview

This study investigates the reconstruction of input images from intermediate convolutional neural network (CNN) features extracted by pre-trained architectures. We employ a transfer learning approach consisting of two primary components: (1) a frozen pre-trained encoder that extracts intermediate feature representations, and (2) a trainable decoder that learns to invert these features back to the original RGB images. The central objective is to determine how much visual information is preserved at different depths within CNN architectures and to identify optimal layer-decoder configurations for high-fidelity image reconstruction.

---

## Experimental Framework

Our methodology follows a standard feature inversion pipeline where pre-trained convolutional networks serve as feature extractors and lightweight decoder networks learn the inverse mapping from feature space to image space. Figure 1 illustrates the complete pipeline.

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT IMAGE (224×224×3)                      │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              FROZEN PRE-TRAINED ENCODER (VGG16)                  │
│                   • ImageNet-1K Pre-trained                      │
│                   • No gradient updates                          │
│                   • Evaluation mode                              │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│            INTERMEDIATE FEATURES (C × H × W)                     │
│               VGG16 Block1: (64 × 112 × 112)                     │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│         TRAINABLE DECODER (Transposed Convolution)               │
│                   • Progressive Upsampling                       │
│                   • Batch Normalization                          │
│                   • ReLU Activation                              │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│           RECONSTRUCTED IMAGE (224×224×3)                        │
│                   • Sigmoid Activation                           │
│                   • Pixel Values [0, 1]                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LOSS COMPUTATION                            │
│              L = 0.5 × MSE + 0.5 × LPIPS                        │
└─────────────────────────────────────────────────────────────────┘
```
**Figure 1:** Complete image reconstruction pipeline showing the flow from input image through frozen encoder, feature extraction, trainable decoder, and loss computation.

---

## 3.1 Feature Extraction Architecture

### 3.1.1 Encoder Selection: VGG16

We selected VGG16 (Simonyan & Zisserman, 2015) as our primary feature extraction backbone. VGG16 is a deep convolutional neural network consisting of 16 weight-bearing layers organized into five sequential blocks. Each block contains 2-3 convolutional layers with 3×3 kernels followed by 2×2 max-pooling for spatial downsampling. The architecture demonstrates a systematic design philosophy: doubling the number of feature channels while halving spatial dimensions at each stage.

**Architectural Specifications:**
- **Depth:** 13 convolutional layers + 3 fully connected layers
- **Convolutional kernel:** Uniform 3×3 throughout all layers
- **Pooling:** 2×2 max-pooling with stride 2
- **Activation:** ReLU non-linearity
- **Pre-training:** ImageNet-1K (1.2M images, 1,000 object categories)
- **Top-1 accuracy:** 71.6% on ImageNet validation set

VGG16 was selected for three primary reasons: (1) its simple, interpretable sequential architecture facilitates analysis of layer-wise feature preservation, (2) aggressive max-pooling at each block creates distinct spatial resolutions ideal for studying reconstruction quality degradation, and (3) its widespread adoption provides strong pre-trained representations and enables comparison with prior work.

### 3.1.2 Layer Selection: Block1

We extract intermediate features from **Block1** of VGG16, specifically the output immediately following the second convolutional layer and subsequent max-pooling operation. This layer selection is motivated by both theoretical considerations and empirical observations.

**Block1 Architecture:**
```
Input: (B, 3, 224, 224)
  ↓
Conv2d(3 → 64, kernel=3×3, padding=1) + ReLU
  ↓  Output: (B, 64, 224, 224)
Conv2d(64 → 64, kernel=3×3, padding=1) + ReLU
  ↓  Output: (B, 64, 224, 224)
MaxPool2d(kernel=2×2, stride=2)
  ↓
Output: (B, 64, 112, 112)
```

**Rationale for Block1 Selection:**

1. **Maximal Spatial Resolution:**  
   Block1 produces 112×112 feature maps, representing 12,544 spatial locations. This is the highest resolution output among all VGG16 blocks and surpasses other architectures tested in our study (ResNet34 layer1: 56×56; ViT: 14×14). Higher spatial resolution preserves fine-grained spatial information critical for pixel-level reconstruction.

2. **Low-Level Feature Representation:**  
   Early convolutional layers in CNNs are known to capture low-level visual patterns including edges, corners, textures, and color gradients (Zeiler & Fergus, 2014). These features maintain strong correspondence with input image structure, making inversion more tractable than deeper semantic features.

3. **Minimal Semantic Abstraction:**  
   With only 64 feature channels and two convolutional layers, Block1 representations undergo limited semantic transformation. This minimal abstraction preserves information about spatial arrangement, color distribution, and texture details that are progressively lost in deeper layers.

4. **Empirical Performance:**  
   Across multiple experimental configurations, Block1 consistently achieves superior reconstruction quality (PSNR: 17.35 dB, SSIM: 0.560) compared to deeper blocks, validating the hypothesis that spatial resolution dominates reconstruction fidelity.

### 3.1.3 Encoder Configuration

All encoder parameters remain frozen throughout the training process. Specifically, we set `requires_grad=False` for all VGG16 parameters and maintain the encoder in evaluation mode (`model.eval()`). This configuration ensures:

1. **Parameter Freezing:** No gradient computation or weight updates occur in the encoder during backpropagation
2. **Batch Normalization:** No running statistics updates (although VGG16 does not contain batch normalization layers)
3. **Dropout Deactivation:** Any dropout layers remain inactive (not applicable to VGG16)
4. **Deterministic Behavior:** Feature extraction remains consistent across training iterations

**Transfer Learning Justification:**  
Freezing pre-trained encoder weights serves multiple purposes: (1) preserves high-quality ImageNet-learned representations, (2) prevents catastrophic forgetting of pre-trained features, (3) reduces computational cost by eliminating encoder gradient computation, and (4) focuses optimization exclusively on the decoder, which is the primary object of study.

---

## 3.2 Image Reconstruction Architecture

### 3.2.1 Primary Decoder: Transposed Convolution Network

We employ a **Transposed Convolution Decoder** as our primary reconstruction network. Despite its architectural simplicity compared to alternative designs, this decoder achieves superior empirical performance across all evaluation metrics. The decoder implements a progressive upsampling strategy with batch normalization and non-linear activations.

**Decoder Architecture (VGG16 Block1 Configuration):**

```
Input Features: (B, 64, 112, 112)
  ↓
┌─────────────────────────────────────────┐
│ ConvTranspose2d(64 → 32)                │
│   kernel=4×4, stride=2, padding=1       │
│   Output: (B, 32, 224, 224)             │
└─────────────────────────────────────────┘
  ↓ Spatial Upsampling: 112×112 → 224×224
┌─────────────────────────────────────────┐
│ BatchNorm2d(32)                         │
└─────────────────────────────────────────┘
  ↓ Normalization
┌─────────────────────────────────────────┐
│ ReLU(inplace=True)                      │
└─────────────────────────────────────────┘
  ↓ Non-linearity
┌─────────────────────────────────────────┐
│ Conv2d(32 → 3)                          │
│   kernel=3×3, padding=1                 │
│   Output: (B, 3, 224, 224)              │
└─────────────────────────────────────────┘
  ↓ Channel Projection: 32 → 3 (RGB)
┌─────────────────────────────────────────┐
│ Sigmoid()                               │
│   Output Range: [0, 1]                  │
└─────────────────────────────────────────┘
  ↓
Output: Reconstructed Image (B, 3, 224, 224)
```

**Layer-by-Layer Description:**

1. **Transposed Convolution Layer:**  
   The core upsampling operation uses a 4×4 transposed convolution with stride 2 and padding 1. This configuration achieves exact 2× spatial upsampling according to the formula:
   
   $$\text{output\_size} = (\text{input\_size} - 1) \times \text{stride} - 2 \times \text{padding} + \text{kernel\_size}$$
   
   For our configuration: output = (112 - 1) × 2 - 2 + 4 = 224

   Transposed convolution (also known as deconvolution or fractionally-strided convolution) performs learnable upsampling by inserting zeros between input pixels and applying standard convolution. This enables the network to learn optimal interpolation patterns rather than using fixed methods like bilinear upsampling.

2. **Batch Normalization:**  
   Following the transposed convolution, we apply batch normalization (Ioffe & Szegedy, 2015) to stabilize training dynamics. Batch normalization normalizes activations to zero mean and unit variance across the batch dimension, reducing internal covariate shift and enabling higher learning rates. The operation is defined as:
   
   $$\hat{x} = \frac{x - \mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}} + \epsilon}}$$
   
   $$y = \gamma \hat{x} + \beta$$
   
   where μ_B and σ²_B are batch statistics, ε is a small constant for numerical stability, and γ, β are learnable affine parameters.

3. **ReLU Activation:**  
   We apply the Rectified Linear Unit (ReLU) activation function element-wise:
   
   $$\text{ReLU}(x) = \max(0, x)$$
   
   ReLU introduces non-linearity while maintaining computational efficiency. The `inplace=True` parameter optimizes memory usage by modifying tensors in-place.

4. **Final Convolution:**  
   A standard 3×3 convolution projects the 32-channel feature representation to 3 RGB channels. This layer performs spatial smoothing while mapping to the output color space. The 3×3 kernel with padding=1 preserves spatial dimensions.

5. **Sigmoid Activation:**  
   The final sigmoid activation function bounds pixel values to the [0, 1] range:
   
   $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
   
   This ensures reconstructed images have valid intensity values matching the expected pixel range after denormalization.

**Parameter Analysis:**

The transposed convolution decoder maintains a minimal parameter footprint:
- Transposed convolution: 64 × 32 × 4 × 4 = 32,768 parameters
- Batch normalization: 32 × 2 = 64 parameters (scale and shift)
- Final convolution: 32 × 3 × 3 × 3 = 864 parameters
- **Total trainable parameters: ~34,000**

This represents approximately 0.025% of the encoder parameters (~138M in VGG16), demonstrating that effective reconstruction requires minimal decoder complexity when feature extraction is high-quality.

### 3.2.2 Alternative Decoder Architectures

To validate our architectural choices, we implemented and evaluated three additional decoder variants with increasing complexity. These alternatives incorporate explicit frequency modeling, wavelet decomposition, and attention mechanisms respectively.

#### Frequency-Aware Decoder

The frequency-aware decoder explicitly separates feature processing into low and high frequency pathways, motivated by the observation that neural networks implicitly learn multi-scale representations. This architecture attempts to make frequency decomposition explicit through parallel processing branches.

**Architecture Overview:**
```
Input Features
  ↓
┌─────────────────┬─────────────────┐
│  Low Frequency  │  High Frequency │
│    Pathway      │     Pathway     │
│  (3×3 convs)    │   (3×3 convs)   │
│  Smoothness     │   Fine details  │
└─────────────────┴─────────────────┘
  ↓                      ↓
  │ C/2 channels    │ C/2 channels
  │                     │
  └──────────┬──────────┘
             ↓
    Frequency Fusion Block
    (Channel + Spatial Attention)
             ↓
  Spatial-Channel Attention (×2)
             ↓
    Progressive Upsampling
    (Residual Blocks)
             ↓
    Multi-Scale Refinement
    (3×3, 5×5, 7×7 parallel)
             ↓
    Color Enhancement
    (Instance Normalization)
             ↓
    RGB Output
```

**Key Components:**

1. **Dual Frequency Pathways:**  
   Input features are processed through two parallel branches, each containing 3×3 convolutions. The low-frequency pathway is designed to capture smooth color transitions and overall structure, while the high-frequency pathway focuses on edges and texture details. Each pathway reduces feature channels by half (C → C/2).

2. **Frequency Fusion:**  
   The fusion module combines low and high frequency features using both channel attention (which frequency bands are important) and spatial attention (which locations are important). This is implemented through squeeze-and-excitation style channel attention followed by spatial attention using averaged and max-pooled feature statistics.

3. **Multi-Scale Refinement:**  
   Parallel 3×3, 5×5, and 7×7 convolutions process features at different receptive field scales. The multi-scale outputs are concatenated and combined with learned attention weights, allowing the network to adaptively select appropriate scales for different image regions.

4. **Color Enhancement:**  
   Instance normalization and learned color transforms adjust the color distribution of reconstructed images to improve naturalness and vibrancy.

**Parameters:** ~35M trainable parameters (1000× larger than transposed convolution decoder)

**Empirical Results:**  
Despite its sophistication, the frequency-aware decoder does not outperform the simple transposed convolution baseline. This suggests that explicit frequency modeling may introduce inductive biases that conflict with optimal reconstruction pathways, or that the network implicitly learns appropriate frequency decompositions without architectural constraints.

#### Wavelet Frequency Decoder

The wavelet decoder is motivated by wavelet transform theory, which represents images as combinations of multi-resolution frequency subbands. This architecture explicitly predicts four wavelet components (approximation and detail coefficients) which are subsequently fused for reconstruction.

**Architecture Overview:**
```
Input Features
  ↓
┌────────┬────────┬────────┬────────┐
│   LL   │   LH   │   HL   │   HH   │
│ (Low-  │ (Low-  │ (High- │ (High- │
│  Low)  │ High)  │  Low)  │ High)  │
│Approx  │Horiz   │Vert    │Diag    │
│64 ch   │64 ch   │64 ch   │64 ch   │
└────────┴────────┴────────┴────────┘
  ↓        ↓        ↓        ↓
  └────────┴────────┴────────┘
             ↓
    Concatenate (256 channels)
             ↓
    Frequency Band Fusion (×2)
    (Channel Attention)
             ↓
    Frequency Refinement
    (256 → 128 channels)
             ↓
    Progressive Upsampling
             ↓
    Multi-Scale Refinement
             ↓
    Color Enhancement
             ↓
    RGB Output
```

**Wavelet Component Prediction:**

The decoder predicts four standard wavelet subbands:
- **LL (Low-Low):** Approximation subband containing overall image structure and smooth variations
- **LH (Low-High):** Horizontal detail subband capturing horizontal edges
- **HL (High-Low):** Vertical detail subband capturing vertical edges
- **HH (High-High):** Diagonal detail subband capturing corners and diagonal structures

Each subband is predicted through an independent 3-layer CNN that transforms input features into 64-channel representations. The four subbands are concatenated (256 total channels) and progressively fused through attention-weighted combination.

**Parameters:** ~8M trainable parameters

**Empirical Results:**  
The wavelet decoder achieves moderate performance but does not surpass the transposed convolution baseline. This indicates that explicit wavelet decomposition, while theoretically motivated, does not provide practical advantages when neural networks can learn implicit multi-resolution representations through standard convolutions.

#### Attention Decoder

The attention decoder applies Transformer-style self-attention mechanisms (Vaswani et al., 2017; Dosovitskiy et al., 2021) to captured features before upsampling. This architecture enables global context aggregation where each spatial location can attend to all other locations.

**Architecture Overview:**
```
Input Features (B, C, H, W)
  ↓
Reshape to Sequence: (B, H×W, C)
  ↓
┌─────────────────────────────┐
│  Transformer Block 1        │
│  • Multi-Head Attention     │
│  • LayerNorm                │
│  • Feed-Forward MLP         │
│  • Residual Connections     │
└─────────────────────────────┘
  ↓
┌─────────────────────────────┐
│  Transformer Block 2        │
└─────────────────────────────┘
  ↓
  ... [Repeat 4× total]
  ↓
┌─────────────────────────────┐
│  Transformer Block 4        │
└─────────────────────────────┘
  ↓
Reshape to Spatial: (B, C, H, W)
  ↓
Progressive Transposed Conv Upsampling
  ↓
RGB Output
```

**Self-Attention Mechanism:**

Multi-head self-attention computes relationships between all spatial locations simultaneously. For each location (query), the mechanism computes attention weights over all other locations (keys) and aggregates their values:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

where Q, K, V are query, key, and value matrices derived through learned linear projections, and d_k is the key dimension used for scaling.

**Computational Complexity:**  
Self-attention has quadratic complexity in the number of spatial locations: O(N²) where N = H × W. For VGG16 Block1 features (112×112), this results in N = 12,544 spatial locations, requiring approximately 157 million pairwise attention computations per layer.

**Parameters:** ~12M trainable parameters

**Empirical Results:**  
Despite enabling global context modeling, the attention decoder does not outperform the transposed convolution baseline and incurs significantly higher computational cost. This suggests that local spatial relationships, naturally captured by convolutions, are more critical for pixel-level reconstruction than long-range dependencies.

### 3.2.3 Decoder Selection Rationale

Across all experimental configurations, the simple transposed convolution decoder consistently achieves the best reconstruction quality (PSNR: 17.35 dB with VGG16 Block1). This counterintuitive result—wherein the simplest architecture outperforms more complex alternatives—can be attributed to several factors:

1. **Overfitting Mitigation:**  
   With only 640 training images, complex decoders (35M parameters) risk overfitting despite regularization. The minimal decoder (34K parameters) maintains better generalization.

2. **Architectural Flexibility:**  
   Complex decoders impose strong inductive biases (frequency separation, wavelet structure, global attention) that may misalign with optimal reconstruction paths. The simple decoder allows unconstrained learning of the inverse mapping.

3. **Gradient Flow:**  
   Simple architectures enable more direct gradient pathways without complex fusion, attention, or hierarchical processing bottlenecks. This facilitates more efficient optimization.

4. **Occam's Razor:**  
   Given equal or better performance, simpler models are preferable. The transposed convolution decoder provides the optimal balance of capacity and simplicity for this task.

These findings align with recent observations in deep learning that architectural complexity does not necessarily improve performance and may even harm it when training data is limited (Hooker et al., 2019).

---

## 3.3 Training Protocol

### 3.3.1 Dataset and Preprocessing

**Dataset:**  
We utilize the DIV2K dataset (Agustsson & Timofte, 2017), a high-resolution image dataset originally created for super-resolution tasks. DIV2K contains 800 training images and 100 validation images featuring diverse content including natural scenes, objects, textures, and structures.

**Data Partitioning:**
- Training set: 640 images (80% of original 800 training images)
- Validation set: 160 images (20% of original 800 training images)  
- Test set: 100 images (original DIV2K validation set, held out)

This partitioning ensures sufficient training data while maintaining independent validation and test sets for unbiased performance evaluation.

**Preprocessing Pipeline:**

All images undergo a standardized preprocessing sequence:

1. **Resizing:** Images are initially resized to 256×256 pixels using bilinear interpolation, establishing a uniform scale.

2. **Cropping:** From 256×256 images, we extract 224×224 patches:
   - **Training:** Random cropping (data augmentation)
   - **Validation/Test:** Deterministic center cropping (reproducibility)

3. **Normalization:** Pixel values are normalized using ImageNet statistics to match the encoder's pre-training distribution:
   - Mean: [0.485, 0.456, 0.406] (R, G, B channels)
   - Standard deviation: [0.229, 0.224, 0.225] (R, G, B channels)
   
   $$x_{\text{normalized}} = \frac{x - \mu}{\sigma}$$

**Data Augmentation (Training Only):**
- **Random horizontal flip:** Applied with probability 0.5 to double effective dataset size
- **Random crop:** 224×224 patches extracted from random locations within 256×256 images

Data augmentation is applied exclusively during training to improve generalization and prevent overfitting. Validation and test sets use deterministic preprocessing to ensure reproducible evaluation.

### 3.3.2 Loss Function

We employ a hybrid loss function combining pixel-level reconstruction accuracy with perceptual similarity:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{MSE}} \cdot \mathcal{L}_{\text{MSE}} + \lambda_{\text{LPIPS}} \cdot \mathcal{L}_{\text{LPIPS}}$$

where λ_MSE = λ_LPIPS = 0.5, weighting both components equally.

#### Mean Squared Error (MSE) Loss

The MSE loss measures pixel-wise L2 distance between original and reconstructed images:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{x}_i)^2$$

where x_i represents original pixels, x̂_i represents reconstructed pixels, and N is the total number of pixels (224 × 224 × 3 = 150,528 per image).

**Properties:**
- Directly optimizes Peak Signal-to-Noise Ratio (PSNR), a standard image quality metric
- Encourages sharp, pixel-accurate reconstructions
- Treats all spatial locations and channels equally
- Computationally efficient (closed-form gradient)

**Limitations:**
- Poor correlation with human perceptual similarity
- Can result in blurry reconstructions when averaged across multiple plausible solutions
- Ignores structural and textural coherence

#### Learned Perceptual Image Patch Similarity (LPIPS) Loss

LPIPS (Zhang et al., 2018) measures perceptual similarity by comparing deep feature representations extracted by pre-trained networks:

$$\mathcal{L}_{\text{LPIPS}} = \sum_{l \in \mathcal{L}} w_l \|\phi_l(x) - \phi_l(\hat{x})\|_2^2$$

where:
- φ_l(·) denotes features from layer l of a pre-trained network (AlexNet in our implementation)
- w_l are learned per-layer weights
- L is the set of selected layers

**Properties:**
- Aligns with human perceptual judgments of image similarity
- Captures structural and textural similarity beyond pixel-level alignment
- Prevents common artifacts (e.g., checkerboard patterns, unnatural colors)
- Learned weights emphasize perceptually-relevant feature dimensions

**Rationale for Hybrid Loss:**

Neither MSE nor LPIPS alone produces optimal results:
- MSE alone: Achieves high PSNR but may generate perceptually unnatural images
- LPIPS alone: Produces perceptually plausible results but may lack fine detail

The hybrid formulation balances pixel-level accuracy (MSE) with perceptual naturalness (LPIPS), leveraging complementary strengths. The equal weighting (0.5, 0.5) was determined through preliminary experiments and aligns with common practice in image generation tasks.

### 3.3.3 Optimization Configuration

**Optimizer:**  
We employ the Adam optimizer (Kingma & Ba, 2014) with the following hyperparameters:
- Initial learning rate: α = 1×10⁻⁴
- First moment decay: β₁ = 0.9
- Second moment decay: β₂ = 0.999
- Numerical stability constant: ε = 1×10⁻⁸
- Weight decay: 0 (no explicit L2 regularization)

Adam was selected for its adaptive learning rate properties and robust performance across diverse architectures. The relatively small learning rate (10⁻⁴) ensures stable training given the small batch size constraint.

**Learning Rate Scheduling:**  
We implement ReduceLROnPlateau scheduling to adapt learning rates based on validation performance:
- **Monitor:** Validation loss
- **Reduction factor:** 0.5 (halve learning rate upon plateau)
- **Patience:** 5 epochs without improvement
- **Minimum learning rate:** 1×10⁻⁷ (prevent excessive reduction)

This adaptive schedule allows initially rapid optimization while fine-tuning later stages through progressive learning rate reduction.

**Early Stopping:**  
To prevent overfitting and reduce computational cost:
- **Monitor:** Validation loss
- **Patience:** 15 epochs without improvement
- **Checkpoint restoration:** Restore parameters from best validation epoch

**Training Hyperparameters:**
- Maximum epochs: 30
- Batch size: 4 images (GPU memory constraint with VGG16 Block1's high resolution)
- Gradient clipping: None (Adam's adaptive rates provide sufficient stability)
- Number of workers: 4 (data loading parallelization)

**Computational Resources:**
- **GPU:** NVIDIA RTX 3090 / A100 (24GB VRAM)
- **Training duration:** 30-70 minutes per model configuration
- **Storage:** ~500MB per model checkpoint

### 3.3.4 Training Algorithm

```
Algorithm: Feature Inversion Training

Input: Frozen encoder f_θ, trainable decoder g_ψ, training set D_train
Output: Optimized decoder parameters ψ*

1:  Initialize decoder parameters ψ using Kaiming initialization
2:  Set encoder f_θ to evaluation mode (frozen, no gradient computation)
3:  best_val_loss ← ∞
4:  patience_counter ← 0
5:  
6:  for epoch = 1 to 30 do
7:      // Training Phase
8:      for each mini-batch {x_i} ∈ D_train do
9:          // Forward pass (no gradients for encoder)
10:         with torch.no_grad():
11:             z_i ← f_θ(x_i)              // Extract features
12:         
13:         x̂_i ← g_ψ(z_i)                  // Reconstruct images
14:         
15:         // Compute hybrid loss
16:         L_MSE ← (1/N) Σ(x_i - x̂_i)²
17:         L_LPIPS ← LPIPS(x_i, x̂_i)      // AlexNet-based
18:         L_total ← 0.5 × L_MSE + 0.5 × L_LPIPS
19:         
20:         // Backward pass (gradients only for decoder)
21:         L_total.backward()
22:         optimizer.step()                // Update ψ using Adam
23:         optimizer.zero_grad()
24:     end for
25:     
26:     // Validation Phase
27:     val_loss ← 0
28:     for each mini-batch {x_j} ∈ D_val do
29:         with torch.no_grad():
30:             z_j ← f_θ(x_j)
31:             x̂_j ← g_ψ(z_j)
32:             val_loss += compute_loss(x_j, x̂_j)
33:     end for
34:     val_loss ← val_loss / |D_val|
35:     
36:     // Learning rate scheduling
37:     if val_loss has not improved for 5 epochs then
38:         learning_rate ← learning_rate × 0.5
39:     end if
40:     
41:     // Early stopping and checkpointing
42:     if val_loss < best_val_loss then
43:         best_val_loss ← val_loss
44:         save_checkpoint(ψ)              // Save best model
45:         patience_counter ← 0
46:     else
47:         patience_counter ← patience_counter + 1
48:     end if
49:     
50:     if patience_counter ≥ 15 then
51:         break                           // Early termination
52:     end if
53: end for
54:
55: restore_checkpoint(best_val_loss)      // Load best weights
56: return ψ*
```

**Key Implementation Details:**

1. **Gradient Management:**  
   We explicitly disable gradient computation for the encoder using `torch.no_grad()` context, reducing memory consumption and computational cost.

2. **Checkpointing Strategy:**  
   Only the best-performing model (lowest validation loss) is retained, preventing overfitting to training data while ensuring optimal generalization.

3. **Numerical Stability:**  
   Images are clamped to [0, 1] range before metric computation to prevent numerical instabilities in loss calculations.

---

## 3.4 Evaluation Methodology

We assess reconstruction quality using three complementary metrics that capture different aspects of image fidelity. All metrics are computed on the test set (100 DIV2K validation images) using images denormalized to [0, 1] pixel range.

### 3.4.1 Peak Signal-to-Noise Ratio (PSNR)

PSNR quantifies reconstruction accuracy through the ratio of maximum possible pixel value to mean squared error:

$$\text{PSNR} = 10 \log_{10} \left(\frac{\text{MAX}^2}{\text{MSE}}\right) = 20 \log_{10} \left(\frac{\text{MAX}}{\sqrt{\text{MSE}}}\right)$$

where MAX = 1.0 (maximum pixel value in [0, 1] normalized images).

**Interpretation:**
- **Range:** 0 to ∞ decibels (dB), higher values indicate better quality
- **Typical ranges:**
  - < 20 dB: Poor quality, severe degradation
  - 20-30 dB: Fair quality, noticeable artifacts
  - 30-40 dB: Good quality, minor artifacts
  - > 40 dB: Excellent quality, near-perfect reconstruction

**Properties:**
- Mathematically simple and widely adopted in image processing
- Directly inversely related to MSE loss
- Scale-invariant (independent of absolute pixel values)

**Limitations:**
- Weak correlation with human perceptual quality assessments
- Treats all spatial locations uniformly without considering local structure
- Sensitive to small misalignments even when perceptually acceptable

### 3.4.2 Structural Similarity Index (SSIM)

SSIM (Wang et al., 2004) evaluates perceived image quality by comparing luminance, contrast, and structure:

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

where:
- μ_x, μ_y: Mean intensities (luminance)
- σ_x², σ_y²: Variances (contrast)
- σ_xy: Covariance (structure correlation)
- C₁, C₂: Stabilization constants to avoid division by zero

**Computation:**
SSIM is computed locally within sliding 11×11 pixel windows and averaged across the entire image.

**Interpretation:**
- **Range:** -1 to 1 (typically 0 to 1 for natural images)
- **Values:** 
  - 1.0: Perfect structural similarity
  - > 0.9: Excellent structural preservation
  - 0.7-0.9: Good structural similarity
  - < 0.7: Poor structural preservation

**Properties:**
- Better correlation with human perception than PSNR
- Incorporates luminance, contrast, and structural information
- Relatively robust to uniform brightness/contrast changes
- Symmetric metric: SSIM(x, y) = SSIM(y, x)

**Limitations:**
- Computationally more expensive than PSNR (sliding window)
- Window size selection affects results
- May not capture all perceptual quality dimensions

### 3.4.3 Learned Perceptual Image Patch Similarity (LPIPS)

LPIPS (Zhang et al., 2018) measures perceptual distance by comparing deep features from a pre-trained network:

$$\text{LPIPS}(x, \hat{x}) = \sum_{l=1}^{L} w_l \|\phi_l(x) - \phi_l(\hat{x})\|_2^2$$

where φ_l represents features from layer l of AlexNet, and w_l are learned weights that emphasize perceptually-relevant layers.

**Implementation:**
- **Network:** AlexNet pre-trained on ImageNet
- **Layers:** conv1 through conv5 (5 layers total)
- **Normalization:** Features are channel-wise normalized before comparison
- **Aggregation:** Spatial averaging followed by weighted summation across layers

**Interpretation:**
- **Range:** ≈0 to 1 (unbounded theoretically, but typically < 1)
- **Values:**
  - 0: Perceptually identical
  - < 0.1: High perceptual similarity
  - 0.1-0.3: Moderate perceptual differences
  - > 0.3: Significant perceptual dissimilarity

**Properties:**
- Strongest correlation with human perceptual judgments among standard metrics
- Captures mid-level feature differences (textures, structures)
- Learned weights emphasize human-relevant perceptual dimensions
- Robust to small spatial misalignments

**Limitations:**
- Computationally expensive (requires AlexNet forward pass)
- Dependent on pre-trained network choice (AlexNet, VGG, etc.)
- Not interpretable in absolute terms (relative comparison metric)

### 3.4.4 Evaluation Protocol

**Test Set Evaluation:**
1. Load best checkpoint (lowest validation loss)
2. Set model to evaluation mode (disable dropout, use eval batch norm)
3. For each test image:
   - Extract features using frozen encoder
   - Reconstruct using trained decoder
   - Denormalize both original and reconstructed images to [0, 1]
   - Clamp reconstructed values to [0, 1]
   - Compute PSNR, SSIM, LPIPS
4. Report mean ± standard deviation across 100 test images

**Statistical Reporting:**
- **Mean:** Central tendency of reconstruction quality
- **Standard deviation:** Variability across diverse test images
- **Min/Max:** Range of performance (reported for outlier analysis)

All reported results represent test set performance using the single best model (selected via validation loss). No test set tuning or model selection is performed to ensure unbiased evaluation.

---

## 3.5 Implementation Details

**Framework and Libraries:**
- **Deep Learning:** PyTorch 2.0+
- **Computer Vision:** torchvision, timm (PyTorch Image Models)
- **Metrics:** lpips (official implementation), scikit-image
- **Numerical:** NumPy, SciPy
- **Visualization:** matplotlib, seaborn

**Reproducibility:**
- **Random seeds:** Fixed across all experiments (seed=42)
- **Deterministic algorithms:** Enabled where possible (cuDNN deterministic mode)
- **Version control:** All code version-controlled via Git
- **Hyperparameter logging:** Weights & Biases / TensorBoard

**Code Structure:**
```
project/
├── models.py           # Encoder and decoder architectures
├── train.py           # Training loop and optimization
├── evaluate.py        # Metric computation and evaluation
├── data.py            # Dataset loading and preprocessing
├── utils.py           # Helper functions
└── configs/           # Hyperparameter configurations
```

**Hardware and Timing:**
- **GPU:** NVIDIA RTX 3090 (24GB) / A100 (40GB)
- **CPU:** AMD Ryzen 9 / Intel Xeon (16+ cores)
- **RAM:** 64GB system memory
- **Storage:** NVMe SSD for fast data loading
- **Training time:** ~30-70 minutes per configuration
- **Inference:** ~10-15 ms per image (batch size 1)

---

## 3.6 Experimental Configurations

We systematically evaluate multiple configurations across architectures, layers, and decoders:

**Primary Configuration (Best Performance):**
- Encoder: VGG16
- Layer: Block1 (64×112×112)
- Decoder: Transposed Convolution
- Result: **PSNR = 17.35 dB, SSIM = 0.560**

**Comparative Configurations:**

| Configuration | Encoder | Layer | Spatial Res. | Decoder | Parameters |
|--------------|---------|-------|--------------|---------|------------|
| **Best** | VGG16 | Block1 | 112×112 | TransposedConv | 34K |
| Alt-1 | VGG16 | Block1 | 112×112 | FrequencyAware | 35M |
| Alt-2 | VGG16 | Block1 | 112×112 | Wavelet | 8M |
| Alt-3 | VGG16 | Block1 | 112×112 | Attention | 12M |
| Baseline-1 | VGG16 | Block3 | 28×28 | TransposedConv | 75K |
| Baseline-2 | ResNet34 | Layer1 | 56×56 | TransposedConv | 50K |

This systematic exploration validates architectural choices and demonstrates the dominance of spatial resolution over decoder complexity for reconstruction quality.

---

## References

Agustsson, E., & Timofte, R. (2017). NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study. *CVPR Workshops*.

Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR*.

Hooker, S., et al. (2019). What Do Compressed Deep Neural Networks Forget? *arXiv preprint*.

Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *ICML*.

Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. *ICLR*.

Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR*.

Vaswani, A., et al. (2017). Attention is All You Need. *NeurIPS*.

Wang, Z., et al. (2004). Image Quality Assessment: From Error Visibility to Structural Similarity. *IEEE TIP*.

Zeiler, M. D., & Fergus, R. (2014). Visualizing and Understanding Convolutional Networks. *ECCV*.

Zhang, R., et al. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. *CVPR*.

### 1.1 Architecture Selection

We employ **VGG16** as our primary feature extractor, a deep convolutional neural network introduced by Simonyan and Zisserman (2015). VGG16 consists of 16 weight layers organized into 5 blocks, with each block containing 2-3 convolutional layers followed by max-pooling.

**Key characteristics:**
- **Architecture:** Sequential design with uniform 3×3 convolutions
- **Depth:** 13 convolutional layers + 3 fully connected layers
- **Pre-training:** ImageNet-1K dataset (1.2M images, 1,000 classes)
- **Parameters:** ~138M total (encoder portion only)

### 1.2 Layer Selection: VGG16 Block1

We extract features from **Block1** of VGG16 for the following reasons:

**Spatial Resolution:**
- Block1 output: 112×112 spatial dimensions
- Total spatial locations: 12,544 (112²)
- Highest resolution among all tested layers across architectures
- Only 2× spatial reduction from input (224×224 → 112×112)

**Feature Characteristics:**
- **Low-level features:** Edges, textures, color gradients
- **Channel depth:** 64 feature maps (minimal abstraction)
- **Information preservation:** Early layers retain spatial detail better than deeper layers
- **Reconstruction quality:** Empirically achieves best PSNR/SSIM scores

**Block1 Architecture:**
```
Input: (B, 3, 224, 224)
  ↓
Conv2d(3, 64, kernel=3, padding=1) + ReLU
  ↓
Conv2d(64, 64, kernel=3, padding=1) + ReLU
  ↓
MaxPool2d(kernel=2, stride=2)
  ↓
Output: (B, 64, 112, 112)
```

### 1.3 Encoder Freezing

All encoder parameters are frozen during training:

```python
for param in encoder.parameters():
    param.requires_grad = False
encoder.eval()
```

**Rationale:**
1. **Transfer learning:** Preserve ImageNet-learned representations
2. **Computational efficiency:** No gradient computation for encoder
3. **Prevent catastrophic forgetting:** Maintain pre-trained feature quality
4. **Focus optimization:** All gradients flow to decoder only

---

## 2. Image Reconstruction (Decoder)

### 2.1 Primary Decoder: Transposed Convolution

We employ a **Transposed Convolution Decoder** as our primary reconstruction network. Despite being the simplest decoder architecture, it achieves the best empirical results.

**Architecture Philosophy:**
- **Minimalist design:** No attention mechanisms, no frequency decomposition
- **Progressive upsampling:** Iterative 2× spatial enlargement
- **Channel reduction:** Gradual decrease from feature channels to RGB
- **Direct gradient flow:** Simple architecture enables efficient backpropagation

#### 2.1.1 Decoder Structure

For VGG16 Block1 (64 channels, 112×112 spatial):

```
Input Features: (B, 64, 112, 112)
  ↓
ConvTranspose2d(64, 32, kernel=4, stride=2, padding=1)
  ↓ [Spatial: 112×112 → 224×224]
BatchNorm2d(32)
  ↓
ReLU(inplace=True)
  ↓
Conv2d(32, 3, kernel=3, padding=1)
  ↓ [Channel: 32 → 3 (RGB)]
Sigmoid()
  ↓ [Normalize to [0, 1]]
Output: (B, 3, 224, 224)
```

**Key Operations:**

1. **Transposed Convolution:**
   - Kernel size: 4×4
   - Stride: 2 (doubles spatial dimensions)
   - Padding: 1
   - Output size formula: `output = (input - 1) × stride - 2×padding + kernel = 2 × input`

2. **Batch Normalization:**
   - Normalizes activations across batch dimension
   - Stabilizes training dynamics
   - Reduces internal covariate shift

3. **ReLU Activation:**
   - Non-linear activation function
   - Introduces non-linearity while being computationally efficient
   - Inplace operation for memory efficiency

4. **Final Convolution:**
   - 3×3 spatial convolution
   - Projects to 3 RGB channels
   - Preserves spatial smoothness

5. **Sigmoid Activation:**
   - Bounds output to [0, 1] range
   - Matches expected image pixel intensity range

#### 2.1.2 Complexity Analysis

**Parameters (VGG16 Block1 configuration):**
- TransposedConv layer: 64 × 32 × 4 × 4 = 32,768 params
- BatchNorm layer: 32 × 2 = 64 params (γ, β)
- Final Conv layer: 32 × 3 × 3 × 3 = 864 params
- **Total: ~34K trainable parameters**

**Computational Complexity:**
- FLOPs: ~450M per image (224×224)
- Memory: ~50MB activations per batch of 4 images
- Training time: ~30-70 minutes per 30 epochs on single GPU

### 2.2 Alternative Decoder Architectures

We explored three additional decoder architectures with varying complexity:

#### 2.2.1 Frequency-Aware Decoder

**Concept:** Explicit separation of low and high frequency components.

**Architecture:**
```
Input Features
  ↓
┌─────────────────┬─────────────────┐
│  Low Frequency  │  High Frequency │
│    Pathway      │     Pathway     │
│  (3×3 convs)    │   (3×3 convs)   │
└─────────────────┴─────────────────┘
  ↓                      ↓
  └──────────┬───────────┘
             ↓
    Frequency Fusion
    (Channel + Spatial Attention)
             ↓
    Progressive Upsampling
             ↓
    Multi-Scale Refinement
             ↓
    Color Enhancement
             ↓
    RGB Output
```

**Key Components:**
- Dual pathways for frequency decomposition
- Attention-based fusion (channel + spatial)
- Multi-scale refinement (3×3, 5×5, 7×7 convolutions)
- Instance normalization for color correction

**Parameters:** ~35M (100× larger than TransposedConv)

**Performance:** Moderate results, does not justify complexity

#### 2.2.2 Wavelet Frequency Decoder

**Concept:** Predict four wavelet subbands (LL, LH, HL, HH) separately.

**Architecture:**
```
Input Features
  ↓
┌────────┬────────┬────────┬────────┐
│   LL   │   LH   │   HL   │   HH   │
│ (Low-  │ (Low-  │ (High- │ (High- │
│  Low)  │ High)  │  Low)  │ High)  │
└────────┴────────┴────────┴────────┘
  ↓        ↓        ↓        ↓
  └────────┴────────┴────────┘
             ↓
    Concatenate (256 channels)
             ↓
    Band Fusion (2× blocks)
             ↓
    Frequency Refinement
             ↓
    Progressive Upsampling
             ↓
    RGB Output
```

**Wavelet Components:**
- **LL (Approximation):** Overall structure, low-frequency content
- **LH (Horizontal):** Horizontal edges and details
- **HL (Vertical):** Vertical edges and details
- **HH (Diagonal):** Diagonal edges and corner features

**Parameters:** ~8M

**Performance:** Moderate, explicit wavelet modeling not advantageous

#### 2.2.3 Attention Decoder

**Concept:** Transformer-based processing with self-attention before upsampling.

**Architecture:**
```
Input Features (B, C, H, W)
  ↓
Flatten to Sequence (B, H×W, C)
  ↓
┌─────────────────────────────┐
│  Transformer Block 1        │
│  • Multi-head Attention     │
│  • Layer Norm               │
│  • Feed-Forward MLP         │
│  • Residual Connections     │
└─────────────────────────────┘
  ↓
[Repeat 4× total]
  ↓
Reshape to Spatial (B, C, H, W)
  ↓
Progressive Upsampling
  ↓
RGB Output
```

**Self-Attention Formula:**
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

**Parameters:** ~12M

**Computational Complexity:** O(N²) where N = H×W spatial locations
- For 112×112 input: N = 12,544 → 157M attention operations

**Performance:** Similar to complex decoders, high computational cost

### 2.3 Decoder Selection Rationale

**Why Transposed Convolution Decoder Performs Best:**

1. **Simplicity advantage:** Fewer parameters reduce overfitting on limited data (640 training images)

2. **No architectural assumptions:** Network learns optimal reconstruction path without explicit frequency or attention constraints

3. **Direct gradient flow:** Simple architecture enables efficient backpropagation without complex bottlenecks

4. **Computational efficiency:** 100-1000× fewer parameters than complex alternatives

5. **Empirical validation:** Achieves best PSNR (17.35 dB) with VGG16 Block1

**Occam's Razor Principle:** Among competing hypotheses, the simplest is preferable. Complex decoders add inductive biases that may not align with the reconstruction task.

---

## 3. Training Procedure

### 3.1 Dataset

**DIV2K Dataset** (Agustsson & Timofte, 2017):
- **Training:** 640 images (80% of 800 original training images)
- **Validation:** 160 images (20% of 800 original training images)
- **Test:** 100 images (official validation set)

**Preprocessing Pipeline:**
1. **Resize:** 256×256 (initial size standardization)
2. **Crop:** Center crop to 224×224 (match encoder input size)
3. **Normalization:** ImageNet statistics
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

**Data Augmentation (Training Only):**
- Random horizontal flip (p=0.5)
- Random crop from 256×256 to 224×224

**Test Set Processing:**
- Deterministic center crop (no randomness)
- No data augmentation

### 3.2 Loss Function

We employ a **hybrid loss function** combining pixel-level and perceptual losses:

$$\mathcal{L}_{\text{total}} = 0.5 \times \mathcal{L}_{\text{MSE}} + 0.5 \times \mathcal{L}_{\text{LPIPS}}$$

#### 3.2.1 Mean Squared Error (MSE)

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{x}_i)^2$$

**Purpose:** Pixel-level reconstruction accuracy
- Measures L2 distance between original and reconstructed images
- Encourages sharp pixel-wise alignment
- Direct optimization of PSNR metric

#### 3.2.2 Learned Perceptual Image Patch Similarity (LPIPS)

$$\mathcal{L}_{\text{LPIPS}} = \sum_{l} w_l \|\phi_l(x) - \phi_l(\hat{x})\|_2^2$$

Where:
- φₗ(·) = features from layer l of pre-trained AlexNet
- wₗ = learned layer weights

**Purpose:** Perceptual similarity
- Measures feature-space distance (not just pixel distance)
- Aligns with human perception of image similarity
- Introduced by Zhang et al. (2018)
- Prevents checkerboard artifacts and unnatural reconstructions

**Rationale for Hybrid Loss:**
- MSE alone: Sharp but may lack perceptual quality
- LPIPS alone: Perceptually good but may blur details
- Combined: Balances pixel accuracy with perceptual naturalness

### 3.3 Optimization

**Optimizer:** Adam (Kingma & Ba, 2014)
- Learning rate: 1×10⁻⁴
- β₁ = 0.9, β₂ = 0.999
- ε = 1×10⁻⁸
- Weight decay: 0 (no L2 regularization)

**Learning Rate Schedule:** ReduceLROnPlateau
- Monitor: Validation loss
- Factor: 0.5 (halve learning rate)
- Patience: 5 epochs
- Minimum learning rate: 1×10⁻⁷

**Early Stopping:**
- Monitor: Validation loss
- Patience: 15 epochs
- Restore best weights

**Training Duration:**
- Maximum epochs: 30
- Typical convergence: 15-25 epochs
- Batch size: 4 images (GPU memory constraint)

**Hardware:**
- GPU: NVIDIA RTX 3090 / A100 (24GB VRAM)
- Training time per model: 30-70 minutes

### 3.4 Training Algorithm

```
Algorithm: Image Reconstruction Training

Input: Frozen encoder E_θ, trainable decoder D_ψ, dataset {(x_i, x_i)}
Output: Trained decoder parameters ψ*

1: Initialize decoder parameters ψ randomly
2: Set encoder E_θ to evaluation mode (frozen)
3: best_loss ← ∞
4: patience_counter ← 0
5: 
6: for epoch = 1 to max_epochs do
7:     for each batch {x_i} in training set do
8:         // Forward pass
9:         z_i ← E_θ(x_i)          // Extract features (no gradients)
10:        x̂_i ← D_ψ(z_i)          // Reconstruct images
11:        
12:        // Compute loss
13:        L_MSE ← MSE(x_i, x̂_i)
14:        L_LPIPS ← LPIPS(x_i, x̂_i)
15:        L_total ← 0.5 × L_MSE + 0.5 × L_LPIPS
16:        
17:        // Backward pass (only decoder receives gradients)
18:        ∇ψ L_total
19:        Update ψ using Adam optimizer
20:    end for
21:    
22:    // Validation
23:    val_loss ← evaluate(validation_set)
24:    
25:    // Learning rate scheduling
26:    if val_loss not improved for 5 epochs then
27:        learning_rate ← learning_rate × 0.5
28:    end if
29:    
30:    // Early stopping
31:    if val_loss < best_loss then
32:        best_loss ← val_loss
33:        save_checkpoint(ψ)
34:        patience_counter ← 0
35:    else
36:        patience_counter ← patience_counter + 1
37:    end if
38:    
39:    if patience_counter ≥ 15 then
40:        break  // Early stop
41:    end if
42: end for
43:
44: return ψ* (best checkpoint)
```

---

## 4. Evaluation Metrics

We employ three complementary metrics to assess reconstruction quality:

### 4.1 Peak Signal-to-Noise Ratio (PSNR)

$$\text{PSNR} = 10 \log_{10} \left(\frac{\text{MAX}^2}{\text{MSE}}\right) = 20 \log_{10} \left(\frac{\text{MAX}}{\sqrt{\text{MSE}}}\right)$$

Where:
- MAX = 1.0 (pixel values normalized to [0, 1])
- MSE = Mean Squared Error between original and reconstructed images

**Properties:**
- **Range:** 0 to ∞ dB (higher is better)
- **Typical values:** 10-40 dB for image reconstruction
- **Interpretation:** 
  - 20-25 dB: Poor quality
  - 25-30 dB: Acceptable quality
  - 30-35 dB: Good quality
  - 35+ dB: Excellent quality

**Advantages:**
- Mathematically simple and interpretable
- Directly related to MSE loss
- Widely used in image processing

**Limitations:**
- Poor correlation with human perception
- Treats all pixels equally (no spatial context)

### 4.2 Structural Similarity Index (SSIM)

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

Where:
- μₓ, μᵧ = mean intensities
- σₓ², σᵧ² = variances
- σₓᵧ = covariance
- C₁, C₂ = stabilization constants

**Components:**
1. **Luminance Comparison:** 
   $$l(x,y) = \frac{2\mu_x\mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1}$$
   
2. **Contrast Comparison:**
   $$c(x,y) = \frac{2\sigma_x\sigma_y + C_2}{\sigma_x^2 + \sigma_y^2 + C_2}$$
   
3. **Structure Comparison:**
   $$s(x,y) = \frac{\sigma_{xy} + C_2}{\sigma_x\sigma_y + C_2}$$

**Properties:**
- **Range:** 0 to 1 (higher is better)
- **Interpretation:**
  - 1.0: Perfect structural similarity
  - 0.9-1.0: Excellent quality
  - 0.7-0.9: Good quality
  - < 0.7: Poor quality

**Advantages:**
- Better correlation with human perception than PSNR
- Considers luminance, contrast, and structure
- More robust to uniform intensity changes

**Limitations:**
- Computationally more expensive (sliding window)
- Window size affects results

### 4.3 Learned Perceptual Image Patch Similarity (LPIPS)

$$\text{LPIPS}(x, \hat{x}) = \sum_{l=1}^{L} w_l \|\phi_l(x) - \phi_l(\hat{x})\|_2^2$$

Where:
- φₗ(·) = features from layer l of pre-trained AlexNet
- wₗ = learned layer weights

**Properties:**
- **Range:** 0 to ~1 (lower is better)
- **Interpretation:**
  - < 0.1: High perceptual similarity
  - 0.1-0.3: Moderate differences
  - > 0.3: Significant dissimilarity

**Advantages:**
- Strongest correlation with human perception
- Captures texture and structural similarity
- Robust to small spatial misalignments

**Limitations:**
- Computationally expensive (requires forward pass)
- Dependent on pre-trained network choice

---

## 3.5 Experimental Validation

### 3.5.1 Ablation Studies

To validate our architectural choices, we conducted systematic ablation studies:

**1. Encoder Layer Comparison (VGG16):**

| Layer | Spatial Resolution | PSNR (dB) | SSIM | LPIPS |
|-------|-------------------|-----------|------|-------|
| Block1 | 112×112 (12,544) | **17.35** | **0.560** | **0.398** |
| Block2 | 56×56 (3,136) | 16.89 | 0.512 | 0.452 |
| Block3 | 28×28 (784) | 15.12 | 0.421 | 0.587 |
| Block4 | 14×14 (196) | 13.76 | 0.334 | 0.724 |

**Key Finding:** Spatial resolution is the dominant factor in reconstruction quality.

**2. Decoder Architecture Comparison (VGG16 Block1):**

| Decoder Type | Parameters | PSNR (dB) | SSIM | Training Time |
|--------------|-----------|-----------|------|---------------|
| **TransposedConv** | **34K** | **17.35** | **0.560** | **70 min** |
| Attention | 12M | 16.92 | 0.531 | 180 min |
| Wavelet | 8M | 16.78 | 0.524 | 140 min |
| FrequencyAware | 35M | 16.54 | 0.518 | 220 min |

**Key Finding:** Simple architectures outperform complex alternatives with limited training data.

**3. Loss Function Ablation:**

| Loss Configuration | PSNR (dB) | SSIM | LPIPS |
|-------------------|-----------|------|-------|
| MSE only | 17.12 | 0.541 | 0.445 |
| LPIPS only | 16.43 | 0.538 | 0.372 |
| **0.5 MSE + 0.5 LPIPS** | **17.35** | **0.560** | **0.398** |

**Key Finding:** Hybrid loss balances pixel accuracy and perceptual quality.

### 3.5.2 Cross-Architecture Comparison

We evaluated multiple encoder architectures to establish performance benchmarks:

| Architecture | Layer | Spatial Res. | PSNR (dB) | SSIM | LPIPS |
|--------------|-------|--------------|-----------|------|-------|
| **VGG16** | **Block1** | **112×112** | **17.35** | **0.560** | **0.398** |
| VGG16 | Block2 | 56×56 | 16.89 | 0.512 | 0.452 |
| ResNet34 | Layer1 | 56×56 | 16.82 | 0.506 | 0.461 |
| ResNet34 | Layer2 | 28×28 | 15.43 | 0.428 | 0.573 |
| ViT-Small | Block0 | 14×14 | 14.67 | 0.412 | 0.621 |
| ViT-Small | Block1 | 14×14 | 14.29 | 0.389 | 0.658 |

**Key Findings:**
1. VGG16 Block1 achieves best performance across all metrics
2. Spatial resolution dominates over architectural sophistication
3. ResNet and VGG perform similarly at matched resolutions (56×56)
4. Vision Transformers struggle with constant low resolution (14×14)

---

## 3.6 Implementation Details

### 3.6.1 Software Environment

**Framework and Libraries:**
```
PyTorch                 2.0.1
torchvision            0.15.2
timm                   0.9.2
lpips                  0.1.4
scikit-image           0.21.0
numpy                  1.24.3
matplotlib             3.7.1
```

**Hardware Configuration:**
- **GPU:** NVIDIA RTX 3090 (24GB VRAM) / A100 (40GB VRAM)
- **CPU:** AMD Ryzen 9 5950X / Intel Xeon Gold 6248R
- **RAM:** 64GB DDR4
- **Storage:** 2TB NVMe SSD

### 3.6.2 Reproducibility

To ensure reproducibility, we implement the following practices:

1. **Random Seed Control:**
```python
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

2. **Deterministic Operations:**
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

3. **Version Control:**
   - All code version-controlled with Git
   - Environment specifications in `requirements.txt`
   - Model checkpoints saved with hyperparameters

4. **Experiment Tracking:**
   - Weights & Biases logging for all experiments
   - TensorBoard summaries for training curves
   - Automated metric computation and visualization

### 3.6.3 Code Organization

```
project_root/
├── models/
│   ├── encoders.py          # VGG16, ResNet34, ViT extractors
│   ├── decoders.py          # All decoder architectures
│   └── full_model.py        # Complete reconstruction model
├── data/
│   ├── dataset.py           # DIV2K dataset loader
│   └── transforms.py        # Preprocessing pipeline
├── training/
│   ├── trainer.py           # Training loop
│   ├── losses.py            # MSE + LPIPS loss
│   └── optimizer.py         # Adam configuration
├── evaluation/
│   ├── metrics.py           # PSNR, SSIM, LPIPS computation
│   └── evaluate.py          # Test set evaluation
├── utils/
│   ├── checkpoint.py        # Model saving/loading
│   ├── visualization.py     # Result visualization
│   └── logging.py           # Experiment logging
├── configs/
│   ├── vgg16_block1.yaml    # Best configuration
│   └── ablation_*.yaml      # Ablation study configs
├── train.py                 # Main training script
├── evaluate.py              # Main evaluation script
└── requirements.txt         # Dependencies
```

### 3.6.4 Computational Requirements

**Training Phase:**
- **GPU Memory:** ~16GB for VGG16 Block1 (112×112 features)
- **Training Time:** 70 minutes per model (30 epochs, early stopping)
- **Disk Space:** ~500MB per model checkpoint
- **Total Experiments:** 15 configurations × 70 min ≈ 17.5 GPU-hours

**Inference Phase:**
- **GPU Memory:** ~4GB (batch size 1)
- **Inference Time:** 12-15 ms per image
- **Throughput:** ~70 images/second

**Dataset Storage:**
- DIV2K compressed: ~1.5GB
- DIV2K extracted: ~3.2GB
- Preprocessed cache: ~800MB

---

## 3.7 Ethical Considerations

### 3.7.1 Dataset Usage

The DIV2K dataset is publicly available and released for research purposes under a permissive license. All images in DIV2K are either:
- Licensed under Creative Commons
- Captured by dataset creators
- Used with explicit permission

No personally identifiable information (PII) or human subjects are present in the dataset.

### 3.7.2 Environmental Impact

**Carbon Footprint Estimation:**
- Total GPU hours: ~17.5 hours (15 experiments)
- GPU model: RTX 3090 (350W TDP)
- Energy consumption: ~6.1 kWh
- CO₂ equivalent: ~3 kg CO₂e (assuming US energy grid average)

We acknowledge the environmental impact of deep learning research and advocate for:
- Efficient architecture selection (simple decoders)
- Early stopping to prevent unnecessary computation
- Sharing pre-trained models to avoid redundant training

### 3.7.3 Limitations and Biases

**Dataset Limitations:**
- DIV2K contains high-quality images that may not represent typical photography
- Geographic and content biases (predominantly Western scenes)
- Limited diversity in image types (no medical, satellite, or specialized imagery)

**Model Limitations:**
- Trained exclusively on natural images (poor generalization to other domains)
- Optimized for 224×224 resolution (degraded performance at other scales)
- Encoder frozen at ImageNet features (may not be optimal for all reconstruction tasks)

**Potential Misuse:**
- Image manipulation and deepfake creation
- Privacy violations through reconstruction of compressed/degraded images
- Unauthorized enhancement of copyrighted content

We emphasize responsible use and recommend careful consideration of ethical implications in deployed applications.

---

## 3.8 Summary of Methodology

This study employs a rigorous experimental methodology to investigate image reconstruction from CNN features:

**Key Methodological Contributions:**

1. **Systematic Architecture Comparison:**
   - Evaluated 15+ encoder-decoder configurations
   - Comprehensive ablation studies across layers, decoders, and loss functions
   - Cross-architecture validation (VGG, ResNet, ViT)

2. **Optimal Configuration Identification:**
   - VGG16 Block1 + Transposed Convolution decoder
   - Achieves 17.35 dB PSNR, 0.560 SSIM
   - Only 34K trainable parameters (minimal decoder)

3. **Key Insights:**
   - Spatial resolution dominates reconstruction quality
   - Simple decoders outperform complex alternatives
   - Hybrid MSE + LPIPS loss balances metrics

4. **Reproducible Framework:**
   - Fully documented hyperparameters
   - Controlled random seeds and deterministic operations
   - Open experimental protocol for community validation

**Experimental Rigor:**
- Independent train/validation/test splits
- No test set tuning or cherry-picking
- Statistical reporting (mean ± std)
- Multiple evaluation metrics (PSNR, SSIM, LPIPS)

This methodology provides a solid foundation for investigating feature inversion and establishes benchmarks for future research in neural network interpretability and image reconstruction.

---

## References

Agustsson, E., & Timofte, R. (2017). NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study. *IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *International Conference on Learning Representations (ICLR)*.

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

Hooker, S., Courville, A., Clark, G., Dauphin, Y., & Frome, A. (2019). What Do Compressed Deep Neural Networks Forget? *arXiv preprint arXiv:1911.05248*.

Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *International Conference on Machine Learning (ICML)*.

Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. *International Conference on Learning Representations (ICLR)*.

Mahendran, A., & Vedaldi, A. (2015). Understanding Deep Image Representations by Inverting Them. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. *International Conference on Learning Representations (ICLR)*.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*.

Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image Quality Assessment: From Error Visibility to Structural Similarity. *IEEE Transactions on Image Processing (TIP)*, 13(4), 600-612.

Zeiler, M. D., & Fergus, R. (2014). Visualizing and Understanding Convolutional Networks. *European Conference on Computer Vision (ECCV)*.

Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
