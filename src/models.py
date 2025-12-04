"""
Neural Network Architectures for Image Reconstruction
=====================================================

This module defines the model architectures for reconstructing images from
intermediate neural network features. The architecture consists of three main
components:

1. Feature Extractors (Frozen Encoders):
   - Extract intermediate features from pre-trained networks
   - ResNet34, VGG16, ViT, PVT architectures supported
   - Frozen weights (no gradient updates during training)

2. Decoders (Trainable):
   - Reconstruct RGB images from extracted features
   - Four variants: FrequencyAware, Wavelet, TransposedConv, Attention
   - Only component trained during experiments

3. Fusion Modules (For Ensembles):
   - Combine features from multiple architectures
   - Three strategies: Attention, Concat, Weighted

Model Pipeline:
    Input Image (224×224×3)
         ↓
    Frozen Encoder (ResNet/VGG/ViT/PVT)
         ↓
    Intermediate Features (C×H×W)
         ↓
    [Optional] Feature Fusion (for ensembles)
         ↓
    Trainable Decoder
         ↓
    Reconstructed Image (224×224×3)

Key Design Principles:
- Transfer learning: Use ImageNet pre-trained encoders (frozen)
- Spatial resolution: Shallow layers preserve more detail
- Decoder simplicity: Simple decoders often perform best
- Feature fusion: Combine complementary architectures

Usage Example:
    >>> # Single architecture model
    >>> config = {
    ...     'architecture': 'vgg16',
    ...     'vgg_block': 'block1',
    ...     'decoder_type': 'transposed_conv',
    ...     'output_size': 224
    ... }
    >>> model = SingleArchModel(config)
    >>> reconstructed = model(images)  # (B, 3, 224, 224)
    
    >>> # Ensemble model
    >>> config = {
    ...     'architectures': ['resnet34', 'vgg16', 'vit_small_patch16_224'],
    ...     'resnet_layer': 'layer1',
    ...     'vgg_block': 'block1',
    ...     'vit_block': 'block1',
    ...     'fusion_strategy': 'attention',
    ...     'fusion_channels': 256,
    ...     'decoder_type': 'transposed_conv',
    ...     'output_size': 224
    ... }
    >>> model = EnsembleModel(config)
    >>> reconstructed = model(images)  # (B, 3, 224, 224)

Experimental Findings:
- VGG16 block1 (112×112) produces best single-model results (17.35 dB PSNR)
- TransposedConv decoder outperforms complex decoders despite simplicity
- Ensemble models provide marginal gains (+0.29 dB) with 4× compute cost
- Spatial resolution more important than semantic depth for reconstruction

Author: Danica Blazanovic, Abbas Khan
Course: CAP6415 - Computer Vision, Fall 2025
Institution: Florida Atlantic University
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm


# ============================================================================
# FEATURE EXTRACTORS (FROZEN ENCODERS)
# ============================================================================
# These classes extract intermediate features from pre-trained networks.
# All extractors are frozen (no gradient updates) to preserve ImageNet features.
# The choice of layer/block significantly impacts reconstruction quality.

class ResNetExtractor(nn.Module):
    """
    Extract intermediate features from ResNet34 architecture.
    
    ResNet34 is a residual network with skip connections that enable training
    deeper networks. It processes images through 4 progressive layers, each
    with different spatial resolutions and channel counts.
    
    Architecture Overview:
        Input (224×224×3)
        ↓ conv1, bn1, relu, maxpool
        ↓ layer1: 64 channels, 56×56 spatial (64 × 56 × 56 = 200,704 values)
        ↓ layer2: 128 channels, 28×28 spatial (128 × 28 × 28 = 100,352 values)
        ↓ layer3: 256 channels, 14×14 spatial (256 × 14 × 14 = 50,176 values)
        ↓ layer4: 512 channels, 7×7 spatial (512 × 7 × 7 = 25,088 values)
    
    Key Characteristics:
    - Residual connections: Enable gradient flow in deep networks
    - Progressive downsampling: 2× spatial reduction per layer
    - Channel expansion: Channels double each layer (64→128→256→512)
    - Batch normalization: Stabilizes training
    
    Reconstruction Quality by Layer (from baseline experiments):
    - layer1 (56×56): BEST for reconstruction due to high resolution
    - layer2 (28×28): Moderate quality, balanced resolution/semantics
    - layer3 (14×14): Poor reconstruction, too much spatial loss
    - layer4 (7×7): Worst reconstruction, minimal spatial information
    
    Args:
        layer_name (str): Which layer to extract features from.
                         Options: 'layer1', 'layer2', 'layer3', 'layer4'
                         Default: 'layer1' (recommended for reconstruction)
    
    Input:
        x (torch.Tensor): RGB images, shape (B, 3, 224, 224)
    
    Output:
        features (torch.Tensor): Extracted features
                                - layer1: (B, 64, 56, 56)
                                - layer2: (B, 128, 28, 28)
                                - layer3: (B, 256, 14, 14)
                                - layer4: (B, 512, 7, 7)
    
    Example:
        >>> extractor = ResNetExtractor('layer1')
        >>> images = torch.randn(4, 3, 224, 224)
        >>> features = extractor(images)  # (4, 64, 56, 56)
    
    Note:
        All parameters are frozen (requires_grad=False) and model is in eval()
        mode to ensure consistent feature extraction during training.
    """
    
    def __init__(self, layer_name='layer1'):
        super().__init__()
        
        # Load pre-trained ResNet34 with ImageNet weights
        # IMAGENET1K_V1 = torchvision's ImageNet-1K weights (top-1: 73.3%)
        resnet = models.resnet34(weights='IMAGENET1K_V1')
        
        # Build sequential feature extractor up to specified layer
        # Start with stem: conv1 (7×7, stride 2) → bn1 → relu → maxpool (3×3, stride 2)
        # This reduces 224×224 → 56×56 before first residual layer
        layers = [
            resnet.conv1,      # 3 → 64 channels, 224×224 → 112×112
            resnet.bn1,        # Batch normalization
            resnet.relu,       # ReLU activation
            resnet.maxpool     # 112×112 → 56×56
        ]
        
        # Progressively add residual layers based on desired extraction point
        # Each layer contains multiple residual blocks
        if layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layers.append(resnet.layer1)  # 56×56, 64 channels
        if layer_name in ['layer2', 'layer3', 'layer4']:
            layers.append(resnet.layer2)  # 28×28, 128 channels (stride 2 downsample)
        if layer_name in ['layer3', 'layer4']:
            layers.append(resnet.layer3)  # 14×14, 256 channels (stride 2 downsample)
        if layer_name == 'layer4':
            layers.append(resnet.layer4)  # 7×7, 512 channels (stride 2 downsample)
        
        # Combine into sequential module for efficient forward pass
        self.extractor = nn.Sequential(*layers)
        
        # Freeze all parameters - no gradient updates during training
        self._freeze()
    
    def _freeze(self):
        """
        Freeze all extractor parameters and set to evaluation mode.
        
        This is critical for feature extraction:
        - Prevents weight updates during training (transfer learning)
        - Disables dropout and uses population stats for batchnorm
        - Ensures consistent features across training/evaluation
        """
        for param in self.parameters():
            param.requires_grad = False  # No gradient computation
        self.eval()  # Evaluation mode (affects batchnorm, dropout)
    
    def forward(self, x):
        """
        Extract features from input images.
        
        Args:
            x (torch.Tensor): Input images, shape (B, 3, 224, 224)
        
        Returns:
            torch.Tensor: Extracted features, shape depends on layer_name:
                         - layer1: (B, 64, 56, 56)
                         - layer2: (B, 128, 28, 28)
                         - layer3: (B, 256, 14, 14)
                         - layer4: (B, 512, 7, 7)
        """
        return self.extractor(x)


class VGGExtractor(nn.Module):
    """
    Extract intermediate features from VGG16 architecture.
    
    VGG16 is a simple sequential CNN with aggressive max-pooling. It consists
    of 5 blocks of convolutional layers followed by max-pooling. Unlike ResNet,
    VGG has no skip connections and uses small 3×3 convolutions throughout.
    
    Architecture Overview:
        Input (224×224×3)
        ↓ block1: 2× conv3×3(64), maxpool → 112×112×64 (803,968 values) **BEST**
        ↓ block2: 2× conv3×3(128), maxpool → 56×56×128 (401,408 values)
        ↓ block3: 3× conv3×3(256), maxpool → 28×28×256 (200,704 values)
        ↓ block4: 3× conv3×3(512), maxpool → 14×14×512 (100,352 values)
        ↓ block5: 3× conv3×3(512), maxpool → 7×7×512 (25,088 values)
    
    Key Characteristics:
    - Simplicity: Only 3×3 convs and 2×2 max pooling
    - Deep stack: Multiple conv layers before pooling
    - Aggressive pooling: 2× reduction after each block
    - High resolution start: block1 maintains 112×112 (largest among all architectures)
    
    Reconstruction Quality by Block (from baseline experiments):
    - block1 (112×112): **BEST OVERALL** - highest spatial resolution
                       Achieves 17.35 dB PSNR with transposed_conv decoder
                       12,544 spatial locations preserve fine details
    - block2 (56×56): Good quality, reasonable resolution
    - block3 (28×28): Moderate quality, balanced resolution/semantics  
    - block4+ (≤14×14): Poor reconstruction, insufficient spatial detail
    
    Why VGG16 block1 is Best:
    1. Highest spatial resolution (112×112) of any architecture/layer tested
    2. Early features preserve texture and color information
    3. Simple architecture → easy for decoder to invert
    4. Only 64 channels → less complex feature space
    
    Args:
        block_name (str): Which block to extract features from.
                         Options: 'block1', 'block2', 'block3', 'block4', 'block5'
                         Default: 'block3' (but 'block1' recommended for reconstruction)
    
    Input:
        x (torch.Tensor): RGB images, shape (B, 3, 224, 224)
    
    Output:
        features (torch.Tensor): Extracted features
                                - block1: (B, 64, 112, 112)  **BEST**
                                - block2: (B, 128, 56, 56)
                                - block3: (B, 256, 28, 28)
                                - block4: (B, 512, 14, 14)
                                - block5: (B, 512, 7, 7)
    
    Example:
        >>> extractor = VGGExtractor('block1')
        >>> images = torch.randn(4, 3, 224, 224)
        >>> features = extractor(images)  # (4, 64, 112, 112) - largest feature map!
    
    Note:
        VGG's features[:n] notation: 
        - features[0-4]: block1 (2 convs + relu + maxpool)
        - features[5-9]: block2
        - features[10-16]: block3
        - features[17-23]: block4
        - features[24-30]: block5
    """
    
    def __init__(self, block_name='block3'):
        super().__init__()
        
        # Load pre-trained VGG16 with ImageNet weights
        # IMAGENET1K_V1 = torchvision's ImageNet-1K weights (top-1: 71.6%)
        vgg = models.vgg16(weights='IMAGENET1K_V1')
        
        # Map block names to feature layer indices
        # VGG features are in sequential container, slice up to desired block
        # Index points to the last layer (maxpool) of each block
        layer_map = {
            'block1': 4,   # 2 convs + 1 maxpool → 112×112×64
            'block2': 9,   # 2 convs + 1 maxpool → 56×56×128
            'block3': 16,  # 3 convs + 1 maxpool → 28×28×256
            'block4': 23,  # 3 convs + 1 maxpool → 14×14×512
            'block5': 30   # 3 convs + 1 maxpool → 7×7×512
        }
        
        # Get index for specified block (default to block3 if invalid)
        idx = layer_map.get(block_name, 16)
        
        # Extract feature layers up to and including specified block
        # [:idx + 1] includes the maxpool layer at index idx
        self.extractor = vgg.features[:idx + 1]
        
        # Freeze all parameters - no training of encoder
        self._freeze()
    
    def _freeze(self):
        """
        Freeze all extractor parameters and set to evaluation mode.
        
        Critical for transfer learning:
        - Preserves ImageNet features learned on 1.2M images
        - Prevents catastrophic forgetting during decoder training
        - Ensures consistent feature extraction
        """
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def forward(self, x):
        """
        Extract features from input images.
        
        Args:
            x (torch.Tensor): Input images, shape (B, 3, 224, 224)
        
        Returns:
            torch.Tensor: Extracted features, shape depends on block_name:
                         - block1: (B, 64, 112, 112)  **Recommended**
                         - block2: (B, 128, 56, 56)
                         - block3: (B, 256, 28, 28)
                         - block4: (B, 512, 14, 14)
                         - block5: (B, 512, 7, 7)
        """
        return self.extractor(x)


class ViTExtractor(nn.Module):
    """
    Extract intermediate features from Vision Transformer (ViT) architecture.
    
    Vision Transformer applies pure self-attention to sequences of image patches.
    Unlike CNNs, ViT maintains constant spatial resolution throughout all blocks,
    operating on a fixed 14×14 grid of 16×16 patches.
    
    Architecture Overview:
        Input (224×224×3)
        ↓ Patch embedding: Split into 14×14 = 196 patches (16×16 each)
        ↓ Position embedding: Add learned positional encodings
        ↓ [CLS] token prepended (197 tokens total)
        ↓ Block 1-12: Transformer blocks (Multi-head self-attention + MLP)
        ↓ Each block: (197 tokens, 384 dims) for ViT-Small
        ↓ Output: Reshaped to (B, 384, 14, 14) for consistency with CNNs
    
    Key Characteristics:
    - Global attention: Each token attends to all others (unlike CNN receptive fields)
    - Constant resolution: All blocks process 14×14 patches
    - Position encoding: Patches have learned positional information
    - Semantic depth: Later blocks capture more abstract concepts
    - Patch-based: 16×16 patches limit fine detail capture
    
    Reconstruction Quality by Block (from baseline experiments):
    - All blocks: 14×14 spatial → limited reconstruction quality
    - block1: Early features, less semantic, better texture
    - block3-6: Mid-level features, balanced abstraction
    - block9-12: Deep semantic features, poor reconstruction
    
    Why ViT struggles for reconstruction:
    1. Low spatial resolution (14×14) compared to CNN shallow layers
    2. Patch granularity (16×16) loses fine details
    3. Global attention dilutes local spatial information
    4. Designed for classification, not spatial reconstruction
    
    Comparison with CNNs:
    - VGG block1: 112×112 = 12,544 locations
    - ResNet layer1: 56×56 = 3,136 locations  
    - ViT any block: 14×14 = 196 locations (65× fewer than VGG!)
    
    Args:
        architecture (str): ViT model variant from timm library
                           Default: 'vit_small_patch16_224' (384 dims, 12 blocks)
                           Other options: 'vit_base_patch16_224' (768 dims, 12 blocks)
                                         'vit_large_patch16_224' (1024 dims, 24 blocks)
        
        block_name (str): Which transformer block to extract from.
                         Format: 'block{N}' where N ∈ [1, 12] for ViT-Small
                         Default: 'block3' (early-mid level features)
    
    Input:
        x (torch.Tensor): RGB images, shape (B, 3, 224, 224)
    
    Output:
        features (torch.Tensor): Extracted features, shape (B, 384, 14, 14)
                                Note: Constant 14×14 spatial for all blocks
                                      Channel dim depends on model variant
    
    Example:
        >>> extractor = ViTExtractor('vit_small_patch16_224', 'block1')
        >>> images = torch.randn(4, 3, 224, 224)
        >>> features = extractor(images)  # (4, 384, 14, 14)
    
    Technical Details:
        The forward pass:
        1. Patch embedding: (B, 3, 224, 224) → (B, 197, 384)
           - 196 image patches + 1 CLS token
        2. Positional encoding: Add learned positions
        3. Transformer blocks: Process with self-attention
        4. Extract at block_idx
        5. Remove CLS token: (B, 197, 384) → (B, 196, 384)
        6. Reshape: (B, 196, 384) → (B, 384, 14, 14)
    """
    
    def __init__(self, architecture='vit_small_patch16_224', block_name='block3'):
        super().__init__()
        
        # Load pre-trained ViT from timm (PyTorch Image Models) library
        # timm provides many ViT variants with different sizes and patch configs
        # pretrained=True loads weights trained on ImageNet-21K then fine-tuned on ImageNet-1K
        self.vit = timm.create_model(architecture, pretrained=True)
        
        # Parse block index from block name (e.g., 'block3' → 3)
        # Subtract 1 because indexing is 0-based but user inputs are 1-based
        # ViT-Small has 12 blocks (indices 0-11), so 'block1' extracts from blocks[0]
        if block_name.startswith('block'):
            self.block_idx = int(block_name.replace('block', '')) - 1
        else:
            # Fallback: default to block 6 (middle of 12 blocks)
            self.block_idx = 5  # blocks[5] = 6th block
        
        # Freeze all parameters
        self._freeze()
    
    def _freeze(self):
        """
        Freeze all ViT parameters and set to evaluation mode.
        
        ViT models have many parameters (22M for ViT-Small):
        - Patch embedding projection
        - Positional embeddings
        - 12 transformer blocks (attention + MLP each)
        - Layer norms
        
        All frozen to preserve ImageNet features.
        """
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def forward(self, x):
        """
        Extract features from input images.
        
        Process:
        1. Convert image to patch embeddings (B, 3, 224, 224) → (B, 197, D)
        2. Add positional encodings
        3. Pass through transformer blocks until block_idx
        4. Remove CLS token and reshape to spatial format
        
        Args:
            x (torch.Tensor): Input images, shape (B, 3, 224, 224)
        
        Returns:
            torch.Tensor: Extracted features, shape (B, D, 14, 14)
                         where D = 384 for ViT-Small
                               D = 768 for ViT-Base
                               D = 1024 for ViT-Large
        
        Note:
            Spatial resolution is always 14×14 regardless of block depth.
            This is a key limitation for reconstruction tasks compared to CNNs.
        """
        # Step 1: Patch embedding - split image into 14×14 grid of 16×16 patches
        # Input: (B, 3, 224, 224)
        # Output: (B, 196, D) where 196 = 14×14 patches
        x = self.vit.patch_embed(x)
        
        # Step 2: Add positional embeddings (includes CLS token)
        # Output: (B, 197, D) where 197 = 196 patches + 1 CLS token
        x = self.vit._pos_embed(x)
        
        # Step 3: Process through transformer blocks up to desired depth
        # Each block: Multi-head Self-Attention → LayerNorm → MLP → LayerNorm
        for i, block in enumerate(self.vit.blocks):
            x = block(x)
            if i == self.block_idx:
                break  # Extract features at this depth
        
        # Step 4: Reshape from sequence to spatial format
        B, N, D = x.shape  # N = 197 (196 patches + 1 CLS)
        
        # Calculate spatial dimensions (sqrt of number of patches)
        # N - 1 excludes CLS token, so 197 - 1 = 196 patches
        # √196 = 14, so H = W = 14
        H = W = int((N - 1) ** 0.5)
        
        # Remove CLS token and reshape to spatial grid
        # x[:, 1:, :] removes CLS at position 0
        # transpose(1, 2) swaps sequence and channel dims: (B, N-1, D) → (B, D, N-1)
        # reshape creates spatial grid: (B, D, N-1) → (B, D, H, W)
        x = x[:, 1:, :].transpose(1, 2).reshape(B, D, H, W)
        
        return x  # (B, D, 14, 14) - constant resolution for all blocks


class PVTExtractor(nn.Module):
    """
    Extract intermediate features from Pyramid Vision Transformer (PVT) architecture.
    
    PVT is a hierarchical vision transformer that combines the best of both worlds:
    - CNN-style spatial pyramid (progressive downsampling)
    - Transformer-style self-attention mechanisms
    
    Unlike standard ViT which maintains constant 14×14 resolution, PVT uses
    4 progressive stages with decreasing spatial resolution, similar to CNNs.
    
    Architecture Overview:
        Input (224×224×3)
        ↓ Patch embedding (7×7, stride 4) → 56×56 initial patches
        ↓ Stage 1: 56×56, 64 channels, spatial reduction ratio R=1
        ↓ Stage 2: 28×28, 128 channels, spatial reduction ratio R=2  
        ↓ Stage 3: 14×14, 320 channels, spatial reduction ratio R=4
        ↓ Stage 4: 7×7, 512 channels, spatial reduction ratio R=8
    
    Key Characteristics:
    - Hierarchical: Progressive spatial downsampling (like ResNet)
    - Efficient attention: Spatial-reduction attention reduces computation
    - Pyramid features: Multi-scale feature hierarchy
    - Channel expansion: Channels increase as spatial dims decrease
    
    PVT-v2-B2 Configuration (used in this study):
    - Parameters: 25M (between ViT-Small and ViT-Base)
    - Depths: [3, 4, 6, 3] transformer blocks per stage
    - Channels: [64, 128, 320, 512]
    - Spatial: [56×56, 28×28, 14×14, 7×7]
    
    Reconstruction Quality by Stage (from baseline experiments):
    - stage1 (56×56): Best PVT layer, comparable to ResNet layer1
    - stage2 (28×28): Moderate quality, similar to ResNet layer2
    - stage3+ (≤14×14): Poor reconstruction, too much downsampling
    
    Comparison with other architectures:
    - Similar spatial hierarchy to ResNet
    - Global attention like ViT but with reduced computational cost
    - Combines CNN inductive biases with transformer flexibility
    
    Why PVT for reconstruction:
    1. Higher resolution than standard ViT (56×56 vs 14×14 at stage1)
    2. Hierarchical features more suitable than single-scale ViT
    3. Spatial reduction attention provides efficiency
    4. Still lower quality than VGG16 block1 (112×112) for reconstruction
    
    Args:
        architecture (str): PVT model variant from timm library
                           Default: 'pvt_v2_b2' (25M params, balanced)
                           Other options: 'pvt_v2_b0' (3.7M params, lightweight)
                                         'pvt_v2_b3' (45M params, large)
                                         'pvt_v2_b4' (62M params, very large)
        
        stage_name (str): Which stage to extract features from.
                         Format: 'stage{N}' where N ∈ [1, 2, 3, 4]
                         Default: 'stage2' (balanced resolution/semantics)
    
    Input:
        x (torch.Tensor): RGB images, shape (B, 3, 224, 224)
    
    Output:
        features (torch.Tensor): Extracted features
                                - stage1: (B, 64, 56, 56)
                                - stage2: (B, 128, 28, 28)
                                - stage3: (B, 320, 14, 14)
                                - stage4: (B, 512, 7, 7)
    
    Example:
        >>> extractor = PVTExtractor('pvt_v2_b2', 'stage1')
        >>> images = torch.randn(4, 3, 224, 224)
        >>> features = extractor(images)  # (4, 64, 56, 56)
    
    Technical Details:
        Uses forward hooks to extract intermediate features:
        1. Register hook on desired stage
        2. Run full forward pass through network
        3. Hook captures output of stage_idx
        4. Return captured features
        
        This is necessary because PVT doesn't expose stages as sequential
        modules like VGG or ResNet.
    """
    
    def __init__(self, architecture='pvt_v2_b2', stage_name='stage2'):
        super().__init__()
        
        # Load pre-trained PVT from timm library
        # PVT-v2 improves on original PVT with:
        # - Overlapping patch embedding (better than non-overlapping)
        # - Linear complexity attention (vs quadratic)
        # - Convolutional feed-forward networks
        self.pvt = timm.create_model(architecture, pretrained=True)
        
        # Parse stage index from stage name (e.g., 'stage2' → 2)
        # Subtract 1 because stages are 0-indexed but user inputs are 1-based
        # PVT has 4 stages (indices 0-3), so 'stage1' → stages[0]
        if stage_name.startswith('stage'):
            self.stage_idx = int(stage_name.replace('stage', '')) - 1
        else:
            # Fallback: default to stage 2 (index 1)
            self.stage_idx = 1  # stages[1] = stage2
        
        # Storage for features captured by forward hook
        self.features = None
        
        # Register forward hook to capture intermediate features
        # Lambda function is called after the stage processes input
        # It stores the output in self.features
        # Args: m (module), i (input), o (output)
        self.pvt.stages[self.stage_idx].register_forward_hook(
            lambda m, i, o: setattr(self, 'features', o)
        )
        
        # Freeze all parameters
        self._freeze()
    
    def _freeze(self):
        """
        Freeze all PVT parameters and set to evaluation mode.
        
        PVT has significant parameter count (25M for PVT-v2-B2):
        - Patch embedding layers
        - Multiple transformer blocks per stage (3-6 blocks)
        - Spatial reduction attention mechanisms
        - Feed-forward networks
        
        All frozen to use as feature extractor only.
        """
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def forward(self, x):
        """
        Extract features from input images using forward hooks.
        
        Process:
        1. Reset features storage
        2. Run complete forward pass through PVT
        3. Forward hook captures output of stage_idx
        4. Return captured features
        
        Args:
            x (torch.Tensor): Input images, shape (B, 3, 224, 224)
        
        Returns:
            torch.Tensor: Extracted features, shape depends on stage_name:
                         - stage1: (B, 64, 56, 56)
                         - stage2: (B, 128, 28, 28)
                         - stage3: (B, 320, 14, 14)
                         - stage4: (B, 512, 7, 7)
        
        Note:
            The forward hook captures features during the forward pass.
            We don't actually need the final output (_ = self.pvt(x)),
            only the intermediate features stored by the hook.
        """
        # Reset feature storage
        self.features = None
        
        # Run forward pass - hook will capture features at stage_idx
        # Discard final output (_) since we only need intermediate features
        _ = self.pvt(x)
        
        # Return features captured by forward hook
        return self.features


# ============================================================================
# DECODERS (TRAINABLE RECONSTRUCTION NETWORKS)
# ============================================================================
# These networks reconstruct RGB images from extracted features.
# Only the decoder is trained - encoders remain frozen.
# Four decoder architectures tested with varying complexity.

class FrequencyAwareDecoder(nn.Module):
    """
    Decoder with explicit frequency decomposition and spatial-channel attention.
    
    This is the most complex decoder architecture, based on the hypothesis that
    image reconstruction benefits from explicit modeling of low and high frequency
    components. It uses dual pathways for frequency processing combined with
    multi-scale refinement.
    
    Architecture Complexity: HIGH
    - Parameters: ~35M (most complex decoder tested)
    - Computation: High due to multiple attention mechanisms
    - Components: Frequency separation, attention blocks, multi-scale refinement
    
    Design Motivation:
    - Neural networks naturally encode features at different frequencies
    - Explicit separation might help decoder learn frequency-specific processing
    - Low freq: Color, gradual transitions, overall structure
    - High freq: Edges, textures, fine details
    - Attention mechanisms select important spatial locations and channels
    
    Architecture Pipeline:
        Input features (C × H × W)
        ↓ 
        Spatial-Channel Attention Blocks (2×)
        ↓
        Split into Low & High Frequency Pathways
        ↓
        Frequency Fusion with Attention
        ↓
        Progressive Upsampling (stride-2, multiple stages)
        ↓
        Multi-Scale Refinement (3 parallel scales: 3×3, 5×5, 7×7)
        ↓
        Color Enhancement Block
        ↓
        RGB Output (224 × 224 × 3)
    
    Experimental Findings:
    - Despite complexity, does NOT outperform simple TransposedConvDecoder
    - Hypothesis: Explicit frequency modeling may be redundant
    - Network learns its own frequency representations implicitly
    - Added complexity increases overfitting risk
    - Higher computation cost without performance gains
    
    Args:
        input_channels (int): Number of channels in input features
                             Examples: 64 (VGG block1), 128 (ResNet layer2)
        
        input_size (int): Spatial dimension of input features (assumes square)
                         Examples: 112 (VGG block1), 56 (ResNet layer1), 14 (ViT)
        
        output_size (int): Target spatial dimension for reconstructed image
                          Default: 224 (matches ImageNet resolution)
    
    Input:
        x (torch.Tensor): Extracted features, shape (B, C, H, W)
    
    Output:
        reconstructed (torch.Tensor): RGB images, shape (B, 3, 224, 224)
                                     Values in range [0, 1] (Sigmoid output)
    
    Example:
        >>> # VGG16 block1 features: 64 channels, 112×112 spatial
        >>> decoder = FrequencyAwareDecoder(64, 112, 224)
        >>> features = torch.randn(4, 64, 112, 112)
        >>> images = decoder(features)  # (4, 3, 224, 224)
    
    Note:
        Despite being theoretically motivated, this decoder is rarely the best
        performer. Consider using TransposedConvDecoder for better results with
        lower complexity. This highlights that "simpler is better" in this task.
    """
    
    def __init__(self, input_channels, input_size, output_size=224):
        super().__init__()
        
        # Low frequency pathway: Captures gradual color changes and structure
        # Uses 3×3 convolutions (small receptive field for smooth features)
        # Reduces channels by 2× to split capacity between low/high freq
        self.low_freq = nn.Sequential(
            nn.Conv2d(input_channels, input_channels//2, 3, padding=1),
            nn.BatchNorm2d(input_channels//2),
            nn.ReLU(inplace=True)
        )
        
        # High frequency pathway: Captures edges, textures, fine details
        # Also uses 3×3 convs (same receptive field, different learned features)
        # Network learns to specialize each pathway during training
        self.high_freq = nn.Sequential(
            nn.Conv2d(input_channels, input_channels//2, 3, padding=1),
            nn.BatchNorm2d(input_channels//2),
            nn.ReLU(inplace=True)
        )
        
        # Fusion module: Combines low and high frequency components
        # Uses channel and spatial attention to weight contributions
        self.freq_fusion = FrequencyFusionBlock(input_channels//2)
        
        # Spatial-channel attention: Applied before frequency split
        # Helps network focus on informative spatial locations and channels
        # 2 sequential blocks for iterative refinement
        self.attn_blocks = nn.ModuleList([
            SpatialChannelAttentionBlock(input_channels) for _ in range(2)
        ])
        
        # Progressive upsampling: Increase spatial resolution to output_size
        # Each block: 2× spatial upsampling + channel reduction
        # Continues until reaching 224×224 target resolution
        self.upsample_blocks = nn.ModuleList()
        current_size = input_size
        in_ch = input_channels
        
        while current_size < output_size:
            # Determine output channels (halve channels, min 64)
            out_ch = max(in_ch//2, 64)
            
            # Add upsampling residual block (transposed conv + skip connection)
            self.upsample_blocks.append(UpsampleResBlock(in_ch, out_ch))
            
            # Update for next iteration
            in_ch = out_ch
            current_size = current_size * 2  # Spatial resolution doubles
        
        # Multi-scale refinement: Parallel processing at different scales
        # Captures both fine details and broader context
        self.refinement = MultiScaleRefinement(in_ch)
        
        # Color enhancement: Adjusts color distribution for natural appearance
        # Uses instance normalization for style transfer-like color correction
        self.color_enhance = ColorEnhancementBlock(in_ch)
        
        # Final RGB projection: Convert features to 3-channel image
        # Tanh activation produces values in [-1, 1], shifted to [0, 1]
        self.to_rgb = nn.Sequential(
            nn.Conv2d(in_ch, 3, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        """
        Reconstruct image from features through frequency-aware processing.
        
        Pipeline:
        1. Apply spatial-channel attention (2×)
        2. Split into low/high frequency pathways
        3. Fuse frequency components with attention
        4. Progressive upsampling to target resolution
        5. Multi-scale refinement
        6. Color enhancement
        7. Convert to RGB and normalize to [0, 1]
        
        Args:
            x (torch.Tensor): Input features, shape (B, C, H, W)
        
        Returns:
            torch.Tensor: Reconstructed RGB images, shape (B, 3, 224, 224)
                         Values in range [0, 1]
        """
        # Step 1: Apply attention blocks for spatial-channel refinement
        for attn in self.attn_blocks:
            x = attn(x)
        
        # Step 2: Split into frequency pathways
        low = self.low_freq(x)   # Smooth, gradual features
        high = self.high_freq(x)  # Sharp, detailed features
        
        # Step 3: Fuse frequency components with attention weighting
        x = self.freq_fusion(low, high)
        
        # Step 4: Progressive upsampling to target resolution
        for block in self.upsample_blocks:
            x = block(x)  # 2× spatial increase per block
        
        # Step 5: Multi-scale refinement for detail recovery
        x = self.refinement(x)
        
        # Step 6: Color enhancement for natural appearance
        x = self.color_enhance(x)
        
        # Step 7: Convert to RGB and normalize
        # Tanh: [-1, 1] → Add 1: [0, 2] → Divide by 2: [0, 1]
        return (self.to_rgb(x) + 1) / 2


class WaveletFrequencyDecoder(nn.Module):
    """
    Decoder with wavelet-based multi-resolution reconstruction.
    
    This decoder is motivated by wavelet decomposition theory, which suggests
    that images can be efficiently represented as combinations of multi-scale
    frequency subbands. The network explicitly predicts four wavelet subbands
    (LL, LH, HL, HH) which are then fused for reconstruction.
    
    Architecture Complexity: MEDIUM-HIGH
    - Parameters: ~8M (moderate complexity)
    - Computation: Moderate, 4 parallel subband predictors
    - Components: Wavelet subband prediction, band fusion, upsampling
    
    Wavelet Transform Background:
    - LL (Low-Low): Approximation, overall image structure
    - LH (Low-High): Horizontal edges and details
    - HL (High-Low): Vertical edges and details  
    - HH (High-High): Diagonal edges and corners
    
    Design Motivation:
    - Neural networks may implicitly encode features in wavelet-like manner
    - Explicit wavelet prediction could align with network's natural representation
    - Multi-resolution approach mimics coarse-to-fine reconstruction
    - Wavelet coefficients are decorrelated, potentially easier to predict
    
    Architecture Pipeline:
        Input features (C × H × W)
        ↓
        4 Parallel Subband Predictors (LL, LH, HL, HH)
        ↓
        Concatenate: 64 channels each → 256 total
        ↓
        Frequency Band Fusion Blocks (2×)
        ↓
        Refinement: Reduce to 128 channels
        ↓
        Progressive Upsampling (stride-2, multiple stages)
        ↓
        Multi-Scale Refinement
        ↓
        Color Enhancement
        ↓
        RGB Output (224 × 224 × 3)
    
    Experimental Findings:
    - Performs moderately well but not better than TransposedConvDecoder
    - Explicit wavelet modeling doesn't provide clear advantage
    - Network may already learn wavelet-like features implicitly
    - Added complexity without corresponding performance gain
    - Interesting theoretical motivation but empirically not optimal
    
    Args:
        input_channels (int): Number of channels in input features
        input_size (int): Spatial dimension of input features (square)
        output_size (int): Target dimension for reconstructed image (default: 224)
    
    Input:
        x (torch.Tensor): Extracted features, shape (B, C, H, W)
    
    Output:
        reconstructed (torch.Tensor): RGB images, shape (B, 3, 224, 224)
                                     Values in range [0, 1]
    
    Example:
        >>> decoder = WaveletFrequencyDecoder(64, 56, 224)
        >>> features = torch.randn(4, 64, 56, 56)
        >>> images = decoder(features)  # (4, 3, 224, 224)
    """
    
    def __init__(self, input_channels, input_size, output_size=224):
        super().__init__()
        
        # Four parallel subband predictors (one for each wavelet component)
        # Each predicts 64-channel representation of its subband
        # Tanh activation bounds predictions to [-1, 1] for stability
        self.to_ll = self._band_predictor(input_channels, 64)  # Approximation
        self.to_lh = self._band_predictor(input_channels, 64)  # Horizontal details
        self.to_hl = self._band_predictor(input_channels, 64)  # Vertical details
        self.to_hh = self._band_predictor(input_channels, 64)  # Diagonal details
        
        # Band fusion modules: Combine wavelet subbands with attention
        # Two sequential fusion blocks for iterative refinement
        # Each block applies channel attention to weight subband contributions
        self.band_fusion = nn.ModuleList([
            FrequencyBandFusion(64) for _ in range(2)
        ])
        
        # Frequency refinement: Process fused subbands
        # Input: 256 channels (4 subbands × 64 each)
        # Output: 128 channels (reduced for upsampling efficiency)
        self.freq_refine = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Progressive upsampling: Increase spatial resolution
        # Similar to FrequencyAwareDecoder upsampling strategy
        self.upsample_blocks = nn.ModuleList()
        current_size = input_size
        in_ch = 128
        
        while current_size < output_size:
            out_ch = max(in_ch//2, 64)
            self.upsample_blocks.append(UpsampleResBlock(in_ch, out_ch))
            in_ch = out_ch
            current_size = current_size * 2
        
        # Multi-scale refinement for detail recovery
        self.refinement = MultiScaleRefinement(in_ch)
        
        # Color enhancement for natural appearance
        self.color_enhance = ColorEnhancementBlock(in_ch)
        
        # Final RGB projection with Tanh activation
        self.to_rgb = nn.Sequential(
            nn.Conv2d(in_ch, 3, 3, padding=1),
            nn.Tanh()
        )
    
    def _band_predictor(self, in_ch, out_ch):
        """
        Create a wavelet subband predictor network.
        
        Each predictor is a small CNN that transforms input features
        into one of the four wavelet subbands. The network learns to
        separate frequency information into appropriate subbands.
        
        Args:
            in_ch (int): Input channels from feature extractor
            out_ch (int): Output channels for subband (typically 64)
        
        Returns:
            nn.Sequential: Subband prediction network
        """
        return nn.Sequential(
            # Expand channels for richer representation
            nn.Conv2d(in_ch, out_ch*2, 3, padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(inplace=True),
            # Compress to target channels with Tanh for bounded output
            nn.Conv2d(out_ch*2, out_ch, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        """
        Reconstruct image through wavelet subband prediction and fusion.
        
        Pipeline:
        1. Predict four wavelet subbands in parallel (LL, LH, HL, HH)
        2. Concatenate subbands: 4 × 64 = 256 channels
        3. Apply band fusion with attention (2×)
        4. Refine fused representation
        5. Progressive upsampling
        6. Multi-scale refinement
        7. Color enhancement
        8. Convert to RGB
        
        Args:
            x (torch.Tensor): Input features, shape (B, C, H, W)
        
        Returns:
            torch.Tensor: Reconstructed RGB images, shape (B, 3, 224, 224)
        """
        # Step 1: Predict all four wavelet subbands in parallel
        # Each subband captures different frequency and orientation information
        ll = self.to_ll(x)  # Low-low: Approximation
        lh = self.to_lh(x)  # Low-high: Horizontal edges
        hl = self.to_hl(x)  # High-low: Vertical edges
        hh = self.to_hh(x)  # High-high: Diagonal edges
        
        # Step 2: Concatenate subbands along channel dimension
        # (B, 64, H, W) × 4 → (B, 256, H, W)
        bands = torch.cat([ll, lh, hl, hh], dim=1)
        
        # Step 3: Fuse subbands with attention-based weighting
        for fusion in self.band_fusion:
            bands = fusion(bands)
        
        # Step 4: Refine fused frequency representation
        x = self.freq_refine(bands)
        
        # Step 5: Progressive upsampling to target resolution
        for block in self.upsample_blocks:
            x = block(x)
        
        # Step 6: Multi-scale refinement for detail recovery
        x = self.refinement(x)
        
        # Step 7: Color enhancement
        x = self.color_enhance(x)
        
        # Step 8: Convert to RGB and normalize to [0, 1]
        return (self.to_rgb(x) + 1) / 2


class TransposedConvDecoder(nn.Module):
    """
    Simple decoder using only transposed convolutions and batch normalization.
    
    This is the SIMPLEST decoder architecture tested - just stride-2 transposed
    convolutions with batch norm and ReLU, progressively upsampling from input
    features to 224×224 RGB images.
    
    Architecture Complexity: LOW
    - Parameters: ~0.2M (smallest decoder tested)
    - Computation: Fast, no attention mechanisms or complex modules
    - Components: Just transposed convs + BatchNorm + ReLU
    
    Design Philosophy:
    - Minimal architectural complexity
    - Let the data and optimization do the work
    - No explicit frequency modeling or attention
    - Progressive 2× upsampling until target resolution
    - Channel reduction at each upsampling stage
    
    Architecture Pipeline:
        Input features (C × H × W)
        ↓
        [Loop until reaching 224×224:]
          Transposed Conv (4×4 kernel, stride 2) → 2× spatial size
          Batch Normalization
          ReLU
          Channels reduced by 2× (min 32)
        ↓
        Final Conv (3×3) → RGB projection
        Sigmoid → [0, 1] normalization
    
    **CRITICAL EXPERIMENTAL FINDING:**
    Despite being the simplest decoder, TransposedConvDecoder achieves
    the BEST or near-best results across most experiments!
    
    Top 10 Models Analysis:
    - 4/10 top models use TransposedConvDecoder
    - VGG16 block1 + TransposedConv = 17.35 dB (best single model)
    - Only 0.29 dB behind best ensemble (17.64 dB)
    - Significantly faster than complex decoders
    
    Why Simple Works Best:
    1. Fewer parameters → Less overfitting on small dataset (640 training images)
    2. No explicit assumptions → Network learns optimal reconstruction path
    3. Direct gradient flow → No complex attention bottlenecks
    4. Computational efficiency → Faster training and inference
    5. Generalization → Simple patterns generalize better than complex ones
    
    This finding supports the principle: "Make things as simple as possible,
    but not simpler" - Einstein. For this task, transposed convolutions are
    sufficient; additional complexity doesn't help.
    
    Args:
        input_channels (int): Number of channels in input features
                             Examples: 64 (VGG block1), 384 (ViT)
        
        input_size (int): Spatial dimension of input features (square)
                         Examples: 112 (VGG block1), 56 (ResNet layer1)
        
        output_size (int): Target dimension for reconstruction (default: 224)
    
    Input:
        x (torch.Tensor): Extracted features, shape (B, C, H, W)
    
    Output:
        reconstructed (torch.Tensor): RGB images, shape (B, 3, 224, 224)
                                     Values in range [0, 1]
    
    Example:
        >>> # Best configuration: VGG16 block1 + TransposedConv
        >>> decoder = TransposedConvDecoder(64, 112, 224)
        >>> features = torch.randn(4, 64, 112, 112)
        >>> images = decoder(features)  # (4, 3, 224, 224)
        >>> # Achieves 17.35 dB PSNR!
    
    Upsampling Example (VGG16 block1):
        Input: 64 channels, 112×112
        ↓ TransposedConv (stride 2): 32 channels, 224×224
        ↓ Final Conv: 3 channels, 224×224
        Total stages: 1 (already near target resolution!)
    
    Upsampling Example (ViT block1):
        Input: 384 channels, 14×14
        ↓ TransposedConv: 192 channels, 28×28
        ↓ TransposedConv: 96 channels, 56×56
        ↓ TransposedConv: 48 channels, 112×112
        ↓ TransposedConv: 32 channels, 224×224
        ↓ Final Conv: 3 channels, 224×224
        Total stages: 4 (more upsampling needed)
    
    Note:
        This decoder's success challenges the assumption that complex tasks
        require complex architectures. Often, simple solutions work best!
    """
    
    def __init__(self, input_channels, input_size, output_size=224):
        super().__init__()
        
        # Build progressive upsampling layers
        layers = []
        current_size = input_size
        in_ch = input_channels
        
        # Keep upsampling until we reach target resolution
        while current_size < output_size:
            # Calculate output channels (halve channels, minimum 32)
            # Gradually reduce feature complexity as we increase spatial size
            out_ch = max(in_ch // 2, 32)
            
            # Transposed convolution: 2× spatial upsampling
            # kernel=4, stride=2, padding=1 → exact 2× upsampling
            # This is mathematically equivalent to:
            # output_size = (input_size - 1) * stride - 2*padding + kernel
            #             = (input_size - 1) * 2 - 2 + 4
            #             = input_size * 2
            layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),  # Normalize for stable training
                nn.ReLU(inplace=True)     # Non-linearity
            ])
            
            # Update for next iteration
            in_ch = out_ch
            current_size = current_size * 2
        
        # Final layer: Convert features to RGB
        # 3×3 conv for spatial smoothing before final projection
        # Sigmoid ensures output in [0, 1] range
        layers.extend([
            nn.Conv2d(in_ch, 3, kernel_size=3, padding=1),  # → RGB channels
            nn.Sigmoid()  # → [0, 1] range
        ])
        
        # Combine all layers into sequential module
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Reconstruct image through simple progressive upsampling.
        
        This forward pass is remarkably simple compared to other decoders:
        - No attention mechanisms
        - No frequency decomposition
        - No multi-scale processing
        - Just: upsample → normalize → activate → repeat
        
        Yet it achieves the best results! This is a powerful lesson in
        the effectiveness of simplicity.
        
        Args:
            x (torch.Tensor): Input features, shape (B, C, H, W)
        
        Returns:
            torch.Tensor: Reconstructed RGB images, shape (B, 3, 224, 224)
                         Values in [0, 1] range
        """
        return self.decoder(x)


class AttentionDecoder(nn.Module):
    """
    Decoder using Transformer blocks with self-attention before upsampling.
    
    This decoder applies multi-head self-attention mechanisms (similar to
    Transformers used in ViT and language models) before upsampling. The
    hypothesis is that self-attention can capture long-range dependencies
    and resolve spatial ambiguities in reconstruction.
    
    Architecture Complexity: HIGH
    - Parameters: ~12M (high due to attention + upsampling)
    - Computation: Expensive, O(N²) attention complexity
    - Components: Transformer blocks, transposed conv upsampling
    
    Transformer/Attention Background:
    - Self-attention: Each spatial location attends to all others
    - Multi-head: Multiple parallel attention mechanisms learn different patterns
    - Global receptive field: Unlike CNNs, attention sees entire spatial extent
    - Position-agnostic: Treats spatial locations as a set (no inherent position)
    
    Design Motivation:
    - Long-range dependencies: Attention can relate distant spatial locations
    - Holistic processing: See entire feature map, not just local patches
    - Flexibility: Learned attention patterns adapt to input
    - Success in vision: ViT showed transformers work well for images
    
    Architecture Pipeline:
        Input features (C × H × W)
        ↓
        Flatten to sequence: (C × H × W) → (H*W × C)
        ↓
        Transformer Blocks (4×):
          - Multi-head Self-Attention
          - Layer Normalization
          - Feed-Forward MLP
          - Residual connections
        ↓
        Reshape to spatial: (H*W × C) → (C × H × W)
        ↓
        Progressive Upsampling (TransposedConv + BatchNorm + ReLU)
        ↓
        RGB Output (224 × 224 × 3)
    
    Experimental Findings:
    - Performance similar to FrequencyAware decoder (moderate)
    - Does NOT outperform simple TransposedConvDecoder
    - High computational cost (quadratic in spatial size)
    - May be bottlenecked by limited spatial resolution of input features
    - Attention is powerful but not necessary for this reconstruction task
    
    Why Attention Doesn't Excel Here:
    1. Reconstruction is largely local - nearby features matter most
    2. CNNs' local inductive bias is actually helpful for images
    3. Input features already have spatial structure from encoder
    4. Global dependencies may not be critical for pixel-level reconstruction
    5. Computational cost outweighs marginal benefits
    
    Comparison with TransposedConvDecoder:
    - TransposedConv: Simpler, faster, better results
    - Attention: More complex, slower, similar or worse results
    - Lesson: Match architecture complexity to task requirements
    
    Args:
        input_channels (int): Number of channels in input features
        input_size (int): Spatial dimension of input features (square)
        output_size (int): Target dimension for reconstruction (default: 224)
        num_blocks (int): Number of Transformer blocks to stack (default: 4)
    
    Input:
        x (torch.Tensor): Extracted features, shape (B, C, H, W)
    
    Output:
        reconstructed (torch.Tensor): RGB images, shape (B, 3, 224, 224)
                                     Values in [0, 1]
    
    Example:
        >>> decoder = AttentionDecoder(64, 56, 224, num_blocks=4)
        >>> features = torch.randn(4, 64, 56, 56)  # 56×56 = 3,136 tokens
        >>> images = decoder(features)  # (4, 3, 224, 224)
    
    Computational Complexity:
        Attention complexity: O(N² × D) where N = H×W spatial locations
        
        Examples:
        - VGG block1: 112×112 = 12,544 tokens → 157M attention operations!
        - ResNet layer1: 56×56 = 3,136 tokens → 10M attention operations
        - ViT block1: 14×14 = 196 tokens → 38K attention operations
        
        This is why attention decoders are slow for high-resolution features.
    
    Note:
        While Transformer architectures revolutionized NLP and have shown
        promise in vision, they're not always the optimal choice. For spatial
        reconstruction tasks, simpler CNN-based approaches often work better.
    """
    
    def __init__(self, input_channels, input_size, output_size=224, num_blocks=4):
        super().__init__()
        
        # Transformer blocks: Process features with self-attention
        # Each block contains:
        # - Multi-head self-attention (global receptive field)
        # - Layer normalization (stabilizes training)
        # - Feed-forward MLP (channel-wise processing)
        # - Residual connections (enables deeper stacks)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(input_channels) for _ in range(num_blocks)
        ])
        
        # Progressive upsampling: Increase spatial resolution after attention
        # Uses same strategy as TransposedConvDecoder
        layers = []
        current_size = input_size
        in_ch = input_channels
        
        while current_size < output_size:
            out_ch = max(in_ch // 2, 32)
            layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ])
            in_ch = out_ch
            current_size = current_size * 2
        
        # Final RGB projection with Sigmoid
        layers.extend([
            nn.Conv2d(in_ch, 3, 3, padding=1),
            nn.Sigmoid()
        ])
        
        self.upsampler = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Reconstruct image through attention processing and upsampling.
        
        Pipeline:
        1. Flatten spatial dimensions to sequence format (B, C, H, W) → (B, H*W, C)
        2. Apply Transformer blocks (self-attention + MLP) × num_blocks
        3. Reshape back to spatial format (B, H*W, C) → (B, C, H, W)
        4. Progressive upsampling with transposed convolutions
        5. Final RGB projection
        
        Args:
            x (torch.Tensor): Input features, shape (B, C, H, W)
        
        Returns:
            torch.Tensor: Reconstructed RGB images, shape (B, 3, 224, 224)
        """
        B, C, H, W = x.shape
        
        # Step 1: Flatten spatial dimensions to sequence
        # (B, C, H, W) → (B, C, H*W) → (B, H*W, C)
        # This converts spatial grid to sequence of tokens for attention
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Step 2: Process with Transformer blocks
        # Each token (spatial location) attends to all other tokens
        # This captures global context but is computationally expensive
        for block in self.transformer_blocks:
            x = block(x)  # Self-attention + MLP with residuals
        
        # Step 3: Reshape back to spatial format
        # (B, H*W, C) → (B, C, H*W) → (B, C, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        # Step 4 & 5: Upsample to target resolution and convert to RGB
        return self.upsampler(x)


# ============================================================================
# SINGLE ARCHITECTURE MODEL
# ============================================================================
# Combines a frozen encoder with a trainable decoder for reconstruction.

class SingleArchModel(nn.Module):
    """
    Single architecture model: Frozen Encoder + Trainable Decoder.
    
    This is the core model architecture for single-encoder experiments. It
    consists of two components:
    
    1. Frozen Encoder: Pre-trained CNN or Transformer
       - Extracts intermediate features from input images
       - Weights are frozen (no gradient updates)
       - Preserves ImageNet-learned representations
    
    2. Trainable Decoder: Custom reconstruction network
       - Learns to invert encoder features back to images
       - Only component trained during experiments
       - Choice of decoder impacts reconstruction quality
    
    Model Pipeline:
        Input Image (224×224×3)
              ↓
        Frozen Encoder (ResNet/VGG/ViT/PVT)
              ↓
        Intermediate Features (C×H×W)
              ↓
        Trainable Decoder
              ↓
        Reconstructed Image (224×224×3)
    
    Supported Architectures:
        - ResNet34: Residual CNN (layer1, layer2)
        - VGG16: Sequential CNN (block1, block3)
        - ViT-Small: Vision Transformer (block1, block3)
        - PVT-v2-B2: Pyramid Vision Transformer (stage1, stage2)
    
    Supported Decoders:
        - FrequencyAwareDecoder: Explicit frequency modeling (complex)
        - WaveletFrequencyDecoder: Wavelet subband prediction (medium)
        - TransposedConvDecoder: Simple transposed convs (BEST!)
        - AttentionDecoder: Transformer-based (complex)
    
    Best Configuration (from experiments):
        Architecture: VGG16
        Layer: block1 (112×112×64)
        Decoder: TransposedConvDecoder
        Performance: 17.35 dB PSNR, 0.560 SSIM
    
    Why This Configuration Works:
    1. VGG block1 has highest spatial resolution (112×112)
    2. Early features preserve color and texture
    3. Simple decoder avoids overfitting
    4. Balanced parameter count (~5M frozen + 0.2M trainable)
    
    Training Strategy:
        - Only decoder parameters have requires_grad=True
        - Encoder in eval() mode (no batchnorm updates, no dropout)
        - Loss: 0.5×MSE + 0.5×LPIPS (pixel + perceptual)
        - Optimizer: Adam with learning rate 1e-4
        - Early stopping: Patience 15 epochs
    
    Args:
        config (dict): Configuration dictionary containing:
            - architecture (str): 'resnet34', 'vgg16', 'vit_small_patch16_224', 'pvt_v2_b2'
            - Layer specification (str): One of:
                * resnet_layer: 'layer1', 'layer2'
                * vgg_block: 'block1', 'block3'
                * vit_block: 'block1', 'block3'
                * pvt_stage: 'stage1', 'stage2'
            - decoder_type (str): 'frequency_aware', 'wavelet', 'transposed_conv', 'attention'
            - output_size (int): Target image size (default: 224)
    
    Example:
        >>> config = {
        ...     'architecture': 'vgg16',
        ...     'vgg_block': 'block1',
        ...     'decoder_type': 'transposed_conv',
        ...     'output_size': 224
        ... }
        >>> model = SingleArchModel(config)
        >>> images = torch.randn(4, 3, 224, 224)
        >>> reconstructed = model(images)  # (4, 3, 224, 224)
    
    Output Information:
        During initialization, prints:
        - Architecture name
        - Feature shape (channels and spatial size)
        - Decoder type
        This helps verify correct model configuration.
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Extract architecture name from config
        arch = config['architecture']
        
        # Print model header for logging
        print(f"\n{'#'*70}")
        print(f"  SINGLE MODEL: {arch.upper()}")
        print(f"{'#'*70}\n")
        
        # ====================================================================
        # CREATE ENCODER (frozen feature extractor)
        # ====================================================================
        # Choose appropriate encoder class based on architecture
        if arch == 'resnet34':
            # ResNet34 with layer selection (layer1 or layer2)
            self.encoder = ResNetExtractor(config['resnet_layer'])
            
        elif arch == 'vgg16':
            # VGG16 with block selection (block1, block3, etc.)
            self.encoder = VGGExtractor(config['vgg_block'])
            
        elif arch.startswith('vit'):
            # Vision Transformer with block selection
            # arch contains full model name (e.g., 'vit_small_patch16_224')
            self.encoder = ViTExtractor(arch, config['vit_block'])
            
        elif arch.startswith('pvt'):
            # Pyramid Vision Transformer with stage selection
            # arch contains full model name (e.g., 'pvt_v2_b2')
            self.encoder = PVTExtractor(arch, config['pvt_stage'])
            
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        # ====================================================================
        # DETERMINE FEATURE DIMENSIONS
        # ====================================================================
        # Run a dummy forward pass to automatically determine feature dimensions
        # This avoids hardcoding dimensions for each architecture/layer combination
        with torch.no_grad():  # No gradients needed for dummy pass
            dummy = torch.randn(1, 3, 224, 224)  # Single 224×224 RGB image
            features = self.encoder(dummy)       # Extract features
            input_channels = features.shape[1]   # Number of channels (C)
            input_size = features.shape[2]       # Spatial dimension (H=W)
        
        # Print feature information for verification
        print(f"Feature shape: {features.shape}")
        print(f"Input channels: {input_channels}, Spatial size: {input_size}")
        
        # ====================================================================
        # CREATE DECODER (trainable reconstruction network)
        # ====================================================================
        # Map decoder type strings to decoder classes
        decoder_map = {
            'frequency_aware': FrequencyAwareDecoder,
            'wavelet': WaveletFrequencyDecoder,
            'transposed_conv': TransposedConvDecoder,
            'attention': AttentionDecoder
        }
        
        # Get decoder class and instantiate
        decoder_class = decoder_map[config['decoder_type']]
        self.decoder = decoder_class(
            input_channels,
            input_size,
            config['output_size']
        )
        
        # Print decoder information
        print(f"Decoder: {config['decoder_type']}")
        print(f"{'#'*70}\n")
    
    def forward(self, x):
        """
        Forward pass: Extract features and reconstruct image.
        
        Pipeline:
        1. Input images pass through frozen encoder
        2. Extracted features pass through trainable decoder
        3. Return reconstructed images
        
        Args:
            x (torch.Tensor): Input images, shape (B, 3, 224, 224)
        
        Returns:
            torch.Tensor: Reconstructed images, shape (B, 3, 224, 224)
                         Values in range [0, 1]
        
        Note:
            Encoder is in eval() mode and frozen, so no batchnorm updates
            or gradient computation occurs in encoder. Only decoder receives
            gradients during training.
        """
        features = self.encoder(x)      # (B, C, H, W) - frozen
        reconstructed = self.decoder(features)  # (B, 3, 224, 224) - trainable
        return reconstructed


# ============================================================================
# ENSEMBLE MODEL
# ============================================================================
# Combines multiple frozen encoders with feature fusion and a trainable decoder.

class EnsembleModel(nn.Module):
    """
    Ensemble model: Multiple Frozen Encoders + Feature Fusion + Trainable Decoder.
    
    This architecture combines features from multiple pre-trained networks,
    hypothesizing that different architectures capture complementary information:
    - ResNet: Residual features, strong for classification
    - VGG: Simple sequential features, good spatial information
    - ViT: Global attention features, semantic understanding
    - PVT: Hierarchical transformer features, multi-scale context
    
    Model Pipeline:
        Input Image (224×224×3)
              ↓
        ┌──────────────┬───────────────┬──────────────┬──────────────┐
        │  ResNet34    │  VGG16        │  ViT-Small   │  PVT-v2-B2   │
        │  (frozen)    │  (frozen)     │  (frozen)    │  (frozen)    │
        └──────────────┴───────────────┴──────────────┴──────────────┘
              ↓              ↓              ↓              ↓
        Features:    64×56×56   64×112×112   384×14×14    64×56×56
              ↓              ↓              ↓              ↓
              └──────────────┴───────────────┴──────────────┘
                              ↓
                      Feature Fusion Module
                      (Attention/Concat/Weighted)
                              ↓
                      Fused Features (256×H×W)
                              ↓
                      Trainable Decoder
                              ↓
                      Reconstructed Image (224×224×3)
    
    Fusion Strategies:
        1. Attention: Learned attention weights per architecture
           - Most sophisticated: learns importance of each network spatially
           - Network: Global pooling → Dense → Softmax → Weighted sum
        
        2. Concatenation: Simple channel-wise stacking
           - Simplest: just concatenate all features along channel dim
           - Network: Conv layers to reduce concatenated channels
        
        3. Weighted: Learnable scalar weights per architecture
           - Middle ground: single weight per network (not spatial)
           - Network: Learnable scalars → Softmax → Weighted sum
    
    Experimental Findings:
        Best ensemble: Weighted fusion + TransposedConv decoder
        Performance: 17.64 dB PSNR, 0.586 SSIM
        
        Comparison with best single model:
        - Ensemble: 17.64 dB (4 encoders, ~20M frozen params)
        - VGG16 alone: 17.35 dB (1 encoder, ~5M frozen params)
        - Gain: Only +0.29 dB (+1.7%) with 4× computational cost!
    
    Why Ensembles Don't Help Much:
    1. Architectures capture similar low-level features for reconstruction
    2. VGG16 block1 already captures necessary information (high resolution)
    3. Fusion adds complexity without addressing fundamental limitations
    4. Spatial resolution is key constraint, not feature diversity
    5. Marginal gains don't justify added computation
    
    Computational Cost:
        Single model: 1× forward pass per image
        Ensemble: 4× forward passes + fusion overhead
        Training time: ~4× longer than single model
        Inference time: ~4× slower than single model
    
    When Ensembles Might Help:
        - Very large datasets (reduce overfitting)
        - High resolution inputs (diverse features at multiple scales)
        - Task requiring both local and global information
        - Computational resources not a constraint
    
    Args:
        config (dict): Configuration dictionary containing:
            - architectures (list): List of architecture names, e.g.,
                                   ['resnet34', 'vgg16', 'vit_small_patch16_224', 'pvt_v2_b2']
            - Layer specifications (one per architecture):
                * resnet_layer: 'layer1', 'layer2'
                * vgg_block: 'block1', 'block3'
                * vit_block: 'block1', 'block3'
                * pvt_stage: 'stage1', 'stage2'
            - fusion_strategy (str): 'attention', 'concat', or 'weighted'
            - fusion_channels (int): Target channels after fusion (typically 256)
            - decoder_type (str): 'frequency_aware', 'wavelet', 'transposed_conv', 'attention'
            - output_size (int): Target image size (default: 224)
    
    Example:
        >>> config = {
        ...     'architectures': ['resnet34', 'vgg16', 'vit_small_patch16_224'],
        ...     'resnet_layer': 'layer1',
        ...     'vgg_block': 'block1',
        ...     'vit_block': 'block1',
        ...     'fusion_strategy': 'weighted',
        ...     'fusion_channels': 256,
        ...     'decoder_type': 'transposed_conv',
        ...     'output_size': 224
        ... }
        >>> model = EnsembleModel(config)
        >>> images = torch.randn(4, 3, 224, 224)
        >>> reconstructed = model(images)  # (4, 3, 224, 224)
    
    Output Information:
        During initialization, prints:
        - Architecture names
        - Feature dimensions per architecture
        - Target spatial size after alignment
        - Fusion strategy
        - Decoder type
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Print ensemble header for logging
        print(f"\n{'#'*70}")
        print(f"  ENSEMBLE MODEL")
        print(f"{'#'*70}\n")
        
        # Store architecture list
        self.architectures = config['architectures']
        
        # ====================================================================
        # CREATE MULTIPLE ENCODERS (all frozen)
        # ====================================================================
        # Use ModuleDict for organized storage of encoders
        # Key: architecture nickname (e.g., 'resnet34', 'vgg', 'vit', 'pvt')
        # Value: corresponding extractor module
        self.encoders = nn.ModuleDict()
        
        # Add ResNet encoder if requested
        if 'resnet34' in self.architectures:
            self.encoders['resnet34'] = ResNetExtractor(config['resnet_layer'])
        
        # Add VGG encoder if requested
        if 'vgg16' in self.architectures:
            self.encoders['vgg16'] = VGGExtractor(config['vgg_block'])
        
        # Add ViT encoder if requested (check for any ViT variant)
        if any(a.startswith('vit') for a in self.architectures):
            # Find the specific ViT variant name
            vit_arch = [a for a in self.architectures if a.startswith('vit')][0]
            self.encoders['vit'] = ViTExtractor(vit_arch, config['vit_block'])
        
        # Add PVT encoder if requested (check for any PVT variant)
        if any(a.startswith('pvt') for a in self.architectures):
            # Find the specific PVT variant name
            pvt_arch = [a for a in self.architectures if a.startswith('pvt')][0]
            self.encoders['pvt'] = PVTExtractor(pvt_arch, config['pvt_stage'])
        
        # ====================================================================
        # DETERMINE FEATURE DIMENSIONS
        # ====================================================================
        # Run dummy forward pass to determine feature dimensions for each encoder
        # Needed for fusion module to know input dimensions
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            
            # Extract features from all encoders
            features_dict = {
                name: enc(dummy) 
                for name, enc in self.encoders.items()
            }
            
            # Get channel counts for each encoder
            # e.g., {'resnet34': 64, 'vgg16': 64, 'vit': 384, 'pvt': 64}
            output_channels = {
                name: f.shape[1] 
                for name, f in features_dict.items()
            }
            
            # Get spatial dimensions for each encoder
            # e.g., [(56, 56), (112, 112), (14, 14), (56, 56)]
            spatial_sizes = [f.shape[-2:] for f in features_dict.values()]
            
            # Determine target spatial size (use maximum to preserve detail)
            # In our case: max(56, 112, 14, 56) = 112 from VGG block1
            target_spatial_size = max(s[0] for s in spatial_sizes)
        
        # Print feature information
        print(f"Architectures: {list(self.encoders.keys())}")
        print(f"Output channels: {output_channels}")
        print(f"Target spatial size: {target_spatial_size}")
        
        # ====================================================================
        # CREATE FUSION MODULE (trainable)
        # ====================================================================
        # Aligns features from different architectures to common spatial size
        # and channel count, then fuses them using specified strategy
        self.fusion = FeatureFusionModule(
            output_channels,           # Dict of {arch: channels}
            config['fusion_strategy'], # 'attention', 'concat', or 'weighted'
            config['fusion_channels'], # Target channels (typically 256)
            target_spatial_size        # Target spatial size (max of inputs)
        )
        
        # ====================================================================
        # CREATE DECODER (trainable)
        # ====================================================================
        # Same decoder options as single model
        decoder_map = {
            'frequency_aware': FrequencyAwareDecoder,
            'wavelet': WaveletFrequencyDecoder,
            'transposed_conv': TransposedConvDecoder,
            'attention': AttentionDecoder
        }
        
        decoder_class = decoder_map[config['decoder_type']]
        self.decoder = decoder_class(
            config['fusion_channels'],  # Input channels = fusion output
            target_spatial_size,        # Input size = fusion spatial size
            config['output_size']       # Target = 224×224
        )
        
        # Print fusion and decoder information
        print(f"Fusion: {config['fusion_strategy']}")
        print(f"Decoder: {config['decoder_type']}")
        print(f"{'#'*70}\n")
    
    def forward(self, x):
        """
        Forward pass: Extract features from all encoders, fuse, and reconstruct.
        
        Pipeline:
        1. Pass input through all frozen encoders in parallel
        2. Collect features in dictionary: {arch_name: features}
        3. Fuse features using specified strategy
        4. Reconstruct image using trainable decoder
        
        Args:
            x (torch.Tensor): Input images, shape (B, 3, 224, 224)
        
        Returns:
            torch.Tensor: Reconstructed images, shape (B, 3, 224, 224)
                         Values in range [0, 1]
        
        Note:
            All encoders run in parallel (could be parallelized for speed).
            Only fusion and decoder receive gradients during training.
        """
        # Step 1: Extract features from all encoders
        # Creates dict: {'resnet34': tensor, 'vgg16': tensor, ...}
        features = {
            name: encoder(x) 
            for name, encoder in self.encoders.items()
        }
        
        # Step 2: Fuse features from multiple sources
        # Output: (B, fusion_channels, target_spatial, target_spatial)
        fused = self.fusion(features)
        
        # Step 3: Reconstruct image from fused features
        # Output: (B, 3, 224, 224)
        reconstructed = self.decoder(fused)
        
        return reconstructed


class FeatureFusionModule(nn.Module):
    """
    Fuses features from multiple architectures with spatial/channel alignment.
    
    This module handles the complex task of combining features from different
    architectures that may have:
    - Different channel counts (64, 128, 384, etc.)
    - Different spatial resolutions (56×56, 112×112, 14×14, etc.)
    - Different feature representations (CNN vs Transformer)
    
    The module performs three key operations:
    1. Channel Alignment: Project all features to same channel count
    2. Spatial Alignment: Interpolate all features to same spatial size
    3. Feature Fusion: Combine aligned features using specified strategy
    
    Fusion Strategies Explained:
    
    1. Attention Fusion:
       - Most sophisticated approach
       - Learns spatial attention maps for each architecture
       - Can adaptively weight different networks per location
       - Process: Global pool → FC → Softmax → Weighted sum
       - Best when: Different networks excel at different regions
    
    2. Concatenation Fusion:
       - Simplest approach
       - Stack all features along channel dimension
       - Let convolutions learn optimal combination
       - Process: Concat → Conv layers to reduce channels
       - Best when: Want network to learn fusion automatically
    
    3. Weighted Fusion:
       - Middle ground approach
       - Single learnable weight per architecture
       - Global weighting (not spatial)
       - Process: Learnable scalars → Softmax → Weighted sum
       - Best when: Some networks consistently better than others
    
    Args:
        feature_channels (dict): Channel counts per architecture
                                Example: {'resnet34': 64, 'vgg16': 64, 'vit': 384}
        
        fusion_strategy (str): How to combine features
                              Options: 'attention', 'concat', 'weighted'
        
        target_channels (int): Output channel count after fusion (typically 256)
        
        target_spatial_size (int): Target spatial dimension (square)
                                  Usually max of input spatial sizes
    
    Input:
        features_dict (dict): Dictionary of features from multiple architectures
                            Keys: architecture names (e.g., 'resnet34', 'vgg16')
                            Values: feature tensors, shape (B, C_i, H_i, W_i)
    
    Output:
        fused (torch.Tensor): Fused features, shape (B, target_channels, 
                             target_spatial_size, target_spatial_size)
    
    Example:
        >>> feature_channels = {'resnet34': 64, 'vgg16': 64, 'vit': 384}
        >>> fusion = FeatureFusionModule(feature_channels, 'attention', 256, 112)
        >>> features = {
        ...     'resnet34': torch.randn(4, 64, 56, 56),
        ...     'vgg16': torch.randn(4, 64, 112, 112),
        ...     'vit': torch.randn(4, 384, 14, 14)
        ... }
        >>> fused = fusion(features)  # (4, 256, 112, 112)
    """
    
    def __init__(self, feature_channels, fusion_strategy='attention', 
                 target_channels=256, target_spatial_size=28):
        super().__init__()
        
        # Store configuration
        self.feature_channels = feature_channels
        self.fusion_strategy = fusion_strategy
        self.target_channels = target_channels
        self.target_size = target_spatial_size
        
        # ====================================================================
        # CHANNEL ALIGNMENT
        # ====================================================================
        # Create 1×1 convolutions to project each architecture's features
        # to the same channel count (target_channels)
        # This standardizes feature dimensionality across architectures
        self.channel_aligners = nn.ModuleDict({
            name: nn.Sequential(
                nn.Conv2d(channels, target_channels, kernel_size=1),  # 1×1 conv projection
                nn.BatchNorm2d(target_channels),  # Normalize projected features
                nn.ReLU(inplace=True)             # Non-linearity
            )
            for name, channels in feature_channels.items()
        })
        
        # ====================================================================
        # FUSION STRATEGY SETUP
        # ====================================================================
        # Build fusion-specific modules based on selected strategy
        if fusion_strategy == 'attention':
            self._build_attention()
        elif fusion_strategy == 'concat':
            self._build_concat()
        elif fusion_strategy == 'weighted':
            self._build_weighted()
    
    def _build_attention(self):
        """
        Build attention-based fusion mechanism.
        
        Creates a network that:
        1. Takes concatenated features from all architectures
        2. Applies global average pooling to get spatial summary
        3. Learns attention weights via FC layers
        4. Applies softmax for normalized weights
        5. Weights sum to 1 across architectures
        
        The attention weights are spatial (same for all locations) but
        can differ per architecture, allowing the network to learn which
        architecture to trust more.
        """
        num_sources = len(self.feature_channels)
        
        # Attention network: Concat features → Global pool → FC → Softmax
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, C*N, H, W) → (B, C*N, 1, 1)
            nn.Conv2d(self.target_channels * num_sources, self.target_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.target_channels, num_sources, 1),
            nn.Softmax(dim=1)  # Normalize weights across architectures
        )
        
        # Refinement: Post-fusion convolutions to integrate weighted features
        self.refine = nn.Sequential(
            nn.Conv2d(self.target_channels, self.target_channels, 3, padding=1),
            nn.BatchNorm2d(self.target_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.target_channels, self.target_channels, 3, padding=1),
            nn.BatchNorm2d(self.target_channels),
            nn.ReLU(inplace=True)
        )
    
    def _build_concat(self):
        """
        Build concatenation-based fusion mechanism.
        
        Simply stacks all features along channel dimension, then uses
        convolutional layers to reduce back to target channel count.
        
        Let network learn optimal combination through training.
        """
        num_sources = len(self.feature_channels)
        
        # Fusion network: Concat features → Conv layers → Reduce channels
        self.fusion = nn.Sequential(
            # Input: target_channels × num_sources (all features stacked)
            nn.Conv2d(self.target_channels * num_sources, 
                     self.target_channels * 2, 3, padding=1),
            nn.BatchNorm2d(self.target_channels * 2),
            nn.ReLU(inplace=True),
            # Output: target_channels (reduced to final size)
            nn.Conv2d(self.target_channels * 2, 
                     self.target_channels, 3, padding=1),
            nn.BatchNorm2d(self.target_channels),
            nn.ReLU(inplace=True)
        )
    
    def _build_weighted(self):
        """
        Build weighted fusion mechanism.
        
        Creates learnable scalar weights (one per architecture) that are
        normalized via softmax. Each architecture's features are multiplied
        by its weight and summed.
        
        Simpler than attention (no spatial variation) but more flexible
        than equal weighting.
        """
        num_sources = len(self.feature_channels)
        
        # Learnable weights: One scalar per architecture
        # Shape: (num_sources, 1, 1, 1) for broadcasting
        # Initialized to ones (equal weighting initially)
        self.weights = nn.Parameter(torch.ones(num_sources, 1, 1, 1))
        
        # Refinement: Post-fusion convolutions
        self.refine = nn.Sequential(
            nn.Conv2d(self.target_channels, self.target_channels, 3, padding=1),
            nn.BatchNorm2d(self.target_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features_dict):
        """
        Fuse features from multiple architectures.
        
        Pipeline:
        1. Sort feature dictionary by architecture name (for consistency)
        2. Align channels: Project each to target_channels via 1×1 convs
        3. Align spatial: Interpolate each to target_spatial_size
        4. Fuse: Combine aligned features using selected strategy
        
        Args:
            features_dict (dict): Features from multiple architectures
                                 Keys: architecture names
                                 Values: feature tensors (B, C_i, H_i, W_i)
        
        Returns:
            torch.Tensor: Fused features, shape (B, target_channels, 
                         target_spatial_size, target_spatial_size)
        """
        aligned_features = []
        
        # Step 1 & 2: Channel alignment for each architecture
        # Sort keys for deterministic order (important for weighted fusion)
        for name in sorted(features_dict.keys()):
            # Project to target channels via 1×1 conv
            feat = self.channel_aligners[name](features_dict[name])
            
            # Step 3: Spatial alignment via bilinear interpolation
            # Only interpolate if spatial size doesn't match target
            if feat.shape[-2:] != (self.target_size, self.target_size):
                feat = F.interpolate(
                    feat,
                    size=(self.target_size, self.target_size),
                    mode='bilinear',
                    align_corners=False  # Recommended for upsampling
                )
            
            aligned_features.append(feat)
        
        # Step 4: Fuse aligned features based on strategy
        if self.fusion_strategy == 'attention':
            # Attention fusion: Learn spatial attention weights
            stacked = torch.cat(aligned_features, dim=1)  # (B, C*N, H, W)
            attn_weights = self.attention(stacked)        # (B, N, 1, 1)
            weights_list = torch.split(attn_weights, 1, dim=1)  # List of (B, 1, 1, 1)
            
            # Apply attention weights and sum
            fused = sum(w * f for w, f in zip(weights_list, aligned_features))
            return self.refine(fused)
            
        elif self.fusion_strategy == 'concat':
            # Concat fusion: Stack and let conv layers combine
            concatenated = torch.cat(aligned_features, dim=1)  # (B, C*N, H, W)
            return self.fusion(concatenated)
            
        elif self.fusion_strategy == 'weighted':
            # Weighted fusion: Learnable scalar weights
            weights = F.softmax(self.weights, dim=0)  # Normalize weights to sum to 1
            
            # Apply weights and sum
            fused = sum(w * f for w, f in zip(weights, aligned_features))
            return self.refine(fused)


# ============================================================================
# HELPER MODULES
# ============================================================================
# Supporting modules used by the complex decoders (FrequencyAware, Wavelet).
# These implement attention mechanisms, upsampling, and refinement operations.

class FrequencyFusionBlock(nn.Module):
    """
    Fuses low and high frequency features using channel and spatial attention.
    
    This block is used in FrequencyAwareDecoder to combine the separately
    processed low and high frequency pathways. It uses two attention mechanisms:
    
    1. Channel Attention: Which frequency bands/channels are most important?
       - Global average pooling → FC layers → Sigmoid
       - Outputs weights per channel
    
    2. Spatial Attention: Which spatial locations are most important?
       - Average and max pooling → Conv → Sigmoid
       - Outputs spatial attention map
    
    The combined features are modulated by both attention mechanisms,
    allowing the network to focus on important frequency content at
    important spatial locations.
    
    Args:
        channels (int): Number of channels in each frequency pathway (low/high)
    
    Input:
        low_freq (torch.Tensor): Low frequency features, shape (B, C, H, W)
        high_freq (torch.Tensor): High frequency features, shape (B, C, H, W)
    
    Output:
        fused (torch.Tensor): Fused features, shape (B, C*2, H, W)
    """
    
    def __init__(self, channels):
        super().__init__()
        
        # Channel attention: Learn importance of each frequency band
        # Process: Global pool → Bottleneck FC → Expand FC → Sigmoid
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, C*2, H, W) → (B, C*2, 1, 1)
            nn.Conv2d(channels*2, channels, 1),  # Bottleneck
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels*2, 1),  # Expand back
            nn.Sigmoid()  # Output channel weights in [0, 1]
        )
        
        # Spatial attention: Learn importance of each spatial location
        # Uses both average and max pooled features (captures different info)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),  # 2 channels (avg + max) → 1
            nn.Sigmoid()  # Output spatial weights in [0, 1]
        )
        
        # Final fusion: 1×1 conv to integrate attended features
        self.fusion = nn.Conv2d(channels*2, channels*2, 1)
    
    def forward(self, low_freq, high_freq):
        """Apply channel and spatial attention to fuse frequency components."""
        # Concatenate frequency pathways
        x = torch.cat([low_freq, high_freq], dim=1)  # (B, C*2, H, W)
        
        # Apply channel attention
        x = x * self.channel_attn(x)
        
        # Create spatial attention map from average and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Average across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max across channels
        spatial_attn_map = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        
        # Apply spatial attention
        x = x * spatial_attn_map
        
        # Final fusion convolution
        return self.fusion(x)


class FrequencyBandFusion(nn.Module):
    """
    Fuses wavelet frequency bands using channel attention.
    
    This block is used in WaveletFrequencyDecoder to combine the four
    wavelet subbands (LL, LH, HL, HH). It applies channel attention to
    learn the relative importance of each subband.
    
    Args:
        band_channels (int): Number of channels per subband (typically 64)
    
    Input:
        x (torch.Tensor): Concatenated wavelet bands, shape (B, band_channels*4, H, W)
    
    Output:
        refined (torch.Tensor): Refined features, shape (B, band_channels*4, H, W)
    """
    
    def __init__(self, band_channels):
        super().__init__()
        total = band_channels * 4  # Total channels = 4 subbands × channels per band
        
        # Band attention: Learn importance of each wavelet subband
        self.band_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global context
            nn.Conv2d(total, total//4, 1),  # Bottleneck
            nn.ReLU(inplace=True),
            nn.Conv2d(total//4, total, 1),  # Expand
            nn.Sigmoid()  # Output weights in [0, 1]
        )
        
        # Refinement: Post-attention processing
        self.refine = nn.Sequential(
            nn.Conv2d(total, total, 3, padding=1),
            nn.BatchNorm2d(total),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """Apply channel attention and refinement to wavelet bands."""
        return self.refine(x * self.band_attention(x))


class SpatialChannelAttentionBlock(nn.Module):
    """
    Applies both channel and spatial attention to input features.
    
    This is a standard attention block used in FrequencyAwareDecoder.
    It combines two complementary attention mechanisms:
    
    1. Channel Attention: Which feature channels are important?
       - Uses global average pooling and FC layers
       - Based on Squeeze-and-Excitation (SE) blocks
    
    2. Spatial Attention: Which spatial locations are important?
       - Uses 1×1 convolution to compress channels
       - Outputs spatial importance map
    
    Args:
        channels (int): Number of input channels
        reduction (int): Channel reduction ratio for bottleneck (default: 16)
    
    Input:
        x (torch.Tensor): Input features, shape (B, C, H, W)
    
    Output:
        attended (torch.Tensor): Attention-modulated features, shape (B, C, H, W)
    """
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        # Channel attention (SE block style)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global pooling
            nn.Conv2d(channels, channels//reduction, 1),  # Reduce
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, channels, 1),  # Expand
            nn.Sigmoid()
        )
        
        # Spatial attention (simple 1×1 conv)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, 1, 1),  # Compress to single channel
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Apply channel attention then spatial attention."""
        x = x * self.channel_attn(x)  # Modulate channels
        return x * self.spatial_attn(x)  # Modulate spatial locations


class UpsampleResBlock(nn.Module):
    """
    Upsampling block with residual connection.
    
    This block performs 2× spatial upsampling while preserving information
    through a residual connection. It's used in FrequencyAware and Wavelet
    decoders for progressive upsampling.
    
    Architecture:
        Main path: TransposedConv (2× up) → Conv → BatchNorm → Conv → BatchNorm
        Residual: Bilinear upsample (2× up) → 1×1 Conv (channel alignment)
        Output: Main + Residual → ReLU
    
    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels (typically in_channels // 2)
    
    Input:
        x (torch.Tensor): Input features, shape (B, in_channels, H, W)
    
    Output:
        upsampled (torch.Tensor): Upsampled features, shape (B, out_channels, H*2, W*2)
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Main path: Transposed conv upsampling + refinement
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=4, stride=2, padding=1  # 2× spatial upsampling
        )
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Residual path: Bilinear upsample + channel projection
        self.residual = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 1)  # Align channels
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """Forward with residual connection."""
        main = self.conv_block(self.upsample(x))
        residual = self.residual(x)
        return self.relu(main + residual)


class MultiScaleRefinement(nn.Module):
    """
    Multi-scale feature refinement using parallel convolutions.
    
    This module processes features at multiple receptive field scales
    simultaneously, capturing both fine details and broader context.
    It's inspired by Inception modules and ASPP (Atrous Spatial Pyramid Pooling).
    
    Uses 3 parallel convolutional paths:
    - 3×3 conv: Local details (small receptive field)
    - 5×5 conv: Medium-scale patterns
    - 7×7 conv: Larger context (large receptive field)
    
    The multi-scale features are concatenated, then combined using learned
    attention weights. This allows the network to adaptively select the
    appropriate scale for each spatial location.
    
    Args:
        channels (int): Number of input/output channels
    
    Input:
        x (torch.Tensor): Input features, shape (B, C, H, W)
    
    Output:
        refined (torch.Tensor): Refined features, shape (B, C, H, W)
    """
    
    def __init__(self, channels):
        super().__init__()
        
        # Three parallel scales (different receptive fields)
        self.scale1 = nn.Conv2d(channels, channels//4, kernel_size=3, padding=1)  # Small
        self.scale2 = nn.Conv2d(channels, channels//4, kernel_size=5, padding=2)  # Medium
        self.scale3 = nn.Conv2d(channels, channels//4, kernel_size=7, padding=3)  # Large
        
        # Attention: Learn importance of each scale
        self.attention = nn.Sequential(
            nn.Conv2d(channels//4*3, channels, 1),  # Compress multi-scale features
            nn.Sigmoid()  # Attention weights
        )
        
        # Fusion: Combine multi-scale features
        self.fusion = nn.Conv2d(channels//4*3, channels, 1)
    
    def forward(self, x):
        """Process at multiple scales and fuse with attention."""
        # Extract multi-scale features
        multi_scale = torch.cat([
            self.scale1(x),  # Fine details
            self.scale2(x),  # Medium patterns
            self.scale3(x)   # Broader context
        ], dim=1)
        
        # Apply attention and add to input (residual connection)
        attention_weights = self.attention(multi_scale)
        refined = self.fusion(multi_scale) * attention_weights
        
        return x + refined  # Residual connection


class ColorEnhancementBlock(nn.Module):
    """
    Enhances color distribution in reconstructed images.
    
    This block applies adaptive color transformations to improve the
    naturalness and vibrancy of reconstructed images. It uses instance
    normalization for style transfer-like color correction.
    
    Architecture:
    - Color transform: Learn adaptive color adjustment via 1×1 convs
    - Color refine: Spatial processing with instance normalization
    - Multiplicative modulation: x * (1 + transform(x))
    - Residual connection: x + refine(x)
    
    Args:
        channels (int): Number of input/output channels
    
    Input:
        x (torch.Tensor): Input features, shape (B, C, H, W)
    
    Output:
        enhanced (torch.Tensor): Color-enhanced features, shape (B, C, H, W)
    """
    
    def __init__(self, channels):
        super().__init__()
        
        # Color transformation: Adaptive color adjustment
        self.color_transform = nn.Sequential(
            nn.Conv2d(channels, 64, 1),  # Bottleneck
            nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, 1),  # Expand
            nn.Sigmoid()  # Output in [0, 1] for scaling
        )
        
        # Color refinement: Spatial processing with instance norm
        # Instance norm helps with color/style normalization
        self.color_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),  # Normalize per-image statistics
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
    
    def forward(self, x):
        """Apply color transformation and refinement."""
        # Multiplicative color modulation: x * (1 + scale)
        # This adjusts feature magnitude based on learned transform
        x = x * (1 + self.color_transform(x))
        
        # Additive color refinement with residual
        return x + self.color_refine(x)


class TransformerBlock(nn.Module):
    """
    Standard Transformer block with self-attention and feed-forward layers.
    
    This is the basic building block of Transformer architectures, used in
    AttentionDecoder. It consists of:
    
    1. Multi-head Self-Attention: Global context aggregation
    2. Layer Normalization: Stabilize training
    3. Feed-Forward Network: Channel-wise processing (MLP)
    4. Residual Connections: Enable deep stacking
    
    Architecture follows the standard "Attention is All You Need" design.
    
    Args:
        dim (int): Feature dimension (number of channels)
        num_heads (int): Number of attention heads (default: 8)
        mlp_ratio (float): MLP hidden dimension ratio (default: 4.0)
    
    Input:
        x (torch.Tensor): Input sequence, shape (B, N, D)
                         where N = sequence length (H×W for images)
                               D = feature dimension
    
    Output:
        processed (torch.Tensor): Processed sequence, shape (B, N, D)
    """
    
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        
        # Layer normalization before attention
        self.norm1 = nn.LayerNorm(dim)
        
        # Multi-head self-attention
        self.attn = SelfAttention(dim, num_heads)
        
        # Layer normalization before MLP
        self.norm2 = nn.LayerNorm(dim)
        
        # Feed-forward MLP (typically 4× hidden dimension)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),  # Smooth activation (better than ReLU for Transformers)
            nn.Linear(mlp_dim, dim)
        )
    
    def forward(self, x):
        """Forward pass with residual connections."""
        # Self-attention with residual
        x = x + self.attn(self.norm1(x))
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class SelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    This implements the core attention operation from "Attention is All You Need".
    Each token attends to all other tokens via scaled dot-product attention.
    
    Attention Formula:
        Attention(Q, K, V) = softmax(QK^T / √d) V
    
    Where:
    - Q (Query): What am I looking for?
    - K (Key): What do I contain?
    - V (Value): What information do I have?
    - d (head_dim): Dimension per head (for scaling)
    
    Multi-head attention runs this operation in parallel H times with
    different learned projections, then concatenates the results.
    
    Args:
        dim (int): Input feature dimension
        num_heads (int): Number of parallel attention heads (default: 8)
    
    Input:
        x (torch.Tensor): Input sequence, shape (B, N, D)
    
    Output:
        attended (torch.Tensor): Attention-processed sequence, shape (B, N, D)
    """
    
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # Dimension per head
        self.scale = self.head_dim ** -0.5  # Scaling factor (1/√d)
        
        # Linear projection for Q, K, V (3× dimensions for all three)
        self.qkv = nn.Linear(dim, dim * 3)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        """
        Compute multi-head self-attention.
        
        Steps:
        1. Project to Q, K, V
        2. Reshape to separate heads
        3. Compute scaled dot-product attention per head
        4. Concatenate heads
        5. Project to output
        """
        B, N, D = x.shape
        
        # Step 1: Project to Q, K, V and reshape for multi-head
        # (B, N, D) → (B, N, 3*D) → (B, N, 3, H, D//H) → (3, B, H, N, D//H)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # Separate Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, H, N, D//H)
        
        # Step 2: Compute scaled dot-product attention
        # QK^T: (B, H, N, D//H) @ (B, H, D//H, N) → (B, H, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # Scale by 1/√d
        attn = attn.softmax(dim=-1)  # Normalize attention weights
        
        # Step 3: Apply attention to values
        # (B, H, N, N) @ (B, H, N, D//H) → (B, H, N, D//H)
        attended = attn @ v
        
        # Step 4: Concatenate heads
        # (B, H, N, D//H) → (B, N, H, D//H) → (B, N, D)
        attended = attended.transpose(1, 2).reshape(B, N, D)
        
        # Step 5: Output projection
        return self.proj(attended)
