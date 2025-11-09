"""
Feature extractors and decoders for image reconstruction.

Implements:
    - FeatureExtractor: Extracts features from ResNet34, VGG16, or ViT
    - SimpleDecoder: Basic transposed convolution decoder
    - AttentionDecoder: Decoder with self-attention mechanisms (Chapter 26)
"""

import torch
import torch.nn as nn
from torchvision import models
import timm


class FeatureExtractor(nn.Module):
    """
    Extract features from pre-trained encoders at specified layers.
    
    The encoder is frozen (no training) and acts as f_θ(x) from Chapter 33:
        z = f_θ(x)  where z are the extracted features
    
    Args:
        architecture: 'resnet34', 'vgg16', or 'vit_base_patch16_224'
        layer_name: Which layer to extract from (e.g., 'layer3' for ResNet)
    """
    
    def __init__(self, architecture='resnet34', layer_name = 'layer3'):
        super().__init__()
        self.architecture = architecture
        self.layer_name = layer_name
        
        # Build the appropriate architecture
        if architecture == 'resnet34':
            self._build_resnet()
        elif architecture == 'vgg16':
            self._build_vgg()
        elif architecture.startswith('vit'):
            self._build_vit()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Freeze all parameters (encoder is not trained)
        for param in self.parameters():
            param.requires_grad = False
        
        # Set to eval mode
        self.eval()
    
    def _build_resnet(self):
        """
        Build ResNet34 feature extractor.
        
        ResNet structure:
            conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4
        
        We extract features up to the specified layer.
        """
        # Load pre-trained ResNet34
        resnet = models.resnet34(weights='IMAGENET1K_V1')
        
        # Start with initial layers (always included)
        layers = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool]
        
        # Add layers progressively based on layer_name
        # Example: if layer_name='layer3', we include layer1, layer2, and layer3
        if self.layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layers.append(resnet.layer1)  # Output: 64 channels
        if self.layer_name in ['layer2', 'layer3', 'layer4']:
            layers.append(resnet.layer2)  # Output: 128 channels
        if self.layer_name in ['layer3', 'layer4']:
            layers.append(resnet.layer3)  # Output: 256 channels
        if self.layer_name == 'layer4':
            layers.append(resnet.layer4)  # Output: 512 channels
        
        # Combine all layers into a sequential module
        self.features = nn.Sequential(*layers)
    
    def _build_vgg(self):
        """
        Build VGG16 feature extractor.
        
        VGG has 5 blocks, each ending with max pooling.
        We map friendly names to feature indices.
        """
        # Load pre-trained VGG16
        vgg = models.vgg16(weights = 'IMAGENET1K_V1')
        
        # Map block names to feature layer indices
        # VGG features are organized as a sequence, we extract up to certain indices
        layer_map = {
            'block1': 4,   # After 1st pooling: 64 channels
            'block2': 9,   # After 2nd pooling: 128 channels
            'block3': 16,  # After 3rd pooling: 256 channels
            'block4': 23,  # After 4th pooling: 512 channels
            'block5': 30   # After 5th pooling: 512 channels
        }
        
        # Get the index for the requested layer (default to block4)
        idx = layer_map.get(self.layer_name, 23)
        
        # Extract features up to (and including) that index
        self.features = vgg.features[:idx + 1]
    
    def _build_vit(self):
        """
        Build Vision Transformer (ViT) feature extractor.
        
        ViT processes images as sequences of patches through transformer blocks.
        We extract features after a specified transformer block.
        """
        # Load pre-trained ViT
        self.vit = timm.create_model(self.architecture, pretrained = True)
        
        # Parse block number from layer_name (e.g., 'block6' -> block index 6)
        if self.layer_name.startswith('block'):
            self.block_idx = int(self.layer_name.replace('block', ''))
        else:
            self.block_idx = 6  # Default to middle block
        
        # ViT needs special handling in forward pass (no sequential features)
        self.features = None
    
    def forward(self, x):
        """
        Extract features from input images.
        
        Args:
            x: Input images [batch_size, 3, height, width]
            
        Returns:
            features: Feature maps [batch_size, channels, h, w]
                      For ResNet/VGG: spatial feature maps
                      For ViT: reshaped token embeddings
        """
        if self.architecture.startswith('vit'):
            return self._forward_vit(x)
        else:
            # For CNN architectures, just pass through sequential layers
            return self.features(x)
    
    def _forward_vit(self, x):
        """
        Forward pass for Vision Transformer.
        
        ViT processes images as:
        1. Split into patches
        2. Linear embedding
        3. Transformer blocks
        4. We extract features after block N and reshape to spatial format
        """
        # Step 1: Convert image to patch embeddings [B, num_patches, embed_dim]
        x = self.vit.patch_embed(x)
        
        # Step 2: Add positional embeddings
        x = self.vit._pos_embed(x)
        
        # Step 3: Pass through transformer blocks up to specified block
        for i, block in enumerate(self.vit.blocks):
            x = block(x)
            if i == self.block_idx:
                break  # Stop at the requested block
        
        # Step 4: Reshape token sequence to spatial format
        B, N, D = x.shape  # Batch, Num_tokens, Dimension
        H = W = int((N - 1) ** 0.5)  # Calculate spatial dimensions (N-1 excludes class token)
        
        # Remove class token (first token) and reshape to [B, D, H, W]
        x = x[:, 1:, :]  # Remove class token
        x = x.transpose(1, 2).reshape(B, D, H, W)  # Reshape to spatial
        
        return x
    
    def get_output_shape(self, input_size=224):
        """
        Helper function to determine output feature shape.
        
        Useful for automatically configuring the decoder.
        
        Args:
            input_size: Input image size (default 224)
            
        Returns:
            (channels, height, width) of output features
        """
        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size, input_size)
        
        # Forward pass (no gradients needed)
        with torch.no_grad():
            output = self.forward(dummy_input)
        
        # Return shape (excluding batch dimension)
        return output.shape[1:]  # (C, H, W)


class SimpleDecoder(nn.Module):
    """
    Simple decoder using transposed convolutions.
    
    Implements g_ψ(z) from Chapter 33:
        x̂ = g_ψ(z)  where x̂ is the reconstructed image
    
    Architecture:
        Features -> [TransposeConv + BatchNorm + ReLU] x N -> Conv -> Sigmoid -> RGB
    
    Args:
        input_channels: Number of channels in input features
        input_size: Spatial size (height/width) of input features
        output_size: Desired output image size (default 224)
    """
    
    def __init__(self, input_channels, input_size, output_size = 224):
        super().__init__()
        
        # Calculate how many upsampling stages we need
        # Each transpose conv with stride=2 doubles the spatial size
        num_upsample = 0
        current_size = input_size
        while current_size < output_size:
            current_size *= 2
            num_upsample += 1
        
        # Build the decoder progressively
        layers = []
        in_ch = input_channels
        
        # Add upsampling blocks
        for i in range(num_upsample):
            out_ch = max(in_ch // 2, 32)  # Halve channels each stage, minimum 32
            
            # Upsampling block: TransposeConv -> BatchNorm -> ReLU
            layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ])
            
            in_ch = out_ch  # Update for next iteration
        
        # Final layer: Convert to RGB with Sigmoid activation
        # Sigmoid ensures output is in [0, 1] range
        layers.extend([
            nn.Conv2d(in_ch, 3, kernel_size=3, padding=1),  # -> 3 channels (RGB)
            nn.Sigmoid()  # Output range [0, 1]
        ])
        
        # Combine all layers
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Reconstruct image from features.
        
        Args:
            x: Feature maps [batch_size, channels, h, w]
            
        Returns:
            reconstructed: RGB images [batch_size, 3, H, W] in range [0, 1]
        """
        return self.decoder(x)


class AttentionDecoder(nn.Module):
    """
    Decoder with self-attention mechanisms.
    
    Based on:
        - Chapter 26: Self-Attention mechanism (Eq 26.6)
        - arxiv:2506.07803v1: Transformer blocks improve reconstruction
    
    Architecture:
        Features -> [TransformerBlock x 4] -> [Upsample + ResBlock x N] -> RGB
    
    The attention mechanism allows the decoder to focus on relevant spatial
    locations when reconstructing, improving quality over simple upsampling.
    
    Args:
        input_channels: Number of channels in input features
        input_size: Spatial size of input features
        output_size: Desired output image size
        num_blocks: Number of transformer blocks (default 4)
    """
    
    def __init__(self, input_channels, input_size, output_size = 224, num_blocks = 4):
        super().__init__()
        
        self.input_channels = input_channels
        
        # Create transformer blocks for self-attention
        # These process features as tokens before upsampling
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(input_channels) for _ in range(num_blocks)
        ])
        
        # Calculate upsampling stages (same as SimpleDecoder)
        num_upsample = 0
        current_size = input_size
        while current_size < output_size:
            current_size *= 2
            num_upsample += 1
        
        # Build upsampling path
        layers = []
        in_ch = input_channels
        
        for i in range(num_upsample):
            out_ch = max(in_ch // 2, 32)
            layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ])
            in_ch = out_ch
        
        # Final RGB conversion
        layers.extend([
            nn.Conv2d(in_ch, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        ])
        
        self.upsampler = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Reconstruct with attention mechanism.
        
        Process:
        1. Reshape features to token sequence
        2. Apply self-attention (allows global information mixing)
        3. Reshape back to spatial format
        4. Upsample to output size
        
        Args:
            x: Feature maps [B, C, H, W]
            
        Returns:
            reconstructed: RGB images [B, 3, H', W']
        """
        B, C, H, W = x.shape
        
        # Step 1: Reshape spatial features to token sequence
        # [B, C, H, W] -> [B, H*W, C]
        # Each spatial location becomes a token
        x = x.flatten(2).transpose(1, 2)  # [B, N, C] where N = H*W
        
        # Step 2: Apply transformer blocks (self-attention + FFN)
        # Each block allows tokens to attend to each other
        for block in self.transformer_blocks:
            x = block(x)
        
        # Step 3: Reshape back to spatial format
        # [B, N, C] -> [B, C, H, W]
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        # Step 4: Upsample to output resolution
        return self.upsampler(x)


class TransformerBlock(nn.Module):
    """
    Single transformer block with self-attention and feed-forward network.
    
    From Chapter 26: Standard transformer architecture
        1. LayerNorm -> Self-Attention -> Residual connection
        2. LayerNorm -> FFN -> Residual connection
    
    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of FFN hidden dimension to input dimension
    """
    
    def __init__(self, dim, num_heads = 8, mlp_ratio = 4.0):
        super().__init__()
        
        # First sub-block: Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        
        # Second sub-block: Feed-forward network
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)  # Expand dimension in FFN
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),  # Smooth activation function
            nn.Linear(mlp_dim, dim)
        )
    
    def forward(self, x):
        """
        Process tokens through attention and FFN with residual connections.
        
        Args:
            x: Tokens [B, N, D]
            
        Returns:
            x: Transformed tokens [B, N, D]
        """
        # Self-attention with residual: x = x + Attention(LayerNorm(x))
        x = x + self.attn(self.norm1(x))
        
        # Feed-forward with residual: x = x + FFN(LayerNorm(x))
        x = x + self.mlp(self.norm2(x))
        
        return x


class SelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    From Chapter 26.6.2, Equation 26.6:
        Q = T·W_q^T,  K = T·W_k^T,  V = T·W_v^T
        A = softmax(QK^T / sqrt(m))
        output = A·V
    
    This allows each token to attend to all other tokens, enabling
    global information mixing across spatial locations.
    
    Args:
        dim: Token dimension
        num_heads: Number of parallel attention heads
    """
    
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # Dimension per head
        self.scale = self.head_dim ** -0.5  # Scaling factor: 1/sqrt(m)
        
        # Linear projections for Q, K, V (all in one for efficiency)
        self.qkv = nn.Linear(dim, dim * 3)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        """
        Compute multi-head self-attention.
        
        Steps:
        1. Project input to Q, K, V
        2. Split into multiple heads
        3. Compute attention: A = softmax(QK^T / sqrt(d))
        4. Apply attention to values: output = A·V
        5. Concatenate heads and project
        
        Args:
            x: Input tokens [B, N, D]
            
        Returns:
            output: Attended tokens [B, N, D]
        """
        B, N, D = x.shape
        
        # Step 1: Compute Q, K, V from input (Equation 26.5)
        # [B, N, D] -> [B, N, 3*D] -> [B, N, 3, num_heads, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Split into Q, K, V
        
        # Step 2: Compute attention scores (Equation 26.6)
        # QK^T: [B, num_heads, N, head_dim] @ [B, num_heads, head_dim, N]
        #     = [B, num_heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # Scaled dot-product
        attn = attn.softmax(dim=-1)  # Normalize to get attention weights
        
        # Step 3: Apply attention to values
        # [B, num_heads, N, N] @ [B, num_heads, N, head_dim]
        # = [B, num_heads, N, head_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        
        # Step 4: Output projection
        x = self.proj(x)
        
        return x


def create_decoder(decoder_type = 'simple', input_channels = 256, input_size = 7, output_size = 224):
    """
    Factory function to create decoder instances.
    
    This provides a convenient way to create different decoder types
    without needing to know their exact constructor arguments.
    
    Args:
        decoder_type: 'simple' or 'attention'
        input_channels: Number of input feature channels
        input_size: Spatial size of input features
        output_size: Desired output image size
        
    Returns:
        decoder: Instantiated decoder module
        
    Example:
        >>> decoder = create_decoder('attention', input_channels=256, 
        ...                          input_size=7, output_size=224)
    """
    if decoder_type == 'simple':
        return SimpleDecoder(input_channels, input_size, output_size)
    elif decoder_type == 'attention':
        return AttentionDecoder(input_channels, input_size, output_size)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")