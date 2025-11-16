"""
PatchGAN Discriminator for Adversarial Training

Discriminator judges whether 70x70 image patches are real or reconstructed.
Used in Run 2 to add adversarial loss for sharper, more realistic reconstructions.

Architecture based on pix2pix (Isola et al., 2017).
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator for image reconstruction quality assessment.
    
    Judges authenticity of 70x70 patches rather than full images.
    This design:
    - Faster training (fewer parameters than full-image discriminator)
    - Better gradient signal (judges local realism)
    - More stable (avoids mode collapse)
    
    Architecture: C64-C128-C256-C512
    where Ck = Conv-BatchNorm-LeakyReLU with k filters
    
    Input: 224x224 RGB image
    Output: 26x26 patch predictions (each = real/fake for 70x70 receptive field)
    """
    
    def __init__(self, in_channels=3):
        """
        Initialize PatchGAN discriminator.
        
        Args:
            in_channels: Input channels (3 for RGB images)
        """
        super().__init__()
        
        # First layer: Conv without BatchNorm (standard for discriminators)
        # 224x224x3 -> 112x112x64
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Second layer: C128
        # 112x112x64 -> 56x56x128
        self.layer2 = self._make_layer(64, 128)
        
        # Third layer: C256
        # 56x56x128 -> 28x28x256
        self.layer3 = self._make_layer(128, 256)
        
        # Fourth layer: C512
        # 28x28x256 -> 26x26x512 (stride=1, no downsampling)
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final layer: output patch predictions
        # 26x26x512 -> 26x26x1
        self.final = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        
    def _make_layer(self, in_channels, out_channels):
        """
        Create Conv-BatchNorm-LeakyReLU block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            
        Returns:
            Sequential block with conv, batchnorm, activation
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass through discriminator.
        
        Args:
            x: Input images [B, 3, 224, 224]
            
        Returns:
            predictions: Patch predictions [B, 1, 26, 26]
                        Each value = probability that 70x70 patch is real
        """
        x = self.layer1(x)   # [B, 64, 112, 112]
        x = self.layer2(x)   # [B, 128, 56, 56]
        x = self.layer3(x)   # [B, 256, 28, 28]
        x = self.layer4(x)   # [B, 512, 26, 26]
        x = self.final(x)    # [B, 1, 26, 26]
        return x


def test_discriminator():
    """Test discriminator with dummy input."""
    print("\nTesting PatchGAN Discriminator...")
    print("="*60)
    
    # Create discriminator
    discriminator = PatchGANDiscriminator(in_channels=3)
    
    # Count parameters
    num_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Parameters: {num_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    print(f"\nInput shape: {dummy_input.shape}")
    
    output = discriminator(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Expected: [2, 1, 26, 26]
    assert output.shape == (2, 1, 26, 26), f"Wrong output shape: {output.shape}"
    
    print("\n" + "="*60)
    print("Discriminator test passed!")
    print("="*60 + "\n")


if __name__ == '__main__':
    test_discriminator()