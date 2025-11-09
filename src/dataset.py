"""
Dataset loaders for image reconstruction project.

Handles data loading, preprocessing, and augmentation.
Uses proper train/validation/test split with DIV2K dataset.
Cross-platform compatible (Windows/Mac/Linux).
"""

import os
import glob
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import platform
import numpy as np


class ImageDataset(Dataset):
    """
    Generic image dataset loader.
    
    Works with any folder structure containing images.
    Supports: DIV2K, ImageNet, COCO, Tiny ImageNet, etc.
    
    Args:
        root_dir: Path to image directory
        transform: torchvision transforms to apply
        split: 'train', 'val', or 'test' (for documentation)
        limit: Maximum number of images to load (useful for quick testing)
    
    Example:
        >>> dataset = ImageDataset('data/DIV2K_train_HR', transform=get_transforms(), limit=100)
        >>> print(f"Loaded {len(dataset)} images")
    """
    
    def __init__(self, root_dir, transform=None, split='train', limit=None):
        self.root_dir = Path(root_dir)  # Use Path for cross-platform compatibility
        self.transform = transform
        self.split = split
        
        # Find all image files recursively
        self.image_paths = self._find_images()
        
        # Limit number of images if specified (for quick experiments)
        if limit and limit < len(self.image_paths):
            self.image_paths = self.image_paths[:limit]
        
        print(f"✓ Found {len(self.image_paths)} images in {root_dir}")
    
    def _find_images(self):
        """
        Recursively find all image files in root_dir.
        
        Searches for common image extensions: jpg, jpeg, png
        Works across different folder structures.
        
        Returns:
            List of image file paths (as strings)
        """
        # Common image extensions (including uppercase variants)
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']
        image_paths = []
        
        # Use rglob for recursive search (cross-platform)
        for ext in extensions:
            image_paths.extend(self.root_dir.rglob(ext))
        
        # Convert Path objects to strings and sort for reproducibility
        return sorted([str(p) for p in image_paths])
    
    def __len__(self):
        """Return total number of images in dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Load and return a single image.
        
        Args:
            idx: Image index
            
        Returns:
            image: Transformed image tensor [C, H, W]
        
        Note: If image fails to load, returns a blank tensor to avoid crashes
        """
        img_path = self.image_paths[idx]
        
        try:
            # Load image and ensure RGB (handle grayscale and RGBA)
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms if provided
            if self.transform:
                image = self.transform(image)
            
            return image
        
        except Exception as e:
            print(f"⚠ Error loading {img_path}: {e}")
            # Return blank image tensor on error (prevents training crash)
            if self.transform:
                return torch.zeros(3, 224, 224)
            else:
                return Image.new('RGB', (224, 224))


def get_transforms(img_size=224, split='train'):
    """
    Get standard preprocessing transforms.
    
    Based on ImageNet normalization statistics used in pre-trained models.
    
    Normalization values from PyTorch documentation:
        mean = [0.485, 0.456, 0.406]  # ImageNet channel means
        std = [0.229, 0.224, 0.225]   # ImageNet channel std devs
    
    These normalize images to have mean≈0 and std≈1, which helps training.
    
    Args:
        img_size: Target image size (square)
        split: 'train' (with augmentation) or 'val'/'test' (without augmentation)
        
    Returns:
        Composed transforms
    """
    if split == 'train':
        # Training transforms: augmentation for better generalization
        return transforms.Compose([
            # Resize to slightly larger than target
            transforms.Resize(img_size + 32),
            
            # Random crop to target size (adds variation)
            transforms.RandomCrop(img_size),
            
            # Random horizontal flip (50% chance)
            transforms.RandomHorizontalFlip(p=0.5),
            
            # Convert PIL Image to tensor [0, 1]
            transforms.ToTensor(),
            
            # Normalize using ImageNet statistics
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/test transforms: no augmentation for consistent evaluation
        return transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),  # Deterministic crop
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def get_dataloaders(data_dir='data', img_size = 224, batch_size = 16, num_workers = 'auto', limit = None, seed = 42):
    """
    Create train, validation, and test dataloaders using proper DIV2K splits.
    
    Uses separate directories and proper splitting:
    - Training: 640 images (80% of DIV2K_train_HR)
    - Validation: 160 images (20% of DIV2K_train_HR)
    - Test: 100 images (DIV2K_test_HR - official validation set)
    
    No data leakage between splits.
    
    Automatically handles platform-specific settings (Windows has issues with
    multiprocessing in DataLoader, so we use num_workers=0 on Windows).
    
    Args:
        data_dir: Root data directory containing DIV2K_train_HR and DIV2K_test_HR folders
        img_size: Target image size
        batch_size: Number of images per batch
        num_workers: Number of worker processes ('auto' detects platform)
        limit: Limit training dataset size (for quick testing). Val/test use full sets.
        seed: Random seed for reproducible train/val split
        
    Returns:
        train_loader, val_loader, test_loader: PyTorch DataLoaders
        
    Example:
        >>> train_loader, val_loader, test_loader = get_dataloaders('data', batch_size=8, limit=100)
        >>> print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    """
    # Auto-detect appropriate num_workers based on platform
    # Windows has multiprocessing issues, so use single process
    if num_workers == 'auto':
        num_workers = 0 if platform.system() == 'Windows' else 4
    
    # Get transforms for train and validation/test
    train_transform = get_transforms(img_size, 'train')
    eval_transform = get_transforms(img_size, 'val')
    
    # Load full training dataset (800 images from DIV2K_train_HR)
    full_train_dataset = ImageDataset(
        f'{data_dir}/DIV2K_train_HR',
        train_transform,
        'train',
        limit  # Apply limit only to training if specified
    )
    
    # Split training data into train (80%) and validation (20%)
    # Use fixed random seed for reproducibility
    np.random.seed(seed)
    num_train = len(full_train_dataset)
    indices = np.random.permutation(num_train)
    
    # Calculate split point (80/20 split)
    train_size = int(0.8 * num_train)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create train and validation subsets
    train_dataset = Subset(full_train_dataset, train_indices)
    
    # Validation uses eval transforms, so we need to recreate dataset
    val_full_dataset = ImageDataset(
        f'{data_dir}/DIV2K_train_HR',
        eval_transform,
        'val',
        limit
    )
    val_dataset = Subset(val_full_dataset, val_indices)
    
    # Test set: 100 images from DIV2K_test_HR (official validation set, no limit)
    test_dataset = ImageDataset(
        f'{data_dir}/DIV2K_test_HR',
        eval_transform,
        'test',
        None  # Always use all 100 test images
    )
    
    # Create DataLoaders
    # pin_memory = True speeds up CPU->GPU transfer (only if CUDA available)
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,  # Shuffle training data each epoch
        num_workers = num_workers,
        pin_memory = torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle = False,  # Don't shuffle validation (consistent evaluation)
        num_workers = num_workers,
        pin_memory = torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,  # Don't shuffle test (consistent evaluation)
        num_workers = num_workers,
        pin_memory = torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader


def denormalize(tensor):
    """
    Reverse ImageNet normalization for visualization.
    
    Converts normalized tensors back to [0, 1] range for display.
    
    The normalization applied:
        normalized = (image - mean) / std
    
    To reverse:
        image = (normalized * std) + mean
    
    Args:
        tensor: Normalized tensor [C, H, W] or [B, C, H, W]
        
    Returns:
        denorm: Denormalized tensor in [0, 1] range
    """
    # ImageNet normalization statistics
    # Move to same device as input tensor
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(tensor.device)
    
    # Handle both single images [C, H, W] and batches [B, C, H, W]
    if tensor.dim() == 4:  # Batch of images
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    # Denormalize: image = (normalized * std) + mean
    denorm = tensor * std + mean
    
    # Clamp to [0, 1] range (in case of numerical errors)
    return denorm.clamp(0, 1)