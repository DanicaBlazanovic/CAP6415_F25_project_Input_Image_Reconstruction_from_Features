"""
Dataset Loading and Preprocessing for Image Reconstruction
==========================================================

This module provides data loading functionality for the image reconstruction
project using the DIV2K dataset. It handles:
- Image loading from disk with error handling
- Train/validation/test splitting with no data leakage
- Data augmentation for training
- ImageNet-style normalization for pre-trained models
- Cross-platform compatibility (Windows, macOS, Linux)

DIV2K Dataset Structure:
    data/
    ├── DIV2K_train_HR/     800 high-resolution images (training source)
    │   ├── 0001.png
    │   ├── 0002.png
    │   └── ...
    └── DIV2K_test_HR/      100 high-resolution images (test/validation)
        ├── 0801.png
        ├── 0802.png
        └── ...

Data Split Strategy:
    - Training: 640 images (80% of DIV2K_train_HR)
    - Validation: 160 images (20% of DIV2K_train_HR)
    - Test: 100 images (100% of DIV2K_test_HR)
    Total: 900 images across all splits with no overlap

Why This Split:
    1. Separate test set: DIV2K_test_HR provides independent evaluation
    2. 80/20 train/val: Standard split ratio for small datasets
    3. Fixed seed: Ensures reproducible splits across experiments
    4. No data leakage: Test images never seen during training/validation

ImageNet Normalization:
    All pre-trained encoders (ResNet, VGG, ViT, PVT) expect ImageNet-normalized
    inputs. We use the standard normalization:
        mean = [0.485, 0.456, 0.406]  # RGB channel means from ImageNet
        std = [0.229, 0.224, 0.225]   # RGB channel standard deviations
    
    This transforms images to have approximately zero mean and unit variance,
    which matches the distribution the pre-trained models were trained on.

Data Augmentation Strategy:
    Training: Random crops and horizontal flips
        - Increases effective dataset size
        - Prevents overfitting on small dataset (640 training images)
        - Improves model generalization
    
    Validation/Test: Deterministic center crops
        - Ensures consistent evaluation metrics
        - No randomness for reproducible results

Cross-Platform Considerations:
    Windows: DataLoader multiprocessing has known issues, use num_workers=0
    macOS/Linux: Can safely use multiple workers (num_workers=4) for speed
    The code automatically detects platform and sets appropriate workers.

Author: Danica Blazanovic, Abbas Khan
Course: CAP6415 - Computer Vision, Fall 2025
Institution: Florida Atlantic University
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
    Generic PyTorch Dataset for loading images from disk.
    
    This class provides a flexible image loading interface that works with any
    folder structure containing images. It recursively searches for common image
    formats and provides robust error handling.
    
    Key Features:
    - Recursive image search (handles nested folders)
    - Multiple format support (JPG, JPEG, PNG)
    - Duplicate detection and removal
    - Graceful error handling (returns blank images on load failure)
    - Optional dataset size limiting for quick experiments
    - Cross-platform path handling using pathlib
    
    Supported Datasets:
    - DIV2K (800 train + 100 test high-resolution images)
    - ImageNet (1.2M training images)
    - COCO (330K images)
    - Tiny ImageNet (100K images)
    - Any custom image folder
    
    Args:
        root_dir (str or Path): Path to directory containing images.
                               Searches recursively for image files.
                               Example: 'data/DIV2K_train_HR'
        
        transform (torchvision.transforms.Compose, optional): Preprocessing
                                                              transforms to apply.
                                                              If None, returns
                                                              raw PIL Images.
                                                              Default: None
        
        split (str, optional): Dataset split name for logging purposes.
                              Options: 'train', 'val', 'test'
                              Does not affect functionality, only for documentation.
                              Default: 'train'
        
        limit (int, optional): Maximum number of images to load.
                              Useful for quick experiments or debugging.
                              If None or larger than dataset size, loads all images.
                              Images are loaded in sorted order before limiting.
                              Default: None
    
    Attributes:
        root_dir (Path): Root directory path as pathlib.Path object
        transform (callable): Transformation pipeline to apply
        split (str): Dataset split name
        image_paths (list): List of all image file paths as strings
    
    Example:
        >>> # Load full training set with transforms
        >>> transform = get_transforms(224, 'train')
        >>> train_dataset = ImageDataset('data/DIV2K_train_HR', transform=transform)
        >>> print(f"Loaded {len(train_dataset)} training images")
        
        >>> # Load limited dataset for quick testing
        >>> debug_dataset = ImageDataset('data/DIV2K_train_HR', 
        ...                              transform=transform, 
        ...                              limit=10)
        >>> print(f"Debug dataset: {len(debug_dataset)} images")
        
        >>> # Use with PyTorch DataLoader
        >>> loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        >>> for batch in loader:
        ...     print(f"Batch shape: {batch.shape}")  # (4, 3, 224, 224)
        ...     break
    
    Note on Image Loading:
        Images are loaded lazily (only when accessed via __getitem__).
        This prevents loading the entire dataset into memory at initialization,
        which is crucial for large datasets like ImageNet.
    
    Note on Error Handling:
        If an image fails to load (corrupt file, permission error, etc.),
        the dataset returns a blank tensor instead of crashing. This allows
        training to continue with a warning rather than stopping completely.
        The blank image will have minimal impact if errors are rare.
    """
    
    def __init__(self, root_dir, transform=None, split='train', limit=None):
        """Initialize dataset by finding all image files in root_dir."""
        
        # Convert to Path object for cross-platform compatibility
        # pathlib handles Windows backslashes and Unix forward slashes automatically
        self.root_dir = Path(root_dir)
        
        # Store transform pipeline (may be None for raw PIL images)
        self.transform = transform
        
        # Store split name for logging (does not affect functionality)
        self.split = split
        
        # Recursively find all image files in directory tree
        # This populates self.image_paths with sorted list of file paths
        self.image_paths = self._find_images()
        
        # Apply dataset size limit if specified
        # Useful for quick experiments: limit=100 loads only first 100 images
        # Images are already sorted, so limiting gives deterministic subset
        if limit and limit < len(self.image_paths):
            self.image_paths = self.image_paths[:limit]
        
        # Print dataset size for verification
        # Using plain text instead of special characters for professional output
        print(f"Found {len(self.image_paths)} images in {root_dir}")
    
    def _find_images(self):
        """
        Recursively find all image files in root_dir.
        
        This method searches the entire directory tree starting from root_dir
        for files with common image extensions. It handles:
        - Recursive search through nested folders
        - Multiple image formats (JPG, JPEG, PNG)
        - Duplicate detection (same file found multiple times)
        - Deterministic ordering (sorted for reproducibility)
        
        Search Strategy:
            Uses pathlib's rglob (recursive glob) for efficient file finding.
            rglob is cross-platform and faster than os.walk for most cases.
        
        Supported Extensions:
            - *.jpg: Most common format, smaller file sizes
            - *.jpeg: Alternative JPG extension
            - *.png: Lossless format, larger files but no compression artifacts
        
        Case Sensitivity:
            pathlib.rglob is case-insensitive on Windows and macOS by default,
            but case-sensitive on Linux. We only search for lowercase extensions
            and rely on the filesystem's native case handling.
        
        Duplicate Handling:
            In some folder structures, the same file might be found multiple
            times (e.g., through symlinks). We use set() to remove duplicates
            before returning the list.
        
        Reproducibility:
            Files are sorted alphabetically to ensure consistent ordering
            across different runs and different systems. This is crucial for:
            - Reproducible train/val splits
            - Consistent results when using dataset limits
            - Debugging (same image appears at same index)
        
        Returns:
            list of str: Sorted list of unique absolute file paths.
                        Empty list if no images found.
        
        Example Output:
            ['/path/to/data/DIV2K_train_HR/0001.png',
             '/path/to/data/DIV2K_train_HR/0002.png',
             '/path/to/data/DIV2K_train_HR/0003.png',
             ...]
        
        Performance:
            For DIV2K dataset (800 images): ~0.1 seconds
            For ImageNet (1.2M images): ~10-30 seconds depending on filesystem
        """
        
        # Define common image extensions to search for
        # Lowercase only - filesystem handles case sensitivity
        extensions = ['*.jpg', '*.jpeg', '*.png']
        
        # List to accumulate all found image paths
        image_paths = []
        
        # Search for each extension recursively
        # rglob('*.jpg') finds all .jpg files in root_dir and subdirectories
        for ext in extensions:
            # Extend list with all files matching this extension
            # rglob returns generator of Path objects
            image_paths.extend(self.root_dir.rglob(ext))
        
        # Convert Path objects to strings for compatibility with PIL
        # Remove duplicates using set() then convert back to list
        # Sort alphabetically for deterministic ordering
        image_paths = sorted(list(set([str(p) for p in image_paths])))
        
        return image_paths
    
    def __len__(self):
        """
        Return total number of images in dataset.
        
        This method is required by PyTorch's Dataset interface.
        It allows using len(dataset) and enables proper batch computation.
        
        Returns:
            int: Number of images in the dataset after any limiting.
        
        Example:
            >>> dataset = ImageDataset('data/DIV2K_train_HR')
            >>> print(len(dataset))  # 800
            >>> limited = ImageDataset('data/DIV2K_train_HR', limit=100)
            >>> print(len(limited))  # 100
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Load and return a single image by index.
        
        This method is required by PyTorch's Dataset interface and is called
        by DataLoader to fetch individual samples. It:
        1. Loads image from disk using PIL
        2. Converts to RGB (handles grayscale and RGBA)
        3. Applies transforms if provided
        4. Returns tensor or PIL Image
        
        Image Loading:
            Uses PIL (Pillow) for robust image loading across formats.
            PIL can handle various image types and color modes.
        
        RGB Conversion:
            .convert('RGB') ensures all images have 3 channels:
            - Grayscale (1 channel): Duplicated to 3 channels
            - RGBA (4 channels): Alpha channel discarded
            - RGB (3 channels): Unchanged
            This standardization is crucial because our models expect 3-channel input.
        
        Transform Pipeline:
            If transforms are provided, they are applied in sequence:
            1. Resize to target size + margin
            2. Random/center crop to exact size
            3. Random horizontal flip (training only)
            4. Convert to tensor (PIL Image -> torch.Tensor)
            5. Normalize using ImageNet statistics
        
        Error Handling:
            If image loading fails (corrupt file, I/O error, permission denied):
            - Print warning with filename and error message
            - Return blank tensor/image to prevent training crash
            - Blank images have minimal impact if errors are rare
            This robust handling allows training to continue even if some images fail.
        
        Args:
            idx (int): Index of image to load. Must be in range [0, len(self)-1].
        
        Returns:
            torch.Tensor: Transformed image tensor of shape (C, H, W) if transforms provided.
                         Typical shape: (3, 224, 224) for ImageNet-style models.
            
            PIL.Image.Image: Raw PIL Image if no transforms provided.
                            Useful for visualization or custom processing.
        
        Raises:
            IndexError: If idx is out of range (handled by Python's list indexing)
        
        Example:
            >>> dataset = ImageDataset('data/DIV2K_train_HR', 
            ...                       transform=get_transforms(224, 'train'))
            >>> img = dataset[0]  # Load first image
            >>> print(img.shape)  # torch.Size([3, 224, 224])
            >>> print(img.min(), img.max())  # Normalized values, roughly [-2, 2]
        
        Performance:
            Image loading time depends on:
            - Image size: DIV2K images are ~2000x2000, ~1-5 MB each
            - Storage speed: SSD vs HDD makes significant difference
            - Transform complexity: Resize and crop are relatively fast
            Typical loading time: 10-50ms per image on modern hardware
        """
        
        # Get file path for this index
        img_path = self.image_paths[idx]
        
        try:
            # Load image from disk using PIL (Python Imaging Library)
            # PIL supports many formats: JPEG, PNG, BMP, TIFF, etc.
            image = Image.open(img_path)
            
            # Convert to RGB mode (3 channels) regardless of original format
            # This handles:
            # - Grayscale images (L mode): Converts to RGB by duplicating channel
            # - RGBA images (4 channels): Removes alpha channel
            # - Palette images (P mode): Converts to RGB using palette
            # - RGB images: No change
            image = image.convert('RGB')
            
            # Apply transformation pipeline if provided
            # Transforms typically include: resize, crop, flip, normalize
            if self.transform:
                image = self.transform(image)
            
            # Return transformed tensor or raw PIL image
            return image
        
        except Exception as e:
            # Image loading failed - print warning for debugging
            # Using plain text warning instead of special characters
            print(f"Warning: Error loading {img_path}: {e}")
            
            # Return blank image to prevent training crash
            # This allows training to continue with degraded data rather than stopping
            if self.transform:
                # Return blank tensor matching expected output shape
                # Assumes 3-channel RGB and 224x224 size (standard for ImageNet)
                return torch.zeros(3, 224, 224)
            else:
                # Return blank PIL image if no transforms
                # RGB mode with standard size
                return Image.new('RGB', (224, 224))


def get_transforms(img_size=224, split='train'):
    """
    Create standard preprocessing transform pipeline for images.
    
    This function returns torchvision transform pipelines optimized for
    image reconstruction tasks using pre-trained encoders. The transforms
    differ between training and validation/test splits.
    
    Transform Design Philosophy:
    
    Training Pipeline:
        Goal: Maximize data diversity to prevent overfitting
        Strategy: Aggressive augmentation while preserving image content
        
        1. Resize to larger size (256x256 for target 224x224)
           Why: Provides margin for random crops
           Effect: Each image can yield multiple different crops
        
        2. Random Crop to target size
           Why: Creates variation from single source image
           Effect: Different spatial regions in each epoch
           Benefit: Increases effective dataset size from 640 to effectively infinite
        
        3. Random Horizontal Flip (50% probability)
           Why: Natural augmentation for most image content
           Effect: Doubles dataset variations
           Note: Vertical flip not used (less natural for most scenes)
        
        4. ToTensor: Convert PIL Image to PyTorch tensor
           Input: PIL Image with values [0, 255] as uint8
           Output: Float tensor with values [0, 1]
           Shape: (C, H, W) instead of PIL's (H, W, C)
        
        5. Normalize using ImageNet statistics
           Why: Match distribution that pre-trained models expect
           Input: Tensor with values [0, 1]
           Output: Tensor with mean approximately 0, std approximately 1
           Formula: output = (input - mean) / std
    
    Validation/Test Pipeline:
        Goal: Consistent, reproducible evaluation
        Strategy: No randomness, deterministic preprocessing
        
        1. Resize to same larger size (consistency with training)
        2. Center Crop (deterministic alternative to random crop)
           Why: Takes center region instead of random location
           Effect: Same crop every time for reproducible metrics
        3. ToTensor and Normalize (identical to training)
    
    ImageNet Normalization Statistics:
        These are the channel-wise mean and standard deviation computed
        over the entire ImageNet-1K training set (1.28M images).
        
        Mean = [0.485, 0.456, 0.406]  # RGB channels
            - Red channel: mean = 0.485 (slightly less than mid-gray)
            - Green channel: mean = 0.456 (slightly less than mid-gray)
            - Blue channel: mean = 0.406 (notably less, skewed toward warm colors)
        
        Std = [0.229, 0.224, 0.225]  # RGB channels
            - All channels have similar standard deviation around 0.22-0.23
            - Indicates roughly similar variation across color channels
        
        Why These Specific Values:
            All torchvision pre-trained models (ResNet, VGG, ViT, etc.) were
            trained with these normalization statistics. Using the same values
            ensures our images match the distribution the models expect.
        
        Effect on Pixel Values:
            After normalization, typical pixel values range from approximately
            -2 to +2 instead of [0, 1]. This distribution is more conducive to
            neural network training (centered around zero).
    
    Why Different Transforms for Train vs Val/Test:
        Training: Augmentation improves generalization
            - Small dataset (640 images) benefits greatly from augmentation
            - Random variations prevent overfitting to specific spatial crops
            - Each epoch sees different crops, effectively multiplying data
        
        Validation/Test: Deterministic processing ensures fair comparison
            - Same crop location every time for consistent metrics
            - No randomness means reproducible results
            - Allows comparing models fairly (same input preprocessing)
    
    Args:
        img_size (int, optional): Target square image size in pixels.
                                 Standard: 224 (ImageNet default)
                                 Can use 256, 384, 512 for larger models
                                 Default: 224
        
        split (str, optional): Dataset split to determine augmentation.
                              'train': Use random crops and flips
                              'val' or 'test': Use deterministic center crop
                              Default: 'train'
    
    Returns:
        torchvision.transforms.Compose: Composed transformation pipeline.
                                       Can be called on PIL Images.
    
    Example:
        >>> # Create training transforms
        >>> train_transform = get_transforms(224, 'train')
        >>> 
        >>> # Load and transform image
        >>> from PIL import Image
        >>> img = Image.open('data/DIV2K_train_HR/0001.png')
        >>> tensor = train_transform(img)
        >>> print(tensor.shape)  # torch.Size([3, 224, 224])
        >>> print(tensor.mean(), tensor.std())  # approximately 0, 1
        >>> 
        >>> # Create validation transforms
        >>> val_transform = get_transforms(224, 'val')
        >>> val_tensor = val_transform(img)  # Deterministic result
    
    Note on Resize + Crop Strategy:
        Why resize to 256 then crop to 224 instead of direct resize to 224?
        
        1. Preserves aspect ratio better:
           Original images may have various aspect ratios
           Resize to 256 on shorter side maintains some original proportions
           Final crop gets square 224x224 region
        
        2. Provides crop margin:
           32 pixel margin (256 - 224 = 32) allows meaningful random crops
           Too small margin would give nearly identical crops
           Too large margin might lose too much context
        
        3. Standard practice:
           This resize + crop strategy is standard in ImageNet training
           Proven effective across many computer vision tasks
    """
    
    if split == 'train':
        # Training pipeline with data augmentation
        # Goal: Maximize diversity while preserving content
        return transforms.Compose([
            # Resize shorter edge to img_size + 32
            # Maintains aspect ratio, allows room for random crop
            transforms.Resize(img_size + 32),
            
            # Randomly crop to exact target size
            # Different crop location each time this transform is applied
            # Effectively multiplies dataset size
            transforms.RandomCrop(img_size),
            
            # Randomly flip horizontally with 50% probability
            # Natural augmentation that preserves image semantics
            # Example: flipped car is still a valid car image
            transforms.RandomHorizontalFlip(p=0.5),
            
            # Convert PIL Image to PyTorch tensor
            # Input: PIL Image, uint8 values [0, 255], shape (H, W, C)
            # Output: Float tensor, values [0, 1], shape (C, H, W)
            transforms.ToTensor(),
            
            # Normalize using ImageNet statistics
            # Input: Tensor with values [0, 1]
            # Output: Tensor with approximately zero mean and unit variance
            # These specific values match pre-trained model expectations
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet RGB channel means
                std=[0.229, 0.224, 0.225]    # ImageNet RGB channel std devs
            )
        ])
    else:
        # Validation/test pipeline without augmentation
        # Goal: Consistent, reproducible preprocessing
        return transforms.Compose([
            # Same resize as training for consistency
            transforms.Resize(img_size + 32),
            
            # Center crop instead of random crop
            # Takes center region deterministically
            # Same crop every time for reproducible evaluation
            transforms.CenterCrop(img_size),
            
            # ToTensor and Normalize identical to training
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def get_dataloaders(data_dir='data', img_size=224, batch_size=16, 
                   num_workers='auto', limit=None, seed=42):
    """
    Create PyTorch DataLoaders for training, validation, and test sets.
    
    This function sets up the complete data loading pipeline for the image
    reconstruction project. It handles:
    - Loading images from DIV2K dataset directories
    - Splitting training data into train/validation sets
    - Creating separate test set from official DIV2K validation
    - Applying appropriate transforms to each split
    - Configuring DataLoader parameters for optimal performance
    - Ensuring reproducibility through fixed random seeds
    - Handling platform-specific multiprocessing issues
    
    Dataset Organization:
    
    Directory Structure:
        data/
        ├── DIV2K_train_HR/    800 images (source for train + val)
        │   ├── 0001.png       2040x1356 pixels, ~5 MB
        │   ├── 0002.png       2040x1356 pixels, ~4 MB
        │   └── ...
        └── DIV2K_test_HR/     100 images (test set)
            ├── 0801.png       Various sizes, ~3-7 MB each
            ├── 0802.png
            └── ...
    
    Data Split Strategy:
        
        Training Set: 640 images (80% of DIV2K_train_HR)
            - Used for model parameter updates
            - Gradient computation and backpropagation
            - Augmented with random crops and flips
            - Shuffled every epoch for varied mini-batches
        
        Validation Set: 160 images (20% of DIV2K_train_HR)
            - Used for hyperparameter tuning and early stopping
            - No gradient computation (model.eval() mode)
            - Deterministic preprocessing (center crops)
            - Not shuffled (consistent ordering)
        
        Test Set: 100 images (100% of DIV2K_test_HR)
            - Used for final model evaluation only
            - Never seen during training or validation
            - Official DIV2K validation set
            - Reports final performance metrics
    
    Why This Split:
        
        1. Separate test directory prevents accidental data leakage
           - Test images physically separated from training
           - Impossible to accidentally use test images in training
        
        2. 80/20 train/val split is standard for small datasets
           - Maximizes training data (640 images)
           - Provides sufficient validation data (160 images)
           - Common practice in computer vision research
        
        3. Fixed random seed ensures reproducibility
           - Same split across different runs
           - Same split across different machines
           - Critical for comparing model versions
        
        4. No overlap between splits
           - Each image appears in exactly one split
           - Training: indices 0-639 (80%)
           - Validation: indices 640-799 (20%)
           - Test: separate directory entirely
    
    Reproducibility Considerations:
        
        Random Seed:
            Setting np.random.seed(42) before creating train/val split ensures:
            - Same random permutation every run
            - Same images in train vs val across experiments
            - Consistent results when comparing models
        
        Why seed=42:
            Common convention in machine learning (from Hitchhiker's Guide)
            Any fixed value works, but 42 is recognizable standard
        
        Sorted Image Paths:
            Image paths are sorted before splitting to ensure:
            - Deterministic ordering regardless of filesystem
            - Same split on Windows, macOS, and Linux
            - Reproducible results across systems
    
    DataLoader Configuration:
        
        Batch Size:
            Default: 16 images per batch (can be overridden)
            Considerations:
            - Larger batches: More stable gradients, faster training
            - Smaller batches: Less memory, potentially better generalization
            - Our default (16): Balance for 16-48GB GPU memory
            - Adjust based on available GPU memory
        
        Shuffling:
            Training: shuffle=True
                - Different order each epoch
                - Prevents model from learning sequence patterns
                - Improves generalization
            Validation/Test: shuffle=False
                - Consistent ordering for reproducible metrics
                - Same order every time for fair comparison
        
        Number of Workers:
            Controls parallel data loading processes
            
            Windows: num_workers=0 (single process)
                - Windows has issues with multiprocessing in DataLoader
                - Known PyTorch limitation on Windows
                - Fork() not available, spawn() is slow
                - Single process avoids these issues
            
            macOS/Linux: num_workers=4 (4 parallel processes)
                - Parallel loading speeds up training significantly
                - 4 workers can prepare batches while GPU trains
                - Reduces GPU idle time waiting for data
                - Optimal number depends on CPU cores (typically 2-8)
            
            Auto-detection:
                'auto' (default) detects platform and sets appropriately
                Ensures code runs optimally on any system
        
        Pin Memory:
            pin_memory=True when CUDA available
                - Pre-allocates page-locked memory on CPU
                - Faster CPU-to-GPU transfer
                - Reduces data transfer time during training
                - No benefit without CUDA (set False for CPU/MPS)
    
    Memory Considerations:
        
        DIV2K images are large (2040x1356 typical, ~5MB each)
        After preprocessing: 224x224x3 float32 = ~600KB per image
        
        Memory usage per batch:
            batch_size=16: ~10 MB in GPU memory
            batch_size=32: ~20 MB in GPU memory
            
        With num_workers > 0:
            Each worker process loads images in parallel
            Memory usage increases with more workers
            Typically not an issue unless num_workers > 8
    
    Args:
        data_dir (str, optional): Root directory containing DIV2K folders.
                                 Must contain 'DIV2K_train_HR' and 'DIV2K_test_HR'.
                                 Default: 'data'
        
        img_size (int, optional): Target image size (square) in pixels.
                                 Images will be resized and cropped to this size.
                                 Standard: 224 for ImageNet-sized models
                                 Default: 224
        
        batch_size (int, optional): Number of images per batch.
                                   Larger batches need more GPU memory.
                                   Typical range: 4-32 for 16-48GB GPUs
                                   Default: 16
        
        num_workers (int or str, optional): Number of data loading processes.
                                           'auto': Platform-specific (recommended)
                                           0: Single process (Windows safe)
                                           >0: Parallel loading (faster on Linux/Mac)
                                           Default: 'auto'
        
        limit (int, optional): Limit training dataset size.
                              Useful for quick experiments and debugging.
                              If None, uses all 800 training images.
                              Validation/test always use full sets.
                              Example: limit=100 for quick test run
                              Default: None
        
        seed (int, optional): Random seed for train/val split.
                             Same seed produces same split.
                             Different seeds produce different splits.
                             Standard value: 42
                             Default: 42
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
            
            train_loader (DataLoader): Training set loader
                - 640 images (or limited amount)
                - Shuffled each epoch
                - Augmented transforms (random crop/flip)
                - For gradient computation and weight updates
            
            val_loader (DataLoader): Validation set loader
                - 160 images (20% of training)
                - Not shuffled
                - Deterministic transforms (center crop)
                - For hyperparameter tuning and early stopping
            
            test_loader (DataLoader): Test set loader
                - 100 images (official DIV2K validation)
                - Not shuffled
                - Deterministic transforms
                - For final performance evaluation
    
    Example:
        >>> # Standard usage
        >>> train_loader, val_loader, test_loader = get_dataloaders()
        >>> print(f"Train batches: {len(train_loader)}")  # 640/16 = 40 batches
        >>> print(f"Val batches: {len(val_loader)}")      # 160/16 = 10 batches
        >>> print(f"Test batches: {len(test_loader)}")    # 100/16 = 7 batches (rounded up)
        >>> 
        >>> # Quick debugging with limited dataset
        >>> debug_train, debug_val, debug_test = get_dataloaders(limit=32)
        >>> print(f"Debug train batches: {len(debug_train)}")  # 32/16 = 2 batches
        >>> 
        >>> # Custom batch size for smaller GPU
        >>> small_train, small_val, small_test = get_dataloaders(batch_size=4)
        >>> 
        >>> # Iterate through training data
        >>> for batch_idx, images in enumerate(train_loader):
        ...     print(f"Batch {batch_idx}: {images.shape}")  # (16, 3, 224, 224)
        ...     # Training code here
    
    Raises:
        FileNotFoundError: If data_dir, DIV2K_train_HR, or DIV2K_test_HR not found
    
    Note on DataLoader Shuffling:
        Training loader is shuffled to ensure varied mini-batches each epoch.
        This prevents the model from learning spurious patterns based on
        image ordering and improves generalization. Validation and test loaders
        are not shuffled to ensure consistent, reproducible evaluation metrics.
    
    Note on Transform Consistency:
        Training images use augmented transforms (random crop/flip) while
        validation/test use deterministic transforms (center crop). This is
        standard practice: augmentation for training generalization,
        deterministic processing for fair evaluation.
    """
    
    # Auto-detect appropriate number of workers based on platform
    # Windows has known multiprocessing issues in DataLoader
    if num_workers == 'auto':
        # Use single process on Windows to avoid multiprocessing errors
        # Use 4 parallel workers on macOS/Linux for faster loading
        num_workers = 0 if platform.system() == 'Windows' else 4
    
    # Create transform pipelines for training and evaluation
    # Training: augmented with random crops and flips
    # Evaluation: deterministic with center crops
    train_transform = get_transforms(img_size, 'train')
    eval_transform = get_transforms(img_size, 'val')
    
    # Load full training dataset from DIV2K_train_HR folder
    # This contains 800 high-resolution images
    # Apply training transforms (random crop/flip)
    # If limit specified, only load first 'limit' images
    full_train_dataset = ImageDataset(
        f'{data_dir}/DIV2K_train_HR',
        train_transform,
        'train',
        limit  # Apply size limit to training only
    )
    
    # Print data directory for verification
    print(data_dir)
    
    # Split training data into train (80%) and validation (20%)
    # Use fixed random seed for reproducible splits
    np.random.seed(seed)
    num_train = len(full_train_dataset)
    print("num of train", num_train)
    
    # Create random permutation of indices
    # This shuffles the order of images before splitting
    # Ensures random distribution in train vs val
    indices = np.random.permutation(num_train)
    
    # Calculate split point for 80/20 division
    print(num_train)
    train_size = int(0.8 * num_train)
    
    # First 80% of shuffled indices go to training
    train_indices = indices[:train_size]
    
    # Remaining 20% go to validation
    val_indices = indices[train_size:]
    
    # Print split sizes for verification
    print("train indices length is ", len(train_indices))
    print("val indices length is ", len(val_indices))
    
    # Create training subset using selected indices
    # Subset wraps the full dataset and only exposes selected indices
    train_dataset = Subset(full_train_dataset, train_indices)
    
    # Verify training set size
    print(len(train_dataset))
    
    # Create validation dataset with evaluation transforms
    # Note: Must reload dataset with different transforms
    # Cannot reuse training dataset because it has augmentation
    val_full_dataset = ImageDataset(
        f'{data_dir}/DIV2K_train_HR',
        eval_transform,  # Use deterministic transforms for validation
        'val',
        limit  # Apply same limit as training for consistency
    )
    
    # Create validation subset using same indices as above
    # This ensures same images as validation, just with different transforms
    val_dataset = Subset(val_full_dataset, val_indices)
    
    # Load test dataset from separate DIV2K_test_HR folder
    # This is the official DIV2K validation set (100 images)
    # Never limit test set - always use all 100 images for consistent evaluation
    test_dataset = ImageDataset(
        f'{data_dir}/DIV2K_test_HR',
        eval_transform,  # Use deterministic transforms
        'test',
        None  # Always use full test set regardless of training limit
    )
    
    # Create DataLoader for training set
    # DataLoader handles batching, shuffling, and parallel loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data each epoch for varied batches
        num_workers=num_workers,  # Parallel loading processes
        pin_memory=torch.cuda.is_available()  # Speed up GPU transfer if available
    )
    
    # Create DataLoader for validation set
    # No shuffling for consistent evaluation
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation for reproducible metrics
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Create DataLoader for test set
    # No shuffling for consistent evaluation
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test for reproducible metrics
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Return all three loaders as tuple
    return train_loader, val_loader, test_loader


def denormalize(tensor):
    """
    Reverse ImageNet normalization for visualization.
    
    This function converts normalized tensors back to the [0, 1] range suitable
    for display. All images in this project are normalized using ImageNet
    statistics for compatibility with pre-trained models. Before visualization
    or saving, we need to reverse this normalization.
    
    Normalization Formula:
        During preprocessing, each image is normalized as:
            normalized = (image - mean) / std
        
        Where:
            image: Original tensor in [0, 1] range
            mean: Channel-wise mean from ImageNet
            std: Channel-wise standard deviation from ImageNet
        
        This transforms images to have approximately:
            mean = 0
            std = 1
        
        After normalization, pixel values typically range from -2 to +2.
    
    Denormalization Formula:
        To reverse the normalization:
            image = (normalized * std) + mean
        
        This recovers the original [0, 1] range.
        
        Derivation:
            normalized = (image - mean) / std
            normalized * std = image - mean
            image = (normalized * std) + mean
    
    Why Denormalize:
        
        1. Visualization:
           Matplotlib, PIL, and OpenCV expect images in [0, 1] or [0, 255]
           Normalized images appear strange (negative values, wrong colors)
           Denormalization restores natural appearance
        
        2. Saving Images:
           Image saving libraries expect standard ranges
           Denormalized images save correctly as PNG/JPG
        
        3. Metrics Computation:
           Some metrics (PSNR, SSIM) expect [0, 1] range
           Denormalization ensures correct metric calculation
        
        4. Human Inspection:
           Debugging and quality checks require viewing actual images
           Denormalized images show what model actually reconstructed
    
    ImageNet Statistics:
        Mean = [0.485, 0.456, 0.406]  # RGB channels
            - Computed over 1.28 million ImageNet training images
            - Red slightly higher (images tend toward warm colors)
            - Blue notably lower (fewer blue-heavy images)
        
        Std = [0.229, 0.224, 0.225]   # RGB channels
            - Similar values across channels
            - Indicates roughly equal variation in each color
    
    Tensor Device Handling:
        The function automatically moves mean/std tensors to the same device
        as the input tensor. This is crucial because:
        - Input might be on CPU, CUDA, or MPS
        - Mean/std must be on same device for tensor operations
        - .to(tensor.device) handles this automatically
    
    Batch vs Single Image:
        The function handles both cases:
        - Single image: shape (C, H, W) = (3, 224, 224)
        - Batch of images: shape (B, C, H, W) = (16, 3, 224, 224)
        
        For batches, mean and std are broadcast across batch dimension.
    
    Clamping:
        After denormalization, values are clamped to [0, 1]:
            - Handles numerical errors from floating point arithmetic
            - Ensures output is valid for visualization
            - Typically, clamping affects <0.1% of pixels
    
    Args:
        tensor (torch.Tensor): Normalized image tensor.
                              Single image: shape (C, H, W) = (3, H, W)
                              Batch: shape (B, C, H, W) = (B, 3, H, W)
                              Values typically in range [-2, 2]
                              Normalized with ImageNet statistics
    
    Returns:
        torch.Tensor: Denormalized image tensor in [0, 1] range.
                     Same shape as input.
                     Suitable for visualization and saving.
                     Values clamped to [0, 1] to handle numerical errors.
    
    Example:
        >>> # Denormalize single image
        >>> normalized = torch.randn(3, 224, 224)  # Simulated normalized image
        >>> denormalized = denormalize(normalized)
        >>> print(denormalized.min(), denormalized.max())  # 0.0, 1.0
        >>> 
        >>> # Denormalize batch
        >>> batch = torch.randn(16, 3, 224, 224)
        >>> denorm_batch = denormalize(batch)
        >>> print(denorm_batch.shape)  # torch.Size([16, 3, 224, 224])
        >>> 
        >>> # Use for visualization
        >>> import matplotlib.pyplot as plt
        >>> img = denormalize(model_output[0])  # Denormalize first image
        >>> plt.imshow(img.permute(1, 2, 0).cpu().numpy())  # Convert to HWC
        >>> plt.show()
        >>> 
        >>> # Save to disk
        >>> from torchvision.utils import save_image
        >>> save_image(denormalize(reconstruction), 'output.png')
    
    Note on Precision:
        Denormalization is not perfectly invertible due to:
        - Floating point rounding errors
        - Potential clipping during normalization
        - Numerical precision limits
        However, differences are negligible (<1e-6) for practical purposes.
    
    Note on Gradient Flow:
        This function can be used in training if needed (e.g., for loss
        computation on denormalized images). PyTorch will track gradients
        through the operations if input has requires_grad=True.
    """
    
    # ImageNet normalization statistics
    # These are the same values used in get_transforms()
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    # Reshape to match image dimensions: (C,) -> (C, 1, 1)
    # This allows broadcasting across spatial dimensions (H, W)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    
    # Move tensors to same device as input
    # Critical for GPU tensors - operations must be on same device
    mean = mean.to(tensor.device)
    std = std.to(tensor.device)
    
    # Handle both single images and batches
    if tensor.dim() == 4:  # Batch of images: (B, C, H, W)
        # Add batch dimension: (C, 1, 1) -> (1, C, 1, 1)
        # Allows broadcasting across batch dimension
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    # If tensor.dim() == 3, mean and std are already correct shape (C, 1, 1)
    
    # Denormalize: reverse the normalization formula
    # normalized = (image - mean) / std
    # image = (normalized * std) + mean
    denorm = tensor * std + mean
    
    # Clamp to [0, 1] range to handle numerical errors
    # Floating point arithmetic may produce values slightly outside [0, 1]
    # Clamping ensures valid range for visualization and metrics
    return denorm.clamp(0, 1)
