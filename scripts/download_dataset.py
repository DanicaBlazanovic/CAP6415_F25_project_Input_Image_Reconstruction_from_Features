#!/usr/bin/env python3
"""
Download and setup DIV2K dataset for image reconstruction experiments.

This script downloads the DIV2K high-resolution images used for training and evaluation:
- DIV2K_train_HR: 800 high-resolution training images (~2K resolution)
- DIV2K_valid_HR: 100 high-resolution validation images (used as test set)

Usage:
    python scripts/download_dataset.py

The script will:
1. Create data/ directory structure
2. Download train and validation sets from official DIV2K source
3. Extract images to data/DIV2K_train_HR/ and data/DIV2K_test_HR/
4. Verify image counts (800 train, 100 test)

Total download size: ~3.5 GB
Total disk space needed: ~7 GB (with extracted files)
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        """Update progress bar.
        
        Args:
            b: Number of blocks transferred
            bsize: Size of each block (bytes)
            tsize: Total size (bytes)
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file from URL with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save downloaded file
    """
    print(f"Downloading: {url}")
    print(f"Saving to: {output_path}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Progress") as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    
    print(f"✓ Downloaded: {output_path}\n")


def extract_zip(zip_path, extract_to):
    """Extract zip file with progress bar.
    
    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
    """
    print(f"Extracting: {zip_path}")
    print(f"Extract to: {extract_to}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        with tqdm(total=len(members), desc="Extracting") as pbar:
            for member in members:
                zip_ref.extract(member, extract_to)
                pbar.update(1)
    
    print(f"✓ Extracted to: {extract_to}\n")


def count_images(directory):
    """Count image files in directory.
    
    Args:
        directory: Directory to count images in
        
    Returns:
        Number of image files (.png, .jpg, .jpeg)
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    count = 0
    
    for file in Path(directory).rglob('*'):
        if file.suffix.lower() in image_extensions:
            count += 1
    
    return count


def main():
    """Main download and setup function."""
    
    print("=" * 80)
    print("DIV2K Dataset Download and Setup")
    print("=" * 80)
    print()
    
    # Define paths
    project_root = Path.cwd()
    data_dir = project_root / "data"
    train_dir = data_dir / "DIV2K_train_HR"
    test_dir = data_dir / "DIV2K_test_HR"
    
    # Create directories
    print("Creating directory structure...")
    data_dir.mkdir(exist_ok=True)
    print(f"✓ Created: {data_dir}")
    print()
    
    # DIV2K URLs (official source)
    train_url = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    valid_url = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
    
    train_zip = data_dir / "DIV2K_train_HR.zip"
    valid_zip = data_dir / "DIV2K_valid_HR.zip"
    
    # Download training set
    print("-" * 80)
    print("STEP 1: Download Training Set (800 images, ~2.5 GB)")
    print("-" * 80)
    
    if train_zip.exists():
        print(f"Training zip already exists: {train_zip}")
        print("Skipping download...")
    else:
        try:
            download_url(train_url, train_zip)
        except Exception as e:
            print(f"✗ Error downloading training set: {e}")
            print("\nPlease download manually from:")
            print(f"  {train_url}")
            print(f"Save to: {train_zip}")
            sys.exit(1)
    
    # Extract training set
    if train_dir.exists() and count_images(train_dir) == 800:
        print(f"Training images already extracted: {train_dir}")
        print(f"Found 800 images. Skipping extraction...\n")
    else:
        try:
            extract_zip(train_zip, data_dir)
            # Verify extraction
            num_train = count_images(train_dir)
            if num_train != 800:
                print(f"✗ Warning: Expected 800 training images, found {num_train}")
            else:
                print(f"✓ Verified: 800 training images in {train_dir}\n")
        except Exception as e:
            print(f"✗ Error extracting training set: {e}")
            sys.exit(1)
    
    # Download validation set (used as test set)
    print("-" * 80)
    print("STEP 2: Download Validation Set (100 images, ~700 MB)")
    print("-" * 80)
    
    if valid_zip.exists():
        print(f"Validation zip already exists: {valid_zip}")
        print("Skipping download...")
    else:
        try:
            download_url(valid_url, valid_zip)
        except Exception as e:
            print(f"✗ Error downloading validation set: {e}")
            print("\nPlease download manually from:")
            print(f"  {valid_url}")
            print(f"Save to: {valid_zip}")
            sys.exit(1)
    
    # Extract validation set and rename to test set
    print()
    if test_dir.exists() and count_images(test_dir) == 100:
        print(f"Test images already extracted: {test_dir}")
        print(f"Found 100 images. Skipping extraction...\n")
    else:
        try:
            extract_zip(valid_zip, data_dir)
            
            # Rename DIV2K_valid_HR to DIV2K_test_HR (our convention)
            valid_extracted = data_dir / "DIV2K_valid_HR"
            if valid_extracted.exists() and not test_dir.exists():
                valid_extracted.rename(test_dir)
                print(f"✓ Renamed: {valid_extracted} -> {test_dir}")
            
            # Verify extraction
            num_test = count_images(test_dir)
            if num_test != 100:
                print(f"✗ Warning: Expected 100 test images, found {num_test}")
            else:
                print(f"✓ Verified: 100 test images in {test_dir}\n")
        except Exception as e:
            print(f"✗ Error extracting validation set: {e}")
            sys.exit(1)
    
    # Cleanup zip files (optional)
    print("-" * 80)
    print("STEP 3: Cleanup")
    print("-" * 80)
    
    total_zip_size = train_zip.stat().st_size + valid_zip.stat().st_size
    total_zip_size_gb = total_zip_size / (1024**3)
    
    print(f"Zip files take up {total_zip_size_gb:.2f} GB of disk space.")
    print("You can safely delete them after extraction.\n")
    
    cleanup = input("Delete zip files to save disk space? (y/n): ").lower().strip()
    if cleanup == 'y':
        try:
            if train_zip.exists():
                train_zip.unlink()
                print(f"✓ Deleted: {train_zip}")
            if valid_zip.exists():
                valid_zip.unlink()
                print(f"✓ Deleted: {valid_zip}")
            print()
        except Exception as e:
            print(f"✗ Error deleting zip files: {e}")
    else:
        print("Keeping zip files.\n")
    
    # Final summary
    print("=" * 80)
    print("SETUP COMPLETE")
    print("=" * 80)
    print()
    print("Dataset structure:")
    print(f"  {train_dir}/  ({count_images(train_dir)} images)")
    print(f"  {test_dir}/   ({count_images(test_dir)} images)")
    print()
    print("Data split used in experiments:")
    print("  - Training:   640 images (80% of DIV2K_train_HR)")
    print("  - Validation: 160 images (20% of DIV2K_train_HR)")
    print("  - Test:       100 images (DIV2K_valid_HR)")
    print()
    print("You can now run training:")
    print("  python scripts/train.py --arch vgg16 --layer block1 --decoder transposed_conv")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
