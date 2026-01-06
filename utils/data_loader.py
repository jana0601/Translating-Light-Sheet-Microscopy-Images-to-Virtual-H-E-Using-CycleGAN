"""
Data loader for fluorescence microscopy images and H&E stained images.
"""
import os
import glob
from typing import Optional, List, Tuple
import numpy as np
import tifffile
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from utils.image_processing import combine_fluorescence_channels, get_transform


class FluorescenceDataset(Dataset):
    """
    Dataset for fluorescence microscopy images (C01 + C02 channels).
    """
    def __init__(self, c01_path: str, c02_path: str, image_size: int = 256,
                 normalize_range: Tuple[float, float] = (-1, 1), 
                 is_training: bool = False):
        """
        Args:
            c01_path: Path to C01 channel directory
            c02_path: Path to C02 channel directory
            image_size: Target image size
            normalize_range: Normalization range
            is_training: Whether this is for training (enables augmentation)
        """
        self.c01_path = c01_path
        self.c02_path = c02_path
        self.image_size = image_size
        self.normalize_range = normalize_range
        self.is_training = is_training
        
        # Get all slice files
        self.slice_files = sorted(glob.glob(os.path.join(c01_path, "slice_*.tif")))
        
        # Get transform
        self.transform = get_transform(image_size, normalize_range, is_training)
    
    def __len__(self) -> int:
        return len(self.slice_files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load and return a fluorescence image.
        
        Returns:
            Tensor of shape (3, H, W) in normalized range
        """
        # Get slice filename
        slice_file = self.slice_files[idx]
        slice_name = os.path.basename(slice_file)
        
        # Load C01 and C02 channels
        c01_path = os.path.join(self.c01_path, slice_name)
        c02_path = os.path.join(self.c02_path, slice_name)
        
        c01 = tifffile.imread(c01_path)
        c02 = tifffile.imread(c02_path)
        
        # Combine channels to RGB
        rgb_image = combine_fluorescence_channels(c01, c02)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        return tensor, slice_name


class HEDataset(Dataset):
    """
    Dataset for H&E stained images (target domain).
    If no H&E images are available, this can be used with synthetic or external datasets.
    """
    def __init__(self, he_path: Optional[str], image_size: int = 256,
                 normalize_range: Tuple[float, float] = (-1, 1),
                 is_training: bool = False):
        """
        Args:
            he_path: Path to H&E images directory (None if not available)
            image_size: Target image size
            normalize_range: Normalization range
            is_training: Whether this is for training
        """
        self.he_path = he_path
        self.image_size = image_size
        self.normalize_range = normalize_range
        self.is_training = is_training
        
        # Get H&E image files
        if he_path and os.path.exists(he_path):
            # Support common image formats
            extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']
            self.he_files = []
            for ext in extensions:
                self.he_files.extend(glob.glob(os.path.join(he_path, ext)))
            self.he_files = sorted(self.he_files)
        else:
            self.he_files = []
        
        # Get transform
        self.transform = get_transform(image_size, normalize_range, is_training)
    
    def __len__(self) -> int:
        return len(self.he_files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load and return an H&E image.
        
        Returns:
            Tensor of shape (3, H, W) in normalized range
        """
        he_file = self.he_files[idx]
        
        # Load image
        if he_file.lower().endswith(('.tif', '.tiff')):
            img = tifffile.imread(he_file)
            # Handle grayscale or multi-channel
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=2)
            elif img.ndim == 3 and img.shape[2] > 3:
                img = img[:, :, :3]  # Take first 3 channels
            pil_image = Image.fromarray(img.astype(np.uint8))
        else:
            pil_image = Image.open(he_file).convert('RGB')
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        return tensor


def create_data_loaders(c01_path: str, c02_path: str, he_path: Optional[str],
                       batch_size: int = 1, image_size: int = 256,
                       normalize_range: Tuple[float, float] = (-1, 1),
                       num_workers: int = 4, is_training: bool = True) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create data loaders for fluorescence and H&E datasets.
    
    Args:
        c01_path: Path to C01 channel directory
        c02_path: Path to C02 channel directory
        he_path: Path to H&E images directory (None if not available)
        batch_size: Batch size
        image_size: Target image size
        normalize_range: Normalization range
        num_workers: Number of data loader workers
        is_training: Whether for training
    
    Returns:
        Tuple of (fluorescence_loader, he_loader)
    """
    # Create fluorescence dataset
    fluorescence_dataset = FluorescenceDataset(
        c01_path, c02_path, image_size, normalize_range, is_training
    )
    
    fluorescence_loader = DataLoader(
        fluorescence_dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create H&E dataset (if available)
    he_dataset = HEDataset(he_path, image_size, normalize_range, is_training)
    he_loader = None
    
    if len(he_dataset) > 0:
        he_loader = DataLoader(
            he_dataset,
            batch_size=batch_size,
            shuffle=is_training,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return fluorescence_loader, he_loader

