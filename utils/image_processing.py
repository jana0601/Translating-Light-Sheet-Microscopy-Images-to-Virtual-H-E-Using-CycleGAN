"""
Image processing utilities for combining fluorescence channels and H&E conversion.
"""
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from typing import Tuple, Optional


def combine_fluorescence_channels(c01: np.ndarray, c02: np.ndarray) -> np.ndarray:
    """
    Combine C01 (nuclear) and C02 (cytoplasm) grayscale channels into RGB image.
    
    Args:
        c01: TO-PRO-3 nuclear staining channel (grayscale, uint16, shape: H x W)
        c02: Eusion cytoplasm channel (grayscale, uint16, shape: H x W)
    
    Returns:
        RGB image as numpy array (uint8, shape: H x W x 3)
        Color mapping:
        - C01 (nuclear, grayscale) -> Blue channel (will become Hematoxylin blue/purple in H&E output)
        - C02 (cytoplasm, grayscale) -> Green channel (will become Eosin pink/red in H&E output)
    """
    # Normalize each grayscale channel to [0, 1] range
    c01_norm = normalize_channel(c01)
    c02_norm = normalize_channel(c02)
    
    # Map grayscale channels to RGB:
    # C01 (nuclear, grayscale) -> Blue channel (Hematoxylin-like)
    # C02 (cytoplasm, grayscale) -> Green channel (Eosin-like)
    # Red channel: slight cytoplasm contribution
    rgb = np.zeros((c01.shape[0], c01.shape[1], 3), dtype=np.float32)
    rgb[:, :, 0] = c02_norm * 0.3  # Red: slight cytoplasm contribution
    rgb[:, :, 1] = c02_norm  # Green: cytoplasm
    rgb[:, :, 2] = c01_norm  # Blue: nuclear
    
    # Convert to uint8
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    
    return rgb_uint8


def normalize_channel(channel: np.ndarray, percentile: Tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    """
    Normalize a single channel using percentile-based normalization.
    
    Args:
        channel: Input channel (uint16)
        percentile: Percentile range for normalization (min, max)
    
    Returns:
        Normalized channel in [0, 1] range (float32)
    """
    # Calculate percentiles to handle outliers
    p_low, p_high = np.percentile(channel, percentile)
    
    # Clip and normalize
    channel_clipped = np.clip(channel, p_low, p_high)
    if p_high > p_low:
        channel_norm = (channel_clipped - p_low) / (p_high - p_low)
    else:
        channel_norm = np.zeros_like(channel_clipped, dtype=np.float32)
    
    return channel_norm.astype(np.float32)


def get_transform(image_size: int = 256, normalize_range: Tuple[float, float] = (-1, 1), 
                  is_training: bool = False) -> transforms.Compose:
    """
    Get image transformation pipeline.
    
    Args:
        image_size: Target image size
        normalize_range: Normalization range (e.g., [-1, 1] for tanh output)
        is_training: Whether to apply data augmentation
    
    Returns:
        Transform composition
    """
    transform_list = []
    
    # Resize
    transform_list.append(transforms.Resize((image_size, image_size), Image.BILINEAR))
    
    # Data augmentation for training
    if is_training:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transform_list.append(transforms.RandomVerticalFlip(p=0.5))
        transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalize
    if normalize_range == (-1, 1):
        transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    elif normalize_range == (0, 1):
        # Already in [0, 1] after ToTensor
        pass
    else:
        raise ValueError(f"Unsupported normalize_range: {normalize_range}")
    
    return transforms.Compose(transform_list)


def denormalize_tensor(tensor: torch.Tensor, normalize_range: Tuple[float, float] = (-1, 1)) -> torch.Tensor:
    """
    Denormalize tensor from model output range to [0, 1].
    
    Args:
        tensor: Normalized tensor (B, C, H, W)
        normalize_range: Original normalization range
    
    Returns:
        Denormalized tensor in [0, 1] range
    """
    if normalize_range == (-1, 1):
        # Denormalize: (x + 1) / 2
        tensor = (tensor + 1.0) / 2.0
    
    # Clip to [0, 1]
    tensor = torch.clamp(tensor, 0.0, 1.0)
    
    return tensor


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to numpy array (uint8 RGB image).
    
    Args:
        tensor: Tensor in [0, 1] range (B, C, H, W) or (C, H, W)
    
    Returns:
        Numpy array (H, W, 3) as uint8
    """
    # Handle batch dimension
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image
    
    # Convert to numpy and transpose from (C, H, W) to (H, W, C)
    img_np = tensor.detach().cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    
    # Convert to uint8
    img_uint8 = (img_np * 255).astype(np.uint8)
    
    return img_uint8


def save_image(image: np.ndarray, filepath: str, format: str = "png"):
    """
    Save image to file.
    
    Args:
        image: Image array (H, W, 3) as uint8
        filepath: Output file path
        format: Image format (png or tiff)
    """
    if format.lower() == "png":
        Image.fromarray(image).save(filepath)
    elif format.lower() in ["tiff", "tif"]:
        Image.fromarray(image).save(filepath, format="TIFF")
    else:
        raise ValueError(f"Unsupported format: {format}")

