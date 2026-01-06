"""
Visualization utilities for comparing original and converted images.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def visualize_comparison(original: np.ndarray, converted: np.ndarray,
                        save_path: Optional[str] = None, title: str = "Comparison"):
    """
    Visualize side-by-side comparison of original and converted images.
    
    Args:
        original: Original fluorescence image (H, W, 3) as uint8
        converted: Converted H&E image (H, W, 3) as uint8
        save_path: Path to save the visualization (optional)
        title: Title for the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original)
    axes[0].set_title('Original Fluorescence')
    axes[0].axis('off')
    
    axes[1].imshow(converted)
    axes[1].set_title('Pseudo H&E')
    axes[1].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def visualize_channels(c01: np.ndarray, c02: np.ndarray, combined: np.ndarray,
                      save_path: Optional[str] = None):
    """
    Visualize individual channels and combined image.
    
    Args:
        c01: C01 channel (nuclear)
        c02: C02 channel (cytoplasm)
        combined: Combined RGB image
        save_path: Path to save the visualization (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(c01, cmap='gray')
    axes[0].set_title('C01 (TO-PRO-3, Nuclear)')
    axes[0].axis('off')
    
    axes[1].imshow(c02, cmap='gray')
    axes[1].set_title('C02 (Eusion, Cytoplasm)')
    axes[1].axis('off')
    
    axes[2].imshow(combined)
    axes[2].set_title('Combined RGB')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()

