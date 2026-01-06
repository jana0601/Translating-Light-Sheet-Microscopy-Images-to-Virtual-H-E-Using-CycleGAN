"""
Test script to verify data loading and preprocessing.
"""
import os
import yaml
from utils.data_loader import create_data_loaders
from utils.image_processing import tensor_to_numpy, save_image
import torch

def test_data_loading():
    """Test data loading functionality."""
    # Load config
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("Testing data loading...")
    
    # Create data loaders
    fluorescence_loader, he_loader = create_data_loaders(
        c01_path=config['data']['c01_path'],
        c02_path=config['data']['c02_path'],
        he_path=config['data']['he_stained_path'],
        batch_size=1,
        image_size=config['image']['image_size'],
        normalize_range=tuple(config['image']['normalize_range']),
        num_workers=0,  # Use 0 for testing
        is_training=False
    )
    
    print(f"Fluorescence dataset size: {len(fluorescence_loader.dataset)}")
    if he_loader:
        print(f"H&E dataset size: {len(he_loader.dataset)}")
    else:
        print("No H&E dataset found (this is OK if you don't have H&E images)")
    
    # Test loading a batch
    print("\nLoading a sample batch...")
    fluorescence_iter = iter(fluorescence_loader)
    sample_tensor, slice_name = next(fluorescence_iter)
    
    print(f"Sample tensor shape: {sample_tensor.shape}")
    print(f"Sample tensor range: [{sample_tensor.min():.3f}, {sample_tensor.max():.3f}]")
    print(f"Sample slice name: {slice_name}")
    
    # Convert to numpy and save as test
    sample_image = tensor_to_numpy(sample_tensor)
    os.makedirs('test_output', exist_ok=True)
    save_image(sample_image, 'test_output/test_sample.png')
    print("\nSample image saved to test_output/test_sample.png")
    
    print("\nData loading test completed successfully!")

if __name__ == '__main__':
    test_data_loading()

