"""
Inference script for converting fluorescence images to pseudo H&E stained images.
"""
import os
import yaml
import torch
import glob
import tifffile
from tqdm import tqdm
from PIL import Image

from models.cyclegan import Generator
from utils.image_processing import (
    combine_fluorescence_channels,
    get_transform,
    denormalize_tensor,
    tensor_to_numpy,
    save_image
)


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path: str, device: torch.device, config: dict) -> Generator:
    """
    Load trained generator model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        config: Configuration dictionary
    
    Returns:
        Loaded generator model
    """
    # Create generator
    generator = Generator(
        input_nc=config['model']['input_nc'],
        output_nc=config['model']['output_nc'],
        ngf=config['model']['ngf'],
        n_residual_blocks=config['model']['n_residual_blocks'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load generator weights (G_A2B: fluorescence -> H&E)
    if 'G_A2B_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['G_A2B_state_dict'])
    elif 'model_state_dict' in checkpoint:
        # Alternative checkpoint format
        generator.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Try loading entire checkpoint as state dict
        generator.load_state_dict(checkpoint)
    
    generator.eval()
    
    return generator


def process_single_slice(c01_path: str, c02_path: str, generator: Generator,
                         device: torch.device, config: dict) -> tuple:
    """
    Process a single slice and convert to pseudo H&E.
    
    Args:
        c01_path: Path to C01 channel file
        c02_path: Path to C02 channel file
        generator: Trained generator model
        device: Device to run inference on
        config: Configuration dictionary
    
    Returns:
        Tuple of (pseudo H&E image as numpy array, slice name)
    """
    # Load channels
    c01 = tifffile.imread(c01_path)
    c02 = tifffile.imread(c02_path)
    
    # Combine channels to RGB
    rgb_image = combine_fluorescence_channels(c01, c02)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)
    
    # Get transform (no augmentation for inference)
    transform = get_transform(
        image_size=config['image']['image_size'],
        normalize_range=tuple(config['image']['normalize_range']),
        is_training=False
    )
    
    # Transform to tensor
    tensor = transform(pil_image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Generate pseudo H&E
    with torch.no_grad():
        fake_he_tensor = generator(tensor)
    
    # Denormalize
    fake_he_tensor = denormalize_tensor(
        fake_he_tensor,
        normalize_range=tuple(config['image']['normalize_range'])
    )
    
    # Convert to numpy
    fake_he_image = tensor_to_numpy(fake_he_tensor)
    
    # Get slice name
    slice_name = os.path.basename(c01_path)
    
    return fake_he_image, slice_name


def inference():
    """Main inference function."""
    # Load configuration
    config = load_config()
    
    # Check checkpoint path
    checkpoint_file = config['inference']['checkpoint_file']
    if checkpoint_file is None or not os.path.exists(checkpoint_file):
        # Try to find latest checkpoint
        checkpoint_dir = config['data']['checkpoint_path']
        latest_checkpoint = os.path.join(checkpoint_dir, 'latest.pth') #checkpoint_epoch_60 latest
        if os.path.exists(latest_checkpoint):
            checkpoint_file = latest_checkpoint
            print(f'Using latest checkpoint: {checkpoint_file}')
        else:
            raise FileNotFoundError(
                f'Checkpoint not found. Please specify checkpoint_file in config.yaml or '
                f'ensure {latest_checkpoint} exists.'
            )
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'Loading model from {checkpoint_file}...')
    generator = load_model(checkpoint_file, device, config)
    print('Model loaded successfully.')
    
    # Get input paths
    c01_path = config['data']['c01_path']
    c02_path = config['data']['c02_path']
    output_path = config['data']['output_path']
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get all slice files
    slice_files = sorted(glob.glob(os.path.join(c01_path, "slice_*.tif")))
    
    if len(slice_files) == 0:
        raise FileNotFoundError(f'No slice files found in {c01_path}')
    
    print(f'Found {len(slice_files)} slices to process.')
    
    # Process all slices
    for slice_file in tqdm(slice_files, desc='Processing slices'):
        slice_name = os.path.basename(slice_file)
        c01_file = os.path.join(c01_path, slice_name)
        c02_file = os.path.join(c02_path, slice_name)
        
        # Check if both files exist
        if not os.path.exists(c02_file):
            print(f'Warning: {c02_file} not found. Skipping {slice_name}.')
            continue
        
        try:
            # Process slice
            fake_he_image, _ = process_single_slice(
                c01_file, c02_file, generator, device, config
            )
            
            # Save output
            output_file = os.path.join(output_path, slice_name.replace('.tif', f'.{config["inference"]["output_format"]}'))
            save_image(fake_he_image, output_file, format=config['inference']['output_format'])
            
        except Exception as e:
            print(f'Error processing {slice_name}: {str(e)}')
            continue
    
    print(f'Inference completed! Output saved to {output_path}')


if __name__ == '__main__':
    inference()

