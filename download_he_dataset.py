"""
Script to download public H&E histopathology image datasets.
"""
import os
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import shutil

def download_file(url: str, dest_path: str, desc: str = "Downloading"):
    """Download a file with progress indication."""
    print(f"{desc} from {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=lambda blocknum, blocksize, totalsize: 
            print(f"\rProgress: {min(100, (blocknum * blocksize / totalsize) * 100):.1f}%", end='') if totalsize > 0 else None)
        print("\nDownload completed!")
        return True
    except Exception as e:
        print(f"\nError downloading: {e}")
        return False

def extract_archive(archive_path: str, extract_to: str):
    """Extract zip or tar archive."""
    print(f"Extracting {archive_path} to {extract_to}...")
    try:
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        elif archive_path.endswith('.tar'):
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(extract_to)
        print("Extraction completed!")
        return True
    except Exception as e:
        print(f"Error extracting: {e}")
        return False

def download_mhist_sample():
    """
    Download MHIST dataset sample.
    MHIST: 3,152 H&E stained colorectal polyp images (224x224 pixels)
    Note: Full dataset requires registration, this is a sample approach.
    """
    print("\n=== MHIST Dataset ===")
    print("MHIST contains 3,152 H&E stained colorectal polyp images (224x224)")
    print("Full dataset: https://bmirds.github.io/MHIST/")
    print("\nNote: Full MHIST dataset requires registration.")
    print("For now, we'll create a script to help you download it manually.")
    
    return False

def download_histoseg_sample():
    """
    Download Histo-Seg dataset sample.
    Histo-Seg: 38 whole slide images with annotations
    """
    print("\n=== Histo-Seg Dataset ===")
    print("Histo-Seg contains 38 chemically stained whole slide images")
    print("Dataset: https://data.mendeley.com/datasets/vccj8mp2cg/1")
    print("\nNote: This dataset requires Mendeley registration.")
    print("We'll provide instructions for manual download.")
    
    return False

def download_sample_he_images():
    """
    Download a small sample of H&E images from public sources.
    This uses a smaller, more accessible dataset for quick setup.
    """
    print("\n=== Downloading Sample H&E Images ===")
    
    # Create target directory
    he_dir = Path("data/he_stained")
    he_dir.mkdir(parents=True, exist_ok=True)
    
    # Option 1: Try to download from a public repository
    # Note: We'll use a GitHub repository or similar that hosts sample H&E images
    print("\nAttempting to download sample H&E images...")
    
    # For demonstration, we'll create a script that downloads from a known source
    # Since direct downloads may not always work, we'll provide multiple options
    
    sample_urls = [
        # These are example URLs - actual URLs may vary
        # You may need to find a publicly accessible H&E image repository
    ]
    
    print("\nSince direct download URLs may require authentication or may change,")
    print("we recommend the following approach:")
    print("\n1. Download from TCGA (The Cancer Genome Atlas):")
    print("   - Visit: https://portal.gdc.cancer.gov/")
    print("   - Search for H&E stained images")
    print("   - Download and place in data/he_stained/")
    
    print("\n2. Download from Camelyon16/17:")
    print("   - Visit: https://camelyon17.grand-challenge.org/")
    print("   - Register and download H&E stained whole slide images")
    
    print("\n3. Use MHIST dataset:")
    print("   - Visit: https://bmirds.github.io/MHIST/")
    print("   - Register and download")
    print("   - Extract images to data/he_stained/")
    
    return False

def create_download_instructions():
    """Create a detailed instruction file for downloading H&E datasets."""
    instructions = """
# H&E Dataset Download Instructions

## Recommended Datasets

### 1. TCGA (The Cancer Genome Atlas) - Recommended
- **Website**: https://portal.gdc.cancer.gov/
- **Size**: Large (GB to TB scale)
- **Format**: Various formats, mostly TIFF
- **Steps**:
  1. Visit the GDC Data Portal
  2. Search for "H&E" or "hematoxylin eosin"
  3. Filter by image type
  4. Download selected images
  5. Place images in `data/he_stained/` directory

### 2. Camelyon16/17
- **Website**: https://camelyon17.grand-challenge.org/
- **Size**: Large (several GB)
- **Format**: TIFF whole slide images
- **Steps**:
  1. Register on the website
  2. Download training/test sets
  3. Extract and place in `data/he_stained/`

### 3. MHIST (Colorectal Polyp Dataset)
- **Website**: https://bmirds.github.io/MHIST/
- **Size**: Medium (~100MB for images)
- **Format**: PNG images (224x224)
- **Steps**:
  1. Register and download
  2. Extract images
  3. Place in `data/he_stained/`

### 4. Histo-Seg
- **Website**: https://data.mendeley.com/datasets/vccj8mp2cg/1
- **Size**: Medium
- **Format**: Whole slide images
- **Steps**:
  1. Register on Mendeley
  2. Download dataset
  3. Extract and place in `data/he_stained/`

## Quick Start (Using Pre-processed Samples)

If you have access to any H&E images:
1. Place them in `data/he_stained/` directory
2. Supported formats: .tif, .tiff, .png, .jpg, .jpeg
3. The training script will automatically detect and use them

## Minimum Requirements

For CycleGAN training, you need at least:
- 50-100 H&E images (more is better)
- Images should be RGB format
- Recommended size: 256x256 or larger (will be resized automatically)

## Notes

- CycleGAN works with unpaired data, so H&E images don't need to match your fluorescence images
- Any H&E stained histopathology images will work
- More diverse H&E images = better training results
"""
    
    with open("HE_DATASET_INSTRUCTIONS.md", "w", encoding="utf-8") as f:
        f.write(instructions)
    
    print("\nCreated HE_DATASET_INSTRUCTIONS.md with detailed download instructions!")
    return True

def download_from_kaggle():
    """
    Attempt to download from Kaggle (requires kaggle API).
    Many H&E datasets are available on Kaggle.
    """
    print("\n=== Kaggle Datasets ===")
    print("Many H&E histopathology datasets are available on Kaggle.")
    print("\nTo download from Kaggle:")
    print("1. Install kaggle: pip install kaggle")
    print("2. Set up Kaggle API credentials (see https://www.kaggle.com/docs/api)")
    print("3. Search for 'H&E' or 'histopathology' datasets")
    print("4. Download using: kaggle datasets download <dataset-name>")
    print("5. Extract and place in data/he_stained/")
    
    return False

def main():
    """Main function to download H&E dataset."""
    print("=" * 60)
    print("H&E Dataset Downloader")
    print("=" * 60)
    
    # Create target directory
    he_dir = Path("data/he_stained")
    he_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nTarget directory: {he_dir.absolute()}")
    
    print("\nAvailable options:")
    print("1. Create download instructions (recommended)")
    print("2. Attempt direct download (may not work for all datasets)")
    print("3. Show dataset recommendations")
    
    # For now, create instructions file
    create_download_instructions()
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Check HE_DATASET_INSTRUCTIONS.md for detailed instructions")
    print("2. Download H&E images from one of the recommended sources")
    print("3. Place images in: data/he_stained/")
    print("4. Run training: python train.py")
    print("\nNote: CycleGAN can work with as few as 50-100 H&E images,")
    print("      but more images will give better results.")

if __name__ == "__main__":
    main()

