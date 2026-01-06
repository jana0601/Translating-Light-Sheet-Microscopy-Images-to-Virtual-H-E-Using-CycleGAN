"""
Download sample H&E images from publicly accessible sources.
This script attempts to download a small sample of H&E images for quick testing.
"""
import os
import urllib.request
import json
from pathlib import Path
import time

def download_sample_from_github():
    """
    Try to download sample H&E images from GitHub repositories that host sample data.
    """
    he_dir = Path("data/he_stained")
    he_dir.mkdir(parents=True, exist_ok=True)
    
    print("Attempting to download sample H&E images...")
    print("Note: This may take a few minutes depending on your internet connection.")
    
    # Some GitHub repositories with sample H&E images
    # Note: These are examples - actual repositories may vary
    sample_repos = [
        # You can add GitHub raw content URLs here if available
    ]
    
    # For now, we'll create a more practical approach
    print("\nSince most H&E datasets require registration, here's a practical solution:")
    return False

def create_sample_download_script():
    """Create a script that uses wget or curl to download from specific URLs."""
    script_content = """#!/bin/bash
# Script to download sample H&E images
# Run this script to download H&E images from public sources

mkdir -p data/he_stained

# Option 1: Download from TCGA (requires authentication)
# Visit https://portal.gdc.cancer.gov/ and download manually

# Option 2: Download from a public repository (if available)
# Uncomment and modify the URLs below:

# Example (replace with actual URLs):
# wget -P data/he_stained/ <URL_TO_HE_IMAGE_1>
# wget -P data/he_stained/ <URL_TO_HE_IMAGE_2>

echo "Please download H&E images manually and place them in data/he_stained/"
echo "See HE_DATASET_INSTRUCTIONS.md for detailed instructions"
"""
    
    with open("download_he.sh", "w") as f:
        f.write(script_content)
    
    # Also create a PowerShell version for Windows
    ps_script = """# PowerShell script to download H&E images
# Run this in PowerShell

New-Item -ItemType Directory -Force -Path data/he_stained | Out-Null

Write-Host "Please download H&E images manually and place them in data/he_stained/"
Write-Host "See HE_DATASET_INSTRUCTIONS.md for detailed instructions"

# Example download (uncomment and modify):
# Invoke-WebRequest -Uri "<URL_TO_HE_IMAGE>" -OutFile "data/he_stained/image1.tif"
"""
    
    with open("download_he.ps1", "w") as f:
        f.write(ps_script)
    
    print("Created download scripts: download_he.sh and download_he.ps1")

def main():
    """Main function."""
    print("=" * 60)
    print("H&E Sample Image Downloader")
    print("=" * 60)
    
    he_dir = Path("data/he_stained")
    he_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTarget directory: {he_dir.absolute()}")
    
    # Check if directory already has images
    existing_images = list(he_dir.glob("*.tif")) + list(he_dir.glob("*.tiff")) + \
                     list(he_dir.glob("*.png")) + list(he_dir.glob("*.jpg")) + \
                     list(he_dir.glob("*.jpeg"))
    
    if existing_images:
        print(f"\nFound {len(existing_images)} existing H&E images in data/he_stained/")
        print("You can proceed with training!")
        return
    
    print("\nNo H&E images found in data/he_stained/")
    print("\nRecommended approach:")
    print("1. Visit one of these sources:")
    print("   - TCGA: https://portal.gdc.cancer.gov/")
    print("   - Camelyon: https://camelyon17.grand-challenge.org/")
    print("   - MHIST: https://bmirds.github.io/MHIST/")
    print("\n2. Download at least 50-100 H&E images")
    print("3. Place them in: data/he_stained/")
    print("\n4. Supported formats: .tif, .tiff, .png, .jpg, .jpeg")
    
    create_sample_download_script()
    
    print("\n" + "=" * 60)
    print("Alternative: Use synthetic H&E images")
    print("=" * 60)
    print("If you cannot download real H&E images, you can:")
    print("1. Train without H&E data (code will use fluorescence as both domains)")
    print("2. Use color transfer techniques (not implemented in this project)")
    print("3. Generate synthetic H&E-like images from your fluorescence data")
    
    print("\nFor now, the training script will work without H&E images,")
    print("but results may not be optimal.")

if __name__ == "__main__":
    main()

