"""
Automated script to download H&E images from publicly accessible sources.
"""
import os
import urllib.request
import json
from pathlib import Path
import time

def download_from_public_repo():
    """
    Attempt to download H&E images from a public repository.
    We'll try to find a directly accessible source.
    """
    he_dir = Path("data/he_stained")
    he_dir.mkdir(parents=True, exist_ok=True)
    
    print("Attempting to download H&E images from public sources...")
    
    # Try to download from a known public dataset
    # Note: Most require registration, but we'll try some alternatives
    
    # Option: Use a GitHub repository with sample images
    # Some repositories host sample H&E images
    
    print("\nSince most H&E datasets require registration, I'll provide")
    print("a practical solution using a smaller, accessible dataset.")
    
    # Try downloading from a public source if available
    # For demonstration, we'll create a script that can be easily modified
    
    return False

def create_kaggle_download_script():
    """Create a script to download from Kaggle (if user has Kaggle API)."""
    kaggle_script = """# Kaggle H&E Dataset Download Script
# Prerequisites: pip install kaggle
# Setup: https://www.kaggle.com/docs/api

# Popular H&E datasets on Kaggle:
# 1. histopathologic-cancer-detection
# 2. colorectal-histology-mnist
# 3. breast-histopathology-images

# Example command:
# kaggle datasets download -d <dataset-name> -p data/he_stained/
# unzip data/he_stained/*.zip -d data/he_stained/
"""
    
    with open("download_kaggle_he.sh", "w") as f:
        f.write(kaggle_script)
    
    print("Created Kaggle download script: download_kaggle_he.sh")

def main():
    """Main download function."""
    print("=" * 70)
    print("Automated H&E Dataset Downloader")
    print("=" * 70)
    
    he_dir = Path("data/he_stained")
    he_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTarget directory: {he_dir.absolute()}")
    
    # Check for existing images
    existing = list(he_dir.glob("*.*"))
    image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
    existing_images = [f for f in existing if f.suffix.lower() in image_extensions]
    
    if existing_images:
        print(f"\n✓ Found {len(existing_images)} existing H&E images!")
        print("You're ready to train!")
        return
    
    print("\nNo H&E images found. Setting up download options...")
    
    # Create instructions
    print("\n" + "=" * 70)
    print("RECOMMENDED: Quick Download Option")
    print("=" * 70)
    print("\nThe easiest way to get H&E images:")
    print("\n1. MHIST Dataset (Recommended for quick start):")
    print("   - Website: https://bmirds.github.io/MHIST/")
    print("   - Size: ~100MB")
    print("   - Format: PNG (224x224)")
    print("   - Steps:")
    print("     a. Visit the website and register")
    print("     b. Download the dataset")
    print("     c. Extract images to: data/he_stained/")
    
    print("\n2. Or use any H&E images you have access to")
    print("   - Place them in: data/he_stained/")
    print("   - Supported: .tif, .tiff, .png, .jpg, .jpeg")
    
    print("\n" + "=" * 70)
    print("IMPORTANT NOTE")
    print("=" * 70)
    print("Most H&E datasets require:")
    print("- Registration on their website")
    print("- Agreement to terms of use")
    print("- Manual download")
    print("\nThis is why we cannot fully automate the download.")
    print("However, once you download images, just place them in")
    print("data/he_stained/ and the training script will use them!")
    
    print("\n" + "=" * 70)
    print("Alternative: Train Without H&E Data")
    print("=" * 70)
    print("You can still train the model without H&E images:")
    print("- The code will use fluorescence images as both domains")
    print("- Results may not be optimal, but it will work")
    print("- Run: python train.py")
    
    # Create a helper script
    helper_script = f"""# Quick helper: Check if H&E images are ready
import os
from pathlib import Path

he_dir = Path("data/he_stained")
if he_dir.exists():
    images = list(he_dir.glob("*.tif")) + list(he_dir.glob("*.tiff")) + \\
             list(he_dir.glob("*.png")) + list(he_dir.glob("*.jpg")) + \\
             list(he_dir.glob("*.jpeg"))
    print(f"Found {{len(images)}} H&E images - Ready to train!")
else:
    print("No H&E images found. See HE_DATASET_INSTRUCTIONS.md")
"""
    
    with open("check_he_images.py", "w") as f:
        f.write(helper_script)
    
    print("\n[OK] Created helper script: check_he_images.py")
    print("  Run it to check if H&E images are ready: python check_he_images.py")

if __name__ == "__main__":
    main()

