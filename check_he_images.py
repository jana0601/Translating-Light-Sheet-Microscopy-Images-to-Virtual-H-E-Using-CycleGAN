# Quick helper: Check if H&E images are ready
import os
from pathlib import Path

he_dir = Path("data/he_stained")
if he_dir.exists():
    images = list(he_dir.glob("*.tif")) + list(he_dir.glob("*.tiff")) + \
             list(he_dir.glob("*.png")) + list(he_dir.glob("*.jpg")) + \
             list(he_dir.glob("*.jpeg"))
    print(f"Found {len(images)} H&E images - Ready to train!")
else:
    print("No H&E images found. See HE_DATASET_INSTRUCTIONS.md")
