#!/bin/bash
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
