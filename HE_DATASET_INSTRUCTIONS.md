
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
