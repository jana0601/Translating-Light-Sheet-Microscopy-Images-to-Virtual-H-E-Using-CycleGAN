# CycleGAN H&E Conversion - Quick Start Guide

## 1. Installation

```bash
pip install -r requirements.txt
```

## 2. Test Data Loading

First, verify that data can be loaded correctly:

```bash
python test_data_loading.py
```

This will verify:
- C01 and C02 channel files can be read correctly
- Image combination and preprocessing work properly
- Output a test image to `test_output/test_sample.png`

## 3. Prepare H&E Dataset

CycleGAN requires target domain (H&E) images for training. Place your H&E images in the following location:

### Location
**Path**: `data/he_stained/` directory


### What Files to Place

**Only image files are needed!** No labels, annotations, or other files required.

**Supported Image Formats:**
- `.tif` or `.tiff` (TIFF format)
- `.png` (PNG format)
- `.jpg` or `.jpeg` (JPEG format)

### Directory Structure

After setup, it should look like:
```
task/
├── data/
│   └── he_stained/
│       ├── image001.png
│       ├── image002.png
│       ├── image003.tif
│       └── ... (all your H&E images)
├── small_dataset/
│   ├── C01/
│   └── C02/
└── ...
```

**Note:** If you don't have H&E data, training can still proceed, but the results may not be ideal. The model will use fluorescence images as both domains (self-supervised learning).


## 4. Train Model

```bash
python train.py
```

Training process:
- Automatically loads data from `small_dataset/C01` and `small_dataset/C02`
- If available, loads H&E data from `data/he_stained/`
- Training progress displayed in terminal
- Checkpoints saved to `checkpoints/` directory
- TensorBoard logs saved to `runs/` directory

**View training progress:**
```bash
tensorboard --logdir runs
```

Then open `http://localhost:6006` in your browser

## 5. Run Inference

After training, edit `config.yaml` to set checkpoint path:

```yaml
inference:
  checkpoint_file: "checkpoints/latest.pth"  # or "checkpoints/checkpoint_epoch_200.pth"
```

Then run:

```bash
python inference.py
```

Output:
- All 256 slices of pseudo H&E images
- Saved in `output/` directory
- Format: `slice_XXXX.png`

