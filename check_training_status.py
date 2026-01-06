"""
Check training status and progress.
"""
import os
from pathlib import Path
import glob

def check_training_status():
    """Check if training is running and show progress."""
    print("=" * 60)
    print("Training Status Check")
    print("=" * 60)
    
    # Check for checkpoints
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        if checkpoints:
            print(f"\n[OK] Found {len(checkpoints)} checkpoint(s):")
            for cp in sorted(checkpoints):
                size_mb = cp.stat().st_size / (1024 * 1024)
                print(f"  - {cp.name} ({size_mb:.2f} MB)")
        else:
            print("\n⚠ No checkpoints found yet (training may just have started)")
    else:
        print("\n⚠ Checkpoints directory not found")
    
    # Check for TensorBoard logs
    runs_dir = Path("runs")
    if runs_dir.exists():
        log_dirs = list(runs_dir.iterdir())
        if log_dirs:
            print(f"\n[OK] TensorBoard logs found in: runs/")
            print("  View progress with: tensorboard --logdir runs")
    
    # Check data
    print("\n" + "=" * 60)
    print("Data Status")
    print("=" * 60)
    
    # Fluorescence data
    c01_files = len(list(Path("small_dataset/C01").glob("*.tif")))
    c02_files = len(list(Path("small_dataset/C02").glob("*.tif")))
    print(f"Fluorescence images: {c01_files} slices (C01 + C02)")
    
    # H&E data
    he_dir = Path("data/he_stained")
    if he_dir.exists():
        he_images = list(he_dir.glob("*.png")) + list(he_dir.glob("*.tif")) + \
                   list(he_dir.glob("*.tiff")) + list(he_dir.glob("*.jpg")) + \
                   list(he_dir.glob("*.jpeg"))
        print(f"H&E images: {len(he_images)} images")
    
    print("\n" + "=" * 60)
    print("Training Tips")
    print("=" * 60)
    print("1. Training is running in the background")
    print("2. Check progress with: tensorboard --logdir runs")
    print("3. Checkpoints are saved in: checkpoints/")
    print("4. Training may take several hours (especially on CPU)")
    print("5. You can stop training with Ctrl+C if needed")

if __name__ == "__main__":
    check_training_status()

