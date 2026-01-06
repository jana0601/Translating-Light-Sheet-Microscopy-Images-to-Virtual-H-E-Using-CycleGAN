# PowerShell script to download H&E images
# Run this in PowerShell

New-Item -ItemType Directory -Force -Path data/he_stained | Out-Null

Write-Host "Please download H&E images manually and place them in data/he_stained/"
Write-Host "See HE_DATASET_INSTRUCTIONS.md for detailed instructions"

# Example download (uncomment and modify):
# Invoke-WebRequest -Uri "<URL_TO_HE_IMAGE>" -OutFile "data/he_stained/image1.tif"
