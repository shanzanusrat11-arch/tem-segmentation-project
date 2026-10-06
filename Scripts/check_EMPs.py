from pathlib import Path

import pandas as pd


# Location of the EMPS dataset
DATA_DIR = Path("data/raw/emps")

metadata_file = DATA_DIR / "metadata.csv"
train_file = DATA_DIR / "train.csv"
test_file = DATA_DIR / "test.csv"
images_dir = DATA_DIR / "images"
segmaps_dir = DATA_DIR / "segmaps"


# Read the metadata
metadata = pd.read_csv(metadata_file)

# EMPS train/test files contain IDs without a header
train_ids = pd.read_csv(train_file, header=None)[0].astype(str)
test_ids = pd.read_csv(test_file, header=None)[0].astype(str)


print("EMPS DATASET CHECK")
print("------------------")

print(f"Metadata records: {len(metadata)}")
print(f"Training IDs:     {len(train_ids)}")
print(f"Test IDs:         {len(test_ids)}")

print(f"Image files:      {len(list(images_dir.glob('*.png')))}")
print(f"Segmentation maps:{len(list(segmaps_dir.glob('*.png')))}")

print()
print("Metadata columns:")
print(metadata.columns.tolist())

print()
print("First five metadata records:")
print(metadata.head())