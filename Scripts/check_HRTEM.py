from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

HRTEM_DIR = PROJECT_DIR / "Data" / "Raw" / "HRTEM"

METADATA_FILE = HRTEM_DIR / "Dataset_metadata.csv"


# --------------------------------------------------
# Read HRTEM metadata
# --------------------------------------------------

metadata = pd.read_csv(METADATA_FILE)


# --------------------------------------------------
# Basic dataset checks
# --------------------------------------------------

print()
print("HRTEM DATASET CHECK")
print("-------------------")

print(f"Metadata records: {len(metadata)}")

print()
print("Metadata columns:")

for column in metadata.columns:
    print(f"  {column}")


print()
print("First five records:")
print(metadata.head())


print()
print("Materials:")
print(metadata["Material"].value_counts(dropna=False))


print()
print("Nominal nanoparticle sizes:")
print(
    metadata["Nanoparticle Size (nm)"]
    .value_counts(dropna=False)
    .sort_index()
)


print()
print("Pixel scales (nm/pixel):")
print(
    metadata["Pixel Scale (nm)"]
    .value_counts(dropna=False)
    .sort_index()
)


print()
print("Number of unique file names:")
print(metadata["File name"].nunique())


print()
print("Number of metadata records:")
print(len(metadata))


print()
print("Duplicate file names:")
duplicate_names = metadata[
    metadata["File name"].duplicated(keep=False)
]

print(
    duplicate_names[
        ["File name", "Folder"]
    ].sort_values("File name")
)