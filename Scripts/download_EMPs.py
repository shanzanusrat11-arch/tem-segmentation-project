from pathlib import Path
from urllib.request import urlretrieve
import shutil
import tempfile
import zipfile


# ==================================================
# Project paths
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

EMPS_DIR = (
    PROJECT_DIR
    / "Data"
    / "Raw"
    / "EMPs"
)

DOWNLOAD_URL = (
    "https://github.com/by256/emps/"
    "archive/refs/heads/main.zip"
)


# ==================================================
# Files/directories expected from EMPS
# ==================================================

REQUIRED_ITEMS = [
    "images",
    "segmaps",
    "metadata.csv",
    "train.csv",
    "test.csv",
    "viadata.json",
]


# ==================================================
# Check whether EMPS is already available
# ==================================================

def emps_is_complete():

    if not EMPS_DIR.exists():
        return False

    for item in REQUIRED_ITEMS:

        if not (EMPS_DIR / item).exists():
            return False

    image_count = len(
        list(
            (EMPS_DIR / "images").glob("*.png")
        )
    )

    segmap_count = len(
        list(
            (EMPS_DIR / "segmaps").glob("*.png")
        )
    )

    return (
        image_count == 465
        and segmap_count == 465
    )


# ==================================================
# Skip download if dataset already exists
# ==================================================

if emps_is_complete():

    print()
    print("EMPS DOWNLOAD")
    print("=============")
    print("EMPS dataset already exists.")
    print("Images: 465")
    print("Segmentation maps: 465")
    print("No download required.")

    raise SystemExit(0)


# ==================================================
# Download EMPS
# ==================================================

print()
print("EMPS DOWNLOAD")
print("=============")
print("EMPS dataset not found.")
print("Downloading official EMPS dataset...")


with tempfile.TemporaryDirectory() as temp_directory:

    temp_directory = Path(temp_directory)

    zip_path = (
        temp_directory
        / "emps.zip"
    )

    # ----------------------------------------------
    # Download GitHub archive
    # ----------------------------------------------

    urlretrieve(
        DOWNLOAD_URL,
        zip_path,
    )

    print("Download complete.")

    # ----------------------------------------------
    # Extract archive
    # ----------------------------------------------

    print("Extracting EMPS dataset...")

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as zip_file:

        zip_file.extractall(
            temp_directory
        )

    extracted_directory = (
        temp_directory
        / "emps-main"
    )

    if not extracted_directory.exists():

        raise RuntimeError(
            "Expected emps-main directory "
            "was not found after extraction."
        )

    # ----------------------------------------------
    # Create destination
    # ----------------------------------------------

    EMPS_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if EMPS_DIR.exists():

        shutil.rmtree(
            EMPS_DIR
        )

    # ----------------------------------------------
    # Copy dataset into Data/Raw/EMPs
    # ----------------------------------------------

    shutil.copytree(
        extracted_directory,
        EMPS_DIR,
    )


# ==================================================
# Validate downloaded dataset
# ==================================================

if not emps_is_complete():

    raise RuntimeError(
        "EMPS download completed, but the "
        "dataset failed validation."
    )


# ==================================================
# Final report
# ==================================================

image_count = len(
    list(
        (EMPS_DIR / "images").glob("*.png")
    )
)

segmap_count = len(
    list(
        (EMPS_DIR / "segmaps").glob("*.png")
    )
)


print()
print("EMPS DOWNLOAD COMPLETE")
print("======================")
print(f"Images:             {image_count}")
print(f"Segmentation maps:  {segmap_count}")
print(f"Location:           {EMPS_DIR}")