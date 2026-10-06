from pathlib import Path
from urllib.request import urlopen, urlretrieve
import json


# ==================================================
# Project paths
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

CO3O4_DIR = (
    PROJECT_DIR
    / "Data"
    / "Raw"
    / "Co3O4"
)


# ==================================================
# Zenodo source
# ==================================================

ZENODO_RECORD_ID = "14927582"

ZENODO_API_URL = (
    f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
)

REQUIRED_FILES = [
    "training_images.h5",
    "training_labels.h5",
]


# ==================================================
# Check whether files already exist
# ==================================================

def co3o4_is_complete():

    for filename in REQUIRED_FILES:

        file_path = CO3O4_DIR / filename

        if not file_path.exists():
            return False

        if file_path.stat().st_size == 0:
            return False

    return True


# ==================================================
# Skip download when files already exist
# ==================================================

print()
print("Co3O4 DOWNLOAD")
print("==============")

if co3o4_is_complete():

    print("Co3O4 dataset already exists.")

    for filename in REQUIRED_FILES:

        file_path = CO3O4_DIR / filename

        size_mb = (
            file_path.stat().st_size
            / (1024 * 1024)
        )

        print(
            f"{filename}: "
            f"{size_mb:.1f} MB"
        )

    print("No download required.")

    raise SystemExit(0)


# ==================================================
# Retrieve official Zenodo metadata
# ==================================================

print(
    "Retrieving file information "
    "from Zenodo..."
)

with urlopen(ZENODO_API_URL) as response:

    metadata = json.load(response)


# ==================================================
# Find download URLs
# ==================================================

download_urls = {}

for file_info in metadata["files"]:

    filename = file_info["key"]

    if filename in REQUIRED_FILES:

        download_urls[filename] = (
            file_info["links"]["self"]
        )


for filename in REQUIRED_FILES:

    if filename not in download_urls:

        raise RuntimeError(
            f"{filename} was not found "
            "in the Zenodo record."
        )


# ==================================================
# Create destination directory
# ==================================================

CO3O4_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==================================================
# Download required files
# ==================================================

for filename in REQUIRED_FILES:

    destination = (
        CO3O4_DIR
        / filename
    )

    if (
        destination.exists()
        and destination.stat().st_size > 0
    ):

        print(
            f"{filename} already exists. "
            "Skipping."
        )

        continue

    temporary_file = (
        CO3O4_DIR
        / f"{filename}.part"
    )

    print()
    print(f"Downloading {filename}...")

    try:

        urlretrieve(
            download_urls[filename],
            temporary_file,
        )

        temporary_file.replace(
            destination
        )

    except Exception:

        if temporary_file.exists():
            temporary_file.unlink()

        raise

    print(
        f"{filename} download complete."
    )


# ==================================================
# Validate result
# ==================================================

if not co3o4_is_complete():

    raise RuntimeError(
        "Co3O4 download completed, "
        "but required files are missing."
    )


# ==================================================
# Final report
# ==================================================

print()
print("Co3O4 DOWNLOAD COMPLETE")
print("=======================")

for filename in REQUIRED_FILES:

    file_path = CO3O4_DIR / filename

    size_mb = (
        file_path.stat().st_size
        / (1024 * 1024)
    )

    print(
        f"{filename}: "
        f"{size_mb:.1f} MB"
    )

print()
print("Required Co3O4 files are available.")