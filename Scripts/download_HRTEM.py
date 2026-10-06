from pathlib import Path
from urllib.parse import quote
from urllib.request import urlretrieve
import argparse

import pandas as pd


# ==================================================
# Project paths
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

HRTEM_DIR = PROJECT_DIR / "Data" / "Raw" / "HRTEM"

METADATA_FILE = PROJECT_DIR / "Data" / "Metadata" / "Dataset_metadata.csv"

IMAGES_DIR = HRTEM_DIR / "images"
LABELS_DIR = HRTEM_DIR / "labels"


# ==================================================
# HRTEM source
# ==================================================

BASE_URL = (
    "https://portal.nersc.gov/project/"
    "m3795/hrtem-generalization/"
)


# ==================================================
# Command-line arguments
# ==================================================

parser = argparse.ArgumentParser(
    description="Download HRTEM nanoparticle images and labels."
)

parser.add_argument(
    "--demo",
    action="store_true",
    help=(
        "Download a deterministic 30-record subset "
        "for the ETL demonstration."
    ),
)

args = parser.parse_args()


# ==================================================
# Check metadata file
# ==================================================

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        f"HRTEM metadata file was not found:\n{METADATA_FILE}"
    )


# ==================================================
# Read metadata
# ==================================================

metadata = pd.read_csv(METADATA_FILE)


# ==================================================
# Select records
# ==================================================

if args.demo:

    # --------------------------------------------------
    # Deterministic demo subset
    #
    # We intentionally include all three materials:
    #
    # Au   = 15 records
    # Ag   = 10 records
    # CdSe = 5 records
    #
    # random_state makes the selection reproducible.
    # Every clean run selects the same records.
    # --------------------------------------------------

    demo_parts = []

    demo_plan = {
        "Au": 15,
        "Ag": 10,
        "CdSe": 5,
    }

    for material, requested_number in demo_plan.items():

        material_records = metadata[
            metadata["Material"] == material
        ]

        if len(material_records) < requested_number:
            raise ValueError(
                f"Not enough {material} records. "
                f"Requested {requested_number}, "
                f"but only {len(material_records)} are available."
            )

        selected_records = material_records.sample(
            n=requested_number,
            random_state=42,
        )

        demo_parts.append(selected_records)

    metadata_to_download = pd.concat(
        demo_parts,
        ignore_index=True,
    )

    # Sort the selected records so the download order
    # is also deterministic and easy to inspect.
    metadata_to_download = metadata_to_download.sort_values(
        by=["Material", "Folder", "File name"]
    ).reset_index(drop=True)

    mode_name = "DEMO MODE"

else:

    metadata_to_download = metadata.copy()

    mode_name = "FULL DATASET MODE"


# ==================================================
# Print summary
# ==================================================

print()
print("HRTEM DOWNLOAD")
print("==============")

print(f"Metadata records available: {len(metadata)}")
print(f"Mode: {mode_name}")
print(f"Records selected: {len(metadata_to_download)}")

print()

print("Selected materials:")

print(
    metadata_to_download["Material"]
    .value_counts()
    .sort_index()
)

print()


# ==================================================
# Create download directories
# ==================================================

IMAGES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LABELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==================================================
# Counters
# ==================================================

images_downloaded = 0
images_existing = 0

labels_downloaded = 0
labels_existing = 0


# ==================================================
# Download selected records
# ==================================================

for position, (_, row) in enumerate(
    metadata_to_download.iterrows(),
    start=1,
):

    # --------------------------------------------------
    # Read metadata
    # --------------------------------------------------

    filename = str(
        row["File name"]
    ).strip()

    folder = str(
        row["Folder"]
    ).strip()

    material = str(
        row["Material"]
    ).strip()


    # --------------------------------------------------
    # Construct label filename
    # --------------------------------------------------

    label_filename = (
        Path(filename).stem
        + "_label.png"
    )


    # --------------------------------------------------
    # Preserve source folder structure locally
    #
    # HRTEM contains duplicate filenames in different
    # acquisition folders. Therefore filename alone
    # cannot uniquely identify a record.
    # --------------------------------------------------

    image_folder = (
        IMAGES_DIR / folder
    )

    label_folder = (
        LABELS_DIR / folder
    )

    image_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_folder.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------
    # Local file paths
    # --------------------------------------------------

    image_path = (
        image_folder / filename
    )

    label_path = (
        label_folder / label_filename
    )


    # --------------------------------------------------
    # Encode remote paths for URL
    # --------------------------------------------------

    encoded_folder = quote(
        folder,
        safe="/",
    )

    encoded_filename = quote(
        filename
    )

    encoded_label_filename = quote(
        label_filename
    )


    # --------------------------------------------------
    # Construct URLs
    # --------------------------------------------------

    image_url = (
        BASE_URL
        + encoded_folder
        + "/"
        + encoded_filename
    )

    label_url = (
        BASE_URL
        + encoded_folder
        + "/Labels/"
        + encoded_label_filename
    )


    # ==================================================
    # Download DM3 image
    # ==================================================

    if image_path.exists():

        print(
            f"[{position}/{len(metadata_to_download)}] "
            f"[{material}] "
            f"Image already exists: {filename}"
        )

        images_existing += 1

    else:

        print(
            f"[{position}/{len(metadata_to_download)}] "
            f"[{material}] "
            f"Downloading image: {filename}"
        )

        # ----------------------------------------------
        # Download to temporary .part file first.
        #
        # Only after the download finishes successfully
        # is it renamed to the real .dm3 filename.
        # ----------------------------------------------

        temp_image_path = image_path.with_suffix(
            image_path.suffix + ".part"
        )

        try:

            urlretrieve(
                image_url,
                temp_image_path,
            )

            temp_image_path.replace(
                image_path
            )

            images_downloaded += 1

        except Exception:

            if temp_image_path.exists():
                temp_image_path.unlink()

            raise


    # ==================================================
    # Download segmentation label
    # ==================================================

    if label_path.exists():

        print(
            f"        Label already exists: "
            f"{label_filename}"
        )

        labels_existing += 1

    else:

        print(
            f"        Downloading label: "
            f"{label_filename}"
        )

        temp_label_path = label_path.with_suffix(
            label_path.suffix + ".part"
        )

        try:

            urlretrieve(
                label_url,
                temp_label_path,
            )

            temp_label_path.replace(
                label_path
            )

            labels_downloaded += 1

        except Exception:

            if temp_label_path.exists():
                temp_label_path.unlink()

            raise


# ==================================================
# Final report
# ==================================================

print()
print("HRTEM DOWNLOAD COMPLETE")
print("=======================")

print(
    f"Records processed:       "
    f"{len(metadata_to_download)}"
)

print(
    f"Images downloaded:       "
    f"{images_downloaded}"
)

print(
    f"Images already present:  "
    f"{images_existing}"
)

print(
    f"Labels downloaded:       "
    f"{labels_downloaded}"
)

print(
    f"Labels already present:  "
    f"{labels_existing}"
)

print()

print("Material counts processed:")

print(
    metadata_to_download["Material"]
    .value_counts()
    .sort_index()
)

print()

if args.demo:

    print(
        "Demo subset successfully processed."
    )

else:

    print(
        "Full HRTEM dataset successfully processed."
    )