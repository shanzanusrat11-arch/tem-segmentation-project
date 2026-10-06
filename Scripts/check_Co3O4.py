from pathlib import Path

import h5py
import numpy as np


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

CO3O4_DIR = PROJECT_DIR / "Data" / "Raw" / "Co3O4"

IMAGES_FILE = CO3O4_DIR / "training_images.h5"
LABELS_FILE = CO3O4_DIR / "training_labels.h5"


# --------------------------------------------------
# Check that the files exist
# --------------------------------------------------

print()
print("Co3O4 DATASET CHECK")
print("-------------------")

print(f"Images file: {IMAGES_FILE}")
print(f"Labels file: {LABELS_FILE}")
print()

if not IMAGES_FILE.exists():
    raise FileNotFoundError(
        f"Could not find:\n{IMAGES_FILE}"
    )

if not LABELS_FILE.exists():
    raise FileNotFoundError(
        f"Could not find:\n{LABELS_FILE}"
    )

print("Both HDF5 files were found.")
print()


# --------------------------------------------------
# Open the HDF5 files
# --------------------------------------------------

with h5py.File(IMAGES_FILE, "r") as image_file, \
     h5py.File(LABELS_FILE, "r") as label_file:

    print("HDF5 contents")
    print("-------------")

    print("Image file keys:")
    print(list(image_file.keys()))

    print()

    print("Label file keys:")
    print(list(label_file.keys()))

    print()


    # --------------------------------------------------
    # Check for the expected datasets
    # --------------------------------------------------

    if "images" not in image_file:
        raise KeyError(
            "Could not find an 'images' dataset "
            "inside training_images.h5"
        )

    if "labels" not in label_file:
        raise KeyError(
            "Could not find a 'labels' dataset "
            "inside training_labels.h5"
        )


    images = image_file["images"]
    labels = label_file["labels"]


    # --------------------------------------------------
    # Display basic information
    # --------------------------------------------------

    print("Dataset information")
    print("-------------------")

    print(f"Images shape: {images.shape}")
    print(f"Images dtype: {images.dtype}")

    print()

    print(f"Labels shape: {labels.shape}")
    print(f"Labels dtype: {labels.dtype}")

    print()


    # --------------------------------------------------
    # Check number of samples
    # --------------------------------------------------

    number_of_images = images.shape[0]
    number_of_labels = labels.shape[0]

    print(f"Number of images: {number_of_images}")
    print(f"Number of labels: {number_of_labels}")

    if number_of_images != number_of_labels:
        raise ValueError(
            "The number of images and labels does not match."
        )

    print("Image and label counts match.")
    print()


    # --------------------------------------------------
    # Examine only the first sample
    #
    # We deliberately do NOT load the entire HDF5
    # dataset into RAM.
    # --------------------------------------------------

    first_image = images[0]
    first_label = labels[0]

    print("First sample")
    print("------------")

    print(f"First image shape: {first_image.shape}")
    print(f"First image dtype: {first_image.dtype}")
    print(f"First image min: {np.min(first_image)}")
    print(f"First image max: {np.max(first_image)}")

    print()

    print(f"First label shape: {first_label.shape}")
    print(f"First label dtype: {first_label.dtype}")
    print(f"First label min: {np.min(first_label)}")
    print(f"First label max: {np.max(first_label)}")

    print()


    # --------------------------------------------------
    # Check label channels
    # --------------------------------------------------

    if first_label.ndim == 3:

        print(
            f"Number of label channels: "
            f"{first_label.shape[-1]}"
        )

        for channel in range(first_label.shape[-1]):

            unique_values = np.unique(
                first_label[..., channel]
            )

            print(
                f"Unique values in label channel "
                f"{channel}: {unique_values}"
            )

    else:

        unique_values = np.unique(first_label)

        print(
            f"Unique label values: {unique_values}"
        )


# --------------------------------------------------
# Final result
# --------------------------------------------------

print()
print("Co3O4 dataset check complete.")
print("The HDF5 files can be read successfully.")