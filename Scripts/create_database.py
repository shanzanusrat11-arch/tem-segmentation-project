from pathlib import Path
import sqlite3

import h5py
import pandas as pd
from PIL import Image


# ==================================================
# Project paths
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "Data"
RAW_DIR = DATA_DIR / "Raw"
METADATA_DIR = DATA_DIR / "Metadata"

DATABASE_DIR = DATA_DIR / "Database"
DATABASE_FILE = DATABASE_DIR / "microscopy.db"

EMPS_DIR = RAW_DIR / "EMPs"
HRTEM_DIR = RAW_DIR / "HRTEM"
CO3O4_DIR = RAW_DIR / "Co3O4"

DATABASE_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# Helper functions
# ==================================================

def relative_path(path):
    """
    Store paths relative to the project directory.

    This avoids storing machine-specific paths such as:
    /Users/shanzanusrat/Documents/...
    """
    return path.relative_to(PROJECT_DIR).as_posix()


def safe_float(value):
    """
    Convert a value to float when possible.
    Return None for missing or invalid values.
    """
    if pd.isna(value):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ==================================================
# Start database
# ==================================================

print()
print("MICROSCOPY DATABASE ETL")
print("=======================")
print()

print("Creating database:")
print(DATABASE_FILE)
print()


# Delete the old generated database.
#
# This makes the build reproducible:
# every run reconstructs the database from source data.
if DATABASE_FILE.exists():
    DATABASE_FILE.unlink()


connection = sqlite3.connect(DATABASE_FILE)
cursor = connection.cursor()


# ==================================================
# Create images table
# ==================================================

cursor.execute(
    """
    CREATE TABLE images (

        image_id TEXT PRIMARY KEY,

        source_dataset TEXT NOT NULL,

        original_filename TEXT,

        image_path TEXT,

        label_path TEXT,

        width_px INTEGER,

        height_px INTEGER,

        image_format TEXT,

        label_format TEXT,

        label_type TEXT,

        material TEXT,

        nominal_particle_size_nm REAL,

        particle_shape TEXT,

        support_material TEXT,

        instrument TEXT,

        dosage_e_per_a2 REAL,

        pixel_size_nm REAL,

        pixel_size_source TEXT,

        source_folder TEXT,

        split TEXT

    );
    """
)


# ==================================================
# SQL insert statement
# ==================================================

INSERT_SQL = """
INSERT INTO images (

    image_id,
    source_dataset,
    original_filename,
    image_path,
    label_path,
    width_px,
    height_px,
    image_format,
    label_format,
    label_type,
    material,
    nominal_particle_size_nm,
    particle_shape,
    support_material,
    instrument,
    dosage_e_per_a2,
    pixel_size_nm,
    pixel_size_source,
    source_folder,
    split

)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


# ==================================================
# Load EMPS
# ==================================================

print("Loading EMPS...")
print("---------------")


emps_metadata_file = EMPS_DIR / "metadata.csv"
emps_train_file = EMPS_DIR / "train.csv"
emps_test_file = EMPS_DIR / "test.csv"

emps_images_dir = EMPS_DIR / "images"
emps_labels_dir = EMPS_DIR / "segmaps"


required_emps_files = [
    emps_metadata_file,
    emps_train_file,
    emps_test_file,
]

for required_file in required_emps_files:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required EMPS file was not found:\n"
            f"{required_file}"
        )


emps_metadata = pd.read_csv(emps_metadata_file)

train_data = pd.read_csv(
    emps_train_file,
    header=None,
)

test_data = pd.read_csv(
    emps_test_file,
    header=None,
)


def normalize_emps_id(value):

    return Path(
        str(value).strip()
    ).stem


train_ids = {
    normalize_emps_id(value)
    for value in train_data.iloc[:, 0]
}

test_ids = {
    normalize_emps_id(value)
    for value in test_data.iloc[:, 0]
}


# Make sure train and test do not overlap.
overlap = train_ids.intersection(test_ids)

if overlap:

    raise ValueError(
        "EMPS train/test files contain overlapping IDs."
    )


emps_loaded = 0


for _, row in emps_metadata.iterrows():

    filename = str(
        row["filename"]
    ).strip()

    stem = Path(filename).stem

    image_path = (
        emps_images_dir / filename
    )

    label_path = (
        emps_labels_dir / filename
    )


    if not image_path.exists():

        raise FileNotFoundError(
            f"EMPS image not found:\n"
            f"{image_path}"
        )


    if not label_path.exists():

        raise FileNotFoundError(
            f"EMPS label not found:\n"
            f"{label_path}"
        )


    # ----------------------------------------------
    # Determine supplied EMPS split
    # ----------------------------------------------

    if stem in train_ids:

        split = "train"

    elif stem in test_ids:

        split = "test"

    else:

        raise ValueError(
            f"EMPS record is not present in "
            f"train.csv or test.csv:\n{filename}"
        )


    # ----------------------------------------------
    # Image dimensions
    # ----------------------------------------------

    with Image.open(image_path) as image:

        width_px, height_px = image.size


    with Image.open(label_path) as label:

        label_width, label_height = label.size


    if (
        width_px != label_width
        or height_px != label_height
    ):

        raise ValueError(
            f"EMPS image/label dimensions do not match:\n"
            f"{filename}"
        )


    # ----------------------------------------------
    # Insert EMPS record
    # ----------------------------------------------

    cursor.execute(
        INSERT_SQL,
        (
            f"EMPS:{stem}",
            "EMPS",
            filename,
            relative_path(image_path),
            relative_path(label_path),
            width_px,
            height_px,
            image_path.suffix.lower().lstrip("."),
            label_path.suffix.lower().lstrip("."),
            "instance",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "scale_bar_in_image",
            relative_path(EMPS_DIR),
            split,
        ),
    )

    emps_loaded += 1


print(f"EMPS records loaded: {emps_loaded}")
print()


# ==================================================
# Load HRTEM
# ==================================================

print("Loading HRTEM demo subset...")
print("----------------------------")


hrtem_metadata_file = (
    METADATA_DIR / "Dataset_metadata.csv"
)

hrtem_images_dir = (
    HRTEM_DIR / "images"
)

hrtem_labels_dir = (
    HRTEM_DIR / "labels"
)


if not hrtem_metadata_file.exists():

    raise FileNotFoundError(
        f"HRTEM metadata file was not found:\n"
        f"{hrtem_metadata_file}"
    )


hrtem_metadata = pd.read_csv(
    hrtem_metadata_file
)


# ==================================================
# Reproduce EXACTLY the same deterministic
# demo selection used by download_HRTEM.py
#
# Au   = 15
# Ag   = 10
# CdSe = 5
#
# Total = 30
#
# random_state=42 means every clean run selects
# the same records.
# ==================================================

demo_plan = {

    "Au": 15,
    "Ag": 10,
    "CdSe": 5,

}


demo_parts = []


for material, requested_number in demo_plan.items():

    material_records = hrtem_metadata[
        hrtem_metadata["Material"] == material
    ]


    if len(material_records) < requested_number:

        raise ValueError(
            f"Not enough HRTEM {material} records. "
            f"Requested {requested_number}, "
            f"but only {len(material_records)} exist."
        )


    selected_records = material_records.sample(
        n=requested_number,
        random_state=42,
    )


    demo_parts.append(
        selected_records
    )


hrtem_demo = pd.concat(
    demo_parts,
    ignore_index=True,
)


# Same deterministic ordering as download_HRTEM.py
hrtem_demo = hrtem_demo.sort_values(
    by=[
        "Material",
        "Folder",
        "File name",
    ]
).reset_index(drop=True)


# Safety check
if len(hrtem_demo) != 30:

    raise ValueError(
        "HRTEM demo selection should contain "
        "exactly 30 records."
    )


hrtem_loaded = 0


for _, row in hrtem_demo.iterrows():

    filename = str(
        row["File name"]
    ).strip()

    folder = str(
        row["Folder"]
    ).strip()

    material = str(
        row["Material"]
    ).strip()


    label_filename = (
        Path(filename).stem
        + "_label.png"
    )


    # ----------------------------------------------
    # Local paths
    #
    # Preserve acquisition folder because HRTEM
    # contains duplicate filenames.
    # ----------------------------------------------

    image_path = (
        hrtem_images_dir
        / folder
        / filename
    )

    label_path = (
        hrtem_labels_dir
        / folder
        / label_filename
    )


    # ----------------------------------------------
    # Verify demo files actually exist
    # ----------------------------------------------

    if not image_path.exists():

        raise FileNotFoundError(
            "A selected HRTEM demo image is missing.\n"
            "Run:\n\n"
            "python Scripts/download_HRTEM.py --demo\n\n"
            f"Missing file:\n{image_path}"
        )


    if not label_path.exists():

        raise FileNotFoundError(
            "A selected HRTEM demo label is missing.\n"
            "Run:\n\n"
            "python Scripts/download_HRTEM.py --demo\n\n"
            f"Missing file:\n{label_path}"
        )


    # ----------------------------------------------
    # Read dimensions from PNG label.
    #
    # DM3 files are not opened here because the
    # relevant physical metadata is already provided
    # by Dataset_metadata.csv.
    # ----------------------------------------------

    with Image.open(label_path) as label:

        width_px, height_px = label.size


    # ----------------------------------------------
    # Metadata
    # ----------------------------------------------

    nominal_particle_size_nm = safe_float(
        row.get("Nanoparticle Size (nm)")
    )

    particle_shape = row.get(
        "Nanoparticle Shape"
    )

    support_material = row.get(
        "Support"
    )

    instrument = row.get(
        "Instrument"
    )

    dosage = safe_float(
        row.get("Dosage (e/A2)")
    )

    pixel_size_nm = safe_float(
        row.get("Pixel Scale (nm)")
    )


    # Clean possible pandas NaN strings
    if pd.isna(particle_shape):
        particle_shape = None

    if pd.isna(support_material):
        support_material = None

    if pd.isna(instrument):
        instrument = None


    # ----------------------------------------------
    # Unique ID
    #
    # Folder is included because HRTEM has duplicate
    # filenames in different acquisition folders.
    # ----------------------------------------------

    image_id = (
        "HRTEM:"
        + folder
        + "/"
        + filename
    )


    cursor.execute(
        INSERT_SQL,
        (
            image_id,
            "HRTEM",
            filename,
            relative_path(image_path),
            relative_path(label_path),
            width_px,
            height_px,
            "dm3",
            "png",
            "binary",
            material,
            nominal_particle_size_nm,
            particle_shape,
            support_material,
            instrument,
            dosage,
            pixel_size_nm,
            "Dataset_metadata.csv",
            folder,
            "demo",
        ),
    )


    hrtem_loaded += 1


print(
    f"HRTEM demo records loaded: "
    f"{hrtem_loaded}"
)

print()


# ==================================================
# Load Co3O4
# ==================================================

print("Loading Co3O4...")
print("----------------")


co3o4_images_file = (
    CO3O4_DIR / "training_images.h5"
)

co3o4_labels_file = (
    CO3O4_DIR / "training_labels.h5"
)


if not co3o4_images_file.exists():

    raise FileNotFoundError(
        f"Co3O4 image HDF5 file was not found:\n"
        f"{co3o4_images_file}"
    )


if not co3o4_labels_file.exists():

    raise FileNotFoundError(
        f"Co3O4 label HDF5 file was not found:\n"
        f"{co3o4_labels_file}"
    )


co3o4_loaded = 0


with h5py.File(
    co3o4_images_file,
    "r",
) as image_h5, h5py.File(
    co3o4_labels_file,
    "r",
) as label_h5:


    images = image_h5["images"]
    labels = label_h5["labels"]


    if len(images) != len(labels):

        raise ValueError(
            "Co3O4 image and label counts "
            "do not match."
        )


    for index in range(len(images)):

        image_shape = images[index].shape

        label_shape = labels[index].shape


        if len(image_shape) != 2:

            raise ValueError(
                f"Unexpected Co3O4 image shape "
                f"at index {index}: "
                f"{image_shape}"
            )


        height_px = int(
            image_shape[0]
        )

        width_px = int(
            image_shape[1]
        )


        if (
            label_shape[0] != height_px
            or label_shape[1] != width_px
        ):

            raise ValueError(
                f"Co3O4 image/label dimensions "
                f"do not match at index {index}."
            )


        # ------------------------------------------
        # Each database record references an index
        # inside the two HDF5 files.
        #
        # We do NOT duplicate the large arrays.
        # ------------------------------------------

        image_reference = (
            relative_path(co3o4_images_file)
            + f"::images[{index}]"
        )

        label_reference = (
            relative_path(co3o4_labels_file)
            + f"::labels[{index}]"
        )


        cursor.execute(
            INSERT_SQL,
            (
                f"Co3O4:{index:03d}",
                "Co3O4",
                f"sample_{index:03d}",
                image_reference,
                label_reference,
                width_px,
                height_px,
                "hdf5",
                "hdf5",
                "one_hot_binary",
                "Co3O4",
                None,
                None,
                None,
                None,
                None,
                None,
                "dataset_description_67_to_86_pm",
                relative_path(CO3O4_DIR),
                None,
            ),
        )


        co3o4_loaded += 1


print(
    f"Co3O4 records loaded: "
    f"{co3o4_loaded}"
)

print()


# ==================================================
# Commit database
# ==================================================

connection.commit()


# ==================================================
# Verification
# ==================================================

print("DATABASE SUMMARY")
print("================")


cursor.execute(
    """
    SELECT source_dataset, COUNT(*)
    FROM images
    GROUP BY source_dataset
    ORDER BY source_dataset;
    """
)


dataset_counts = cursor.fetchall()


for dataset, count in dataset_counts:

    print(
        f"{dataset}: {count} records"
    )


cursor.execute(
    """
    SELECT COUNT(*)
    FROM images;
    """
)

total_records = cursor.fetchone()[0]


print("----------------")
print(
    f"Total: {total_records} records"
)

print()


# ==================================================
# Expected counts
# ==================================================

expected_counts = {

    "EMPS": 465,
    "HRTEM": 30,
    "Co3O4": 256,

}


actual_counts = {
    dataset: count
    for dataset, count in dataset_counts
}


for dataset, expected in expected_counts.items():

    actual = actual_counts.get(
        dataset,
        0,
    )

    if actual != expected:

        raise ValueError(
            f"{dataset} count is incorrect. "
            f"Expected {expected}, "
            f"but database contains {actual}."
        )


expected_total = sum(
    expected_counts.values()
)


if total_records != expected_total:

    raise ValueError(
        f"Database total is incorrect. "
        f"Expected {expected_total}, "
        f"but found {total_records}."
    )


# ==================================================
# EMPS split verification
# ==================================================

print("EMPS split counts")
print("-----------------")


cursor.execute(
    """
    SELECT split, COUNT(*)
    FROM images
    WHERE source_dataset = 'EMPS'
    GROUP BY split
    ORDER BY split;
    """
)


for split, count in cursor.fetchall():

    print(
        f"{split}: {count}"
    )


# ==================================================
# HRTEM material verification
# ==================================================

print()
print("HRTEM demo material counts")
print("---------------------------")


cursor.execute(
    """
    SELECT material, COUNT(*)
    FROM images
    WHERE source_dataset = 'HRTEM'
    GROUP BY material
    ORDER BY material;
    """
)


hrtem_material_counts = (
    cursor.fetchall()
)


for material, count in hrtem_material_counts:

    print(
        f"{material}: {count}"
    )


expected_material_counts = {

    "Ag": 10,
    "Au": 15,
    "CdSe": 5,

}


actual_material_counts = {
    material: count
    for material, count
    in hrtem_material_counts
}


if (
    actual_material_counts
    != expected_material_counts
):

    raise ValueError(
        "HRTEM demo material counts "
        "are incorrect."
    )


# ==================================================
# Final success message
# ==================================================

connection.close()


print()
print("==============================")
print("DATABASE ETL SUCCESSFUL")
print("==============================")

print(
    f"Database location:\n"
    f"{DATABASE_FILE}"
)

print()

print(
    "Expected reproducible database:"
)

print(
    "EMPS   = 465"
)

print(
    "HRTEM  = 30"
)

print(
    "Co3O4  = 256"
)

print(
    "TOTAL  = 751"
)