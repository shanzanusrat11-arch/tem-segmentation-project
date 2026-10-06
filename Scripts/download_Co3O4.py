from pathlib import Path
from urllib.request import Request, urlopen
import json
def download_with_resume(url, temporary_file):
    """Download a large file and resume an interrupted download."""

    existing_size = 0

    if temporary_file.exists():
        existing_size = temporary_file.stat().st_size

    headers = {}

    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        print(
            f"Resuming from "
            f"{existing_size / (1024 ** 2):.1f} MB..."
        )

    request = Request(
        url,
        headers=headers,
    )

    with urlopen(request, timeout=120) as response:

        # A 206 response means the server accepted our Range request.
        if existing_size > 0 and response.status == 206:
            mode = "ab"
        else:
            # Server did not resume, so safely restart the .part file.
            mode = "wb"
            existing_size = 0

        downloaded = existing_size
        chunk_size = 8 * 1024 * 1024

        with open(temporary_file, mode) as output_file:

            while True:
                chunk = response.read(chunk_size)

                if not chunk:
                    break

                output_file.write(chunk)
                downloaded += len(chunk)

                print(
                    f"\rDownloaded: "
                    f"{downloaded / (1024 ** 2):.1f} MB",
                    end="",
                    flush=True,
                )

    print()

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

    max_attempts = 5

for attempt in range(1, max_attempts + 1):

    try:

        print(
            f"Download attempt "
            f"{attempt}/{max_attempts}"
        )

        download_with_resume(
    download_urls[filename],
    temporary_file,
)

        temporary_file.replace(
            destination
        )

        break

    except Exception as error:

        print(
            f"Download attempt {attempt} failed: "
            f"{error}"
        )



        if attempt == max_attempts:
            raise

        print("Retrying download...")

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