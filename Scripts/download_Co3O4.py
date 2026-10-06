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

# ==================================================
# Retrieve official Zenodo metadata
# ==================================================

print()
print("Co3O4 DOWNLOAD")
print("==============")
print("Retrieving file information from Zenodo...")

with urlopen(ZENODO_API_URL) as response:
    metadata = json.load(response)


# ==================================================
# Find download URLs and expected file sizes
# ==================================================

download_urls = {}
expected_sizes = {}

for file_info in metadata["files"]:

    filename = file_info["key"]

    if filename in REQUIRED_FILES:

        download_urls[filename] = (
            file_info["links"]["self"]
        )

        expected_sizes[filename] = (
            file_info["size"]
        )


for filename in REQUIRED_FILES:

    if filename not in download_urls:

        raise RuntimeError(
            f"{filename} was not found "
            "in the Zenodo record."
        )


# ==================================================
# Check whether a downloaded file is complete
# ==================================================

def file_is_complete(filename):

    file_path = CO3O4_DIR / filename

    if not file_path.exists():
        return False

    return (
        file_path.stat().st_size
        == expected_sizes[filename]
    )


def co3o4_is_complete():

    return all(
        file_is_complete(filename)
        for filename in REQUIRED_FILES
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

    destination = CO3O4_DIR / filename
    temporary_file = CO3O4_DIR / f"{filename}.part"

    expected_size = expected_sizes[filename]

    # --------------------------------------------------
    # Skip a file only when its size is exactly correct
    # --------------------------------------------------

    if file_is_complete(filename):

        size_mb = (
            destination.stat().st_size
            / (1024 * 1024)
        )

        print(
            f"{filename} already exists and is complete "
            f"({size_mb:.1f} MB). Skipping."
        )

        continue

    # --------------------------------------------------
    # If an incomplete final file exists, turn it back
    # into a .part file so the download can resume.
    # --------------------------------------------------

    if destination.exists():

        current_size = destination.stat().st_size

        print(
            f"{filename} is incomplete "
            f"({current_size / (1024 ** 2):.1f} MB)."
        )

        if current_size < expected_size:

            if temporary_file.exists():
                temporary_file.unlink()

            destination.replace(temporary_file)

            print(
                "Incomplete file moved to .part "
                "so the download can resume."
            )

        else:

            destination.unlink()

            if temporary_file.exists():
                temporary_file.unlink()

            print(
                "Invalid oversized file removed. "
                "Download will restart."
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

            actual_size = temporary_file.stat().st_size

            if actual_size != expected_size:

                raise RuntimeError(
                    f"Incomplete download for {filename}: "
                    f"expected {expected_size} bytes, "
                    f"got {actual_size} bytes."
                )

            temporary_file.replace(destination)

            print(
                f"{filename} download complete."
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


# ==================================================
# Validate result
# ==================================================

if not co3o4_is_complete():

    raise RuntimeError(
        "Co3O4 download completed, "
        "but one or more files have an incorrect size."
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
print("All Co3O4 files verified successfully.")