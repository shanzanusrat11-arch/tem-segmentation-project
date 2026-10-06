from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


DATA_DIR = Path("data/raw/emps")

images_dir = DATA_DIR / "images"
segmaps_dir = DATA_DIR / "segmaps"


# Search for an image containing several particle instances
for mask_path in segmaps_dir.glob("*.png"):

    mask_array = np.array(Image.open(mask_path))

    unique_values = np.unique(mask_array)

    # 0 is the background.
    # More than 5 unique values means several particles are present.
    if len(unique_values) > 5:
        filename = mask_path.name
        break


# Find the matching original image and segmentation map
image_path = images_dir / filename
mask_path = segmaps_dir / filename


# Open the image and segmentation map
image = Image.open(image_path)
mask = Image.open(mask_path)

mask_array = np.array(mask)


# Print information about this image
print("Selected file:", filename)
print("Image size:", image.size)
print("Mask size:", mask.size)
print("Mask data type:", mask_array.dtype)

print("Unique mask values:")
print(np.unique(mask_array))

print("Number of particles:")
print(len(np.unique(mask_array)) - 1)


# Display the original image and segmentation side by side
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original EM Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask_array, cmap="nipy_spectral")
plt.title("Ground-Truth Instance Segmentation")
plt.axis("off")

plt.tight_layout()
plt.show()