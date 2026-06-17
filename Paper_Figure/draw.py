import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

selected_paths = {
    "AM": Path("../BKDATASET/AM/20260312_124533_870770_001597.bmp"),
    "Chirp": Path("../BKDATASET/Chirp/20260312_125801_006715_003521.bmp"),
    "FM": Path("../BKDATASET/FM/20260312_161250_754774_005007.bmp"),
    "Normal": Path("../BKDATASET/Normal/20260312_130729_101843_000090.bmp"),
}

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

for i, class_name in enumerate(["AM", "Chirp", "FM", "Normal"]):
    ax = axes[i]
    img_path = selected_paths[class_name]

    if not img_path.exists():
        ax.text(0.5, 0.5, f"File not found\n{img_path.name}", ha="center", va="center", fontsize=12)
        ax.axis("off")
        continue

    img = Image.open(img_path)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(class_name, fontsize=13, labelpad=10)

plt.subplots_adjust(wspace=0.08, hspace=0.08, bottom=0.08, top=0.98)
plt.savefig("dataset_examples.png", dpi=300, bbox_inches="tight")
plt.show()