# src/create_small_dataset.py

from pathlib import Path
import shutil
import random

random.seed(42)

SOURCE_DIR = Path("data/processed")
TARGET_DIR = Path("data/processed_small")

SIZES = {
    "train": 5000,
    "val": 1500,
    "test": 1500,
}

CLASSES = ["cats", "dogs"]


for split, count in SIZES.items():

    for class_name in CLASSES:

        source = SOURCE_DIR / split / class_name
        target = TARGET_DIR / split / class_name

        target.mkdir(
            parents=True,
            exist_ok=True
        )

        images = [
            p for p in source.iterdir()
            if p.is_file()
        ]

        random.shuffle(images)

        selected_images = images[:count]

        print(
            f"{split}/{class_name}: "
            f"{len(selected_images)} images"
        )

        for image in selected_images:

            shutil.copy2(
                image,
                target / image.name
            )

print("\nSmall dataset created successfully.")