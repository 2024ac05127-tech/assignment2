import random
from pathlib import Path

from PIL import Image


# ==================================================
# Configuration
# ==================================================

SOURCE_DIR = Path("data/raw/cats_and_dogs")

OUTPUT_DIR = Path("data/processed")

IMAGE_SIZE = (224, 224)

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42

random.seed(SEED)


# ==================================================
# Check source directory
# ==================================================

if not SOURCE_DIR.exists():

    raise FileNotFoundError(
        f"Source directory does not exist: {SOURCE_DIR}"
    )


print("Source directory:", SOURCE_DIR)


# ==================================================
# Classes
# ==================================================

classes = ["Cat", "Dog"]


# ==================================================
# Create output directories
# ==================================================

for split in ["train", "val", "test"]:

    for class_name in classes:

        output_path = (
            OUTPUT_DIR
            / split
            / class_name
        )

        output_path.mkdir(
            parents=True,
            exist_ok=True
        )


# ==================================================
# Process each class
# ==================================================

for class_name in classes:

    source_class_dir = (
        SOURCE_DIR / class_name
    )

    if not source_class_dir.exists():

        print(
            f"WARNING: {source_class_dir} "
            f"does not exist"
        )

        continue


    # Get ALL files first
    files = [
        f
        for f in source_class_dir.iterdir()
        if f.is_file()
    ]


    print(
        f"\n{class_name}: "
        f"Found {len(files)} files"
    )


    # Shuffle

    random.shuffle(files)


    # ==================================================
    # Split
    # ==================================================

    total = len(files)

    train_end = int(
        total * TRAIN_RATIO
    )

    val_end = train_end + int(
        total * VAL_RATIO
    )


    train_files = files[:train_end]

    val_files = files[
        train_end:val_end
    ]

    test_files = files[
        val_end:
    ]


    splits = {

        "train": train_files,

        "val": val_files,

        "test": test_files

    }


    # ==================================================
    # Process images
    # ==================================================

    for split, split_files in splits.items():

        successful = 0

        failed = 0


        for index, image_path in enumerate(
            split_files
        ):

            try:

                # Open image
                image = Image.open(
                    image_path
                )


                # Convert to RGB
                image = image.convert(
                    "RGB"
                )


                # Resize
                image = image.resize(
                    IMAGE_SIZE
                )


                # Output path
                output_path = (
                    OUTPUT_DIR
                    / split
                    / class_name
                    / f"{class_name}_{index}.jpg"
                )


                # Save
                image.save(
                    output_path,
                    "JPEG"
                )


                successful += 1


            except Exception as e:

                failed += 1

                print(
                    f"Could not process "
                    f"{image_path}: {e}"
                )


        print(
            f"{split} {class_name}: "
            f"{successful} processed, "
            f"{failed} failed"
        )


# ==================================================
# Final summary
# ==================================================

print("\n================================")
print("Preprocessing completed")
print("================================")


for split in ["train", "val", "test"]:

    print(f"\n{split}:")

    for class_name in classes:

        directory = (
            OUTPUT_DIR
            / split
            / class_name
        )

        count = len(
            list(
                directory.glob("*.jpg")
            )
        )

        print(
            f"  {class_name}: {count}"
        )