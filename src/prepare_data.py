import random
from pathlib import Path

from PIL import Image


# ============================================================
# Configuration
# ============================================================

# Root location of the downloaded Kaggle dataset
RAW_DIR = Path("data/raw/cats_and_dogs")

# Where the processed dataset will be created
PROCESSED_DIR = Path("data/processed")

# Image size required for CNN
IMAGE_SIZE = (224, 224)

# Dataset split
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Reproducibility
SEED = 42

random.seed(SEED)


# ============================================================
# Check split ratios
# ============================================================

assert (
    TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0
), "Train/Validation/Test ratios must add up to 1."


# ============================================================
# Find Cat/Dog directories automatically
# ============================================================

def find_class_directories():

    possible_locations = [
        RAW_DIR,
        RAW_DIR / "PetImages"
    ]

    for location in possible_locations:

        if not location.exists():
            continue

        entries = {
            item.name.lower(): item
            for item in location.iterdir()
            if item.is_dir()
        }

        cat_dir = (
            entries.get("cat")
            or entries.get("cats")
        )

        dog_dir = (
            entries.get("dog")
            or entries.get("dogs")
        )

        if cat_dir and dog_dir:

            print("Dataset found at:")
            print(f"  {location}")
            print(f"  Cats: {cat_dir}")
            print(f"  Dogs: {dog_dir}")

            return cat_dir, dog_dir

    raise FileNotFoundError(
        "\nCould not find Cat and Dog folders.\n\n"
        "Expected one of these structures:\n\n"
        "data/raw/cats_and_dogs/Cat/\n"
        "data/raw/cats_and_dogs/Dog/\n\n"
        "OR\n\n"
        "data/raw/cats_and_dogs/cats/\n"
        "data/raw/cats_and_dogs/dogs/\n\n"
        "OR\n\n"
        "data/raw/cats_and_dogs/PetImages/Cat/\n"
        "data/raw/cats_and_dogs/PetImages/Dog/\n"
    )


# ============================================================
# Get valid image files
# ============================================================

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".jfif"
}


def get_images(directory):

    images = []

    for file in directory.iterdir():

        if not file.is_file():
            continue

        if file.suffix.lower() in VALID_EXTENSIONS:
            images.append(file)

    return images


# ============================================================
# Process images
# ============================================================

def process_images(
    image_files,
    output_directory,
    class_name
):

    successful = 0
    failed = 0

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    for index, image_path in enumerate(image_files):

        try:

            # Open image
            image = Image.open(image_path)

            # Convert every image to RGB
            image = image.convert("RGB")

            # Resize to 224 x 224
            image = image.resize(
                IMAGE_SIZE
            )

            # Always save as JPG
            output_path = (
                output_directory
                / f"{class_name}_{index:06d}.jpg"
            )

            image.save(
                output_path,
                format="JPEG",
                quality=95
            )

            successful += 1

        except Exception as error:

            failed += 1

            print(
                f"Skipping {image_path.name}: "
                f"{error}"
            )

    return successful, failed


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Cats vs Dogs - Data Preprocessing")
    print("=" * 60)

    # --------------------------------------------------------
    # Find source folders
    # --------------------------------------------------------

    cat_directory, dog_directory = (
        find_class_directories()
    )

    # --------------------------------------------------------
    # Read images
    # --------------------------------------------------------

    cat_images = get_images(
        cat_directory
    )

    dog_images = get_images(
        dog_directory
    )

    print()
    print(
        f"Found {len(cat_images)} cat images"
    )

    print(
        f"Found {len(dog_images)} dog images"
    )

    if len(cat_images) == 0:
        raise ValueError(
            "No cat images were found."
        )

    if len(dog_images) == 0:
        raise ValueError(
            "No dog images were found."
        )

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    random.shuffle(cat_images)
    random.shuffle(dog_images)

    # --------------------------------------------------------
    # Create splits
    # --------------------------------------------------------

    def split_images(images):

        total = len(images)

        train_end = int(
            total * TRAIN_RATIO
        )

        val_end = int(
            total * (TRAIN_RATIO + VAL_RATIO)
        )

        train = images[:train_end]

        val = images[
            train_end:val_end
        ]

        test = images[
            val_end:
        ]

        return train, val, test

    (
        cat_train,
        cat_val,
        cat_test
    ) = split_images(cat_images)

    (
        dog_train,
        dog_val,
        dog_test
    ) = split_images(dog_images)

    # --------------------------------------------------------
    # Define splits
    #
    # IMPORTANT:
    # Output folder names are ALWAYS lowercase.
    # --------------------------------------------------------

    splits = {

        "train": {
            "cats": cat_train,
            "dogs": dog_train
        },

        "val": {
            "cats": cat_val,
            "dogs": dog_val
        },

        "test": {
            "cats": cat_test,
            "dogs": dog_test
        }
    }

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    print()
    print("Processing images...")
    print()

    for split_name, classes in splits.items():

        for class_name, images in classes.items():

            output_directory = (
                PROCESSED_DIR
                / split_name
                / class_name
            )

            successful, failed = process_images(
                images,
                output_directory,
                class_name
            )

            print(
                f"{split_name:5s} | "
                f"{class_name:5s} | "
                f"{successful:6d} processed | "
                f"{failed:4d} failed"
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Preprocessing completed")
    print("=" * 60)

    for split_name in [
        "train",
        "val",
        "test"
    ]:

        cats_count = len(
            list(
                (
                    PROCESSED_DIR
                    / split_name
                    / "cats"
                ).glob("*.jpg")
            )
        )

        dogs_count = len(
            list(
                (
                    PROCESSED_DIR
                    / split_name
                    / "dogs"
                ).glob("*.jpg")
            )
        )

        total = (
            cats_count + dogs_count
        )

        print(
            f"{split_name:5s}: "
            f"cats={cats_count}, "
            f"dogs={dogs_count}, "
            f"total={total}"
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()