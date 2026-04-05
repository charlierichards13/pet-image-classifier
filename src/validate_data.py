from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def validate_dataset():
    print("Scanning dataset...\n")

    total_files = 0
    extension_counts = Counter()
    invalid_files = []

    for class_dir in DATA_DIR.iterdir():
        if class_dir.is_dir():
            print(f"Class: {class_dir.name}")

            files = list(class_dir.glob("*"))
            print(f"  Number of files: {len(files)}")

            for file in files:
                total_files += 1
                ext = file.suffix.lower()

                extension_counts[ext] += 1

                if ext not in VALID_EXTENSIONS:
                    invalid_files.append(file)

    print("\n--- Summary ---")
    print(f"Total files: {total_files}")
    print("\nFile types:")
    for ext, count in extension_counts.items():
        print(f"  {ext}: {count}")

    if invalid_files:
        print("\nInvalid files detected:")
        for f in invalid_files:
            print(f"  {f}")
    else:
        print("\nNo invalid files found.")


if __name__ == "__main__":
    validate_dataset()