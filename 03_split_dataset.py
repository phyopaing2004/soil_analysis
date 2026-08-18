from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from config import (
    CLEAN_DIR, SPLIT_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    VALID_EXTENSIONS, SEED
)

print("=" * 72)
print("STEP 3 - TRAIN / VALIDATION / TEST SPLIT")
print("=" * 72)

if not CLEAN_DIR.exists():
    raise SystemExit("Run 02_clean_dataset.py first.")

for split in ["train", "val", "test"]:
    d = SPLIT_DIR / split
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

classes = [
    d for d in CLEAN_DIR.iterdir()
    if d.is_dir()
]

if len(classes) < 2:
    raise SystemExit("At least two classes are required.")

for class_dir in sorted(classes):
    images = [
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]

    if len(images) < 3:
        print(f"[WARNING] {class_dir.name}: only {len(images)} images")
        continue

    train_files, temp = train_test_split(
        images, test_size=(1 - TRAIN_RATIO), random_state=SEED
    )

    val_fraction_of_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    val_files, test_files = train_test_split(
        temp, test_size=(1 - val_fraction_of_temp), random_state=SEED
    )

    for split, selected in [
        ("train", train_files),
        ("val", val_files),
        ("test", test_files),
    ]:
        out = SPLIT_DIR / split / class_dir.name
        out.mkdir(parents=True, exist_ok=True)
        for src in selected:
            shutil.copy2(src, out / src.name)

    print(
        f"{class_dir.name:25} "
        f"train={len(train_files):4} "
        f"val={len(val_files):4} "
        f"test={len(test_files):4}"
    )

print("\nSplit completed.")
