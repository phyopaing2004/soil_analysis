from pathlib import Path
from PIL import Image
from collections import Counter
from config import RAW_DIR, VALID_EXTENSIONS

def find_class_dirs(root):
    """Find directories that directly contain valid image files."""
    candidates = []
    for d in root.rglob("*"):
        if d.is_dir():
            imgs = [p for p in d.iterdir()
                    if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
            if imgs:
                candidates.append((d, imgs))
    return candidates

print("=" * 72)
print("STEP 1 - DATASET CHECK")
print("=" * 72)

if not RAW_DIR.exists():
    raise SystemExit(f"Dataset folder not found: {RAW_DIR}")

candidates = find_class_dirs(RAW_DIR)

if not candidates:
    raise SystemExit(
        "No image class folders found. Put the extracted dataset under dataset/raw/."
    )

total = valid = broken = 0
class_counts = Counter()

for class_dir, images in candidates:
    c_valid = c_broken = 0
    for path in images:
        total += 1
        try:
            with Image.open(path) as img:
                img.verify()
            valid += 1
            c_valid += 1
        except Exception:
            broken += 1
            c_broken += 1
            print(f"[BROKEN] {path}")
    class_counts[class_dir.name] += c_valid
    print(f"{class_dir.name:25} valid={c_valid:5} broken={c_broken:4}")

print("-" * 72)
print(f"TOTAL  : {total}")
print(f"VALID  : {valid}")
print(f"BROKEN : {broken}")
print(f"CLASSES: {len(class_counts)}")
print("-" * 72)

for name, count in sorted(class_counts.items()):
    print(f"{name:25} {count:5}")
