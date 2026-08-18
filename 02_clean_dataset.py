from pathlib import Path
from PIL import Image
import shutil
from config import RAW_DIR, CLEAN_DIR, VALID_EXTENSIONS

def find_class_dirs(root):
    result = []
    for d in root.rglob("*"):
        if d.is_dir():
            imgs = [p for p in d.iterdir()
                    if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
            if imgs:
                result.append((d, imgs))
    return result

print("=" * 72)
print("STEP 2 - CLEAN DATASET")
print("=" * 72)

if not RAW_DIR.exists():
    raise SystemExit("dataset/raw does not exist.")

copied = removed = 0
class_dirs = find_class_dirs(RAW_DIR)

for class_dir, images in class_dirs:
    target_dir = CLEAN_DIR / class_dir.name
    target_dir.mkdir(parents=True, exist_ok=True)

    for path in images:
        try:
            with Image.open(path) as img:
                img.verify()

            target = target_dir / path.name
            if target.exists():
                target = target_dir / f"{path.stem}_{copied}{path.suffix}"

            shutil.copy2(path, target)
            copied += 1
        except Exception:
            removed += 1
            print(f"[SKIP BROKEN] {path}")

print("-" * 72)
print(f"Copied : {copied}")
print(f"Skipped: {removed}")
print(f"Output : {CLEAN_DIR}")
