import argparse
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

from config import *

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to an image")
parser.add_argument("--model", default="mobilenetv2", help="Model folder name")
args = parser.parse_args()

model_path = MODEL_DIR / args.model / "best_model.keras"
class_path = MODEL_DIR / args.model / "class_names.json"

if not model_path.exists():
    raise SystemExit(f"Model not found: {model_path}")

if not Path(args.image).exists():
    raise SystemExit(f"Image not found: {args.image}")

model = tf.keras.models.load_model(model_path)

with open(class_path, encoding="utf-8") as f:
    class_names = json.load(f)

img = tf.keras.utils.load_img(
    args.image,
    target_size=IMG_SIZE
)
arr = tf.keras.utils.img_to_array(img)
arr = np.expand_dims(arr, axis=0)

probs = model.predict(arr, verbose=0)[0]
indices = np.argsort(probs)[::-1][:3]

print("=" * 60)
print("SOIL CLASSIFICATION - TOP 3")
print("=" * 60)

for rank, idx in enumerate(indices, 1):
    print(
        f"{rank}. {class_names[idx]:25} "
        f"{probs[idx] * 100:.2f}%"
    )
