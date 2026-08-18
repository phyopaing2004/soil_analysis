import json
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config import *

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int",
)

y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)

rows = []

for model_dir in sorted(MODEL_DIR.iterdir()):
    if not model_dir.is_dir():
        continue

    model_path = model_dir / "best_model.keras"
    if not model_path.exists():
        continue

    model = tf.keras.models.load_model(model_path)
    probs = model.predict(test_ds, verbose=0)
    pred = np.argmax(probs, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="weighted", zero_division=0
    )

    rows.append({
        "model": model_dir.name,
        "accuracy": accuracy_score(y_true, pred),
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
    })

if not rows:
    raise SystemExit("No trained models found.")

df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
out = RESULT_DIR / "model_comparison.csv"
df.to_csv(out, index=False)

print(df.to_string(index=False))
print(f"\nSaved: {out}")
print(f"Best model: {df.iloc[0]['model']}")
