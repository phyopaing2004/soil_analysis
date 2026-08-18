import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from config import *

MODEL_NAME = "efficientnetb0"

model_path = MODEL_DIR / MODEL_NAME / "best_model.keras"
class_path = MODEL_DIR / MODEL_NAME / "class_names.json"

if not model_path.exists():
    raise SystemExit(f"Model not found: {model_path}")

model = tf.keras.models.load_model(model_path)

with open(class_path, encoding="utf-8") as f:
    class_names = json.load(f)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int",
)

y_true, y_pred = [], []

for images, labels in test_ds:
    probs = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(probs, axis=1))
    y_true.extend(labels.numpy())

report = classification_report(
    y_true, y_pred, target_names=class_names, digits=4
)

print("=" * 72)
print(f"EVALUATION: {MODEL_NAME}")
print("=" * 72)
print(report)

report_path = RESULT_DIR / "reports" / f"{MODEL_NAME}_classification_report.txt"
report_path.write_text(report, encoding="utf-8")

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(9, 7))
plt.imshow(cm)
plt.colorbar()
plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
plt.yticks(range(len(class_names)), class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"{MODEL_NAME} Confusion Matrix")

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
out = RESULT_DIR / "confusion_matrices" / f"{MODEL_NAME}.png"
plt.savefig(out, dpi=160)
plt.close()

print(f"Report saved: {report_path}")
print(f"Matrix saved: {out}")
