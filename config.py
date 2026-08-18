from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "dataset" / "raw"
CLEAN_DIR = BASE_DIR / "dataset" / "cleaned"
SPLIT_DIR = BASE_DIR / "dataset" / "split"

TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR = SPLIT_DIR / "val"
TEST_DIR = SPLIT_DIR / "test"

MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 8
FINE_TUNE_LAYERS = 30

LEARNING_RATE = 1e-4
FINE_TUNE_LR = 1e-5

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Start small on low-end PCs.
MODELS_TO_TRAIN = [
    "mobilenetv2",
    "efficientnetb0",
    # "resnet50",
    # "densenet121",
    # "inceptionv3",
    # "xception",
    # "custom_cnn",
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for d in [MODEL_DIR, RESULT_DIR / "plots",
          RESULT_DIR / "confusion_matrices",
          RESULT_DIR / "reports",
          RESULT_DIR / "predictions"]:
    d.mkdir(parents=True, exist_ok=True)
