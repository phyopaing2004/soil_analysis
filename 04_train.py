import json
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    MobileNetV2, EfficientNetB0, ResNet50, DenseNet121,
    InceptionV3, Xception
)

from config import *

tf.keras.utils.set_random_seed(SEED)

def load_dataset(path, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
        label_mode="int"
    )

if not TRAIN_DIR.exists():
    raise SystemExit("Run 03_split_dataset.py first.")

train_ds = load_dataset(TRAIN_DIR, True)
val_ds = load_dataset(VAL_DIR, False)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.12),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.12),
], name="augmentation")

def get_backbone(name):
    kwargs = dict(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    if name == "mobilenetv2":
        return MobileNetV2(**kwargs)
    if name == "efficientnetb0":
        return EfficientNetB0(**kwargs)
    if name == "resnet50":
        return ResNet50(**kwargs)
    if name == "densenet121":
        return DenseNet121(**kwargs)
    if name == "inceptionv3":
        return InceptionV3(input_shape=(299, 299, 3), weights="imagenet", include_top=False)
    if name == "xception":
        return Xception(input_shape=(299, 299, 3), weights="imagenet", include_top=False)
    return None

def build_model(name):
    if name == "custom_cnn":
        inputs = layers.Input(shape=(224, 224, 3))
        x = augmentation(inputs)
        x = layers.Rescaling(1./255)(x)
        for filters in [32, 64, 128, 256]:
            x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.35)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.30)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        model = Model(inputs, outputs)
        return model, None

    backbone = get_backbone(name)
    backbone.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = augmentation(inputs)

    # Inception/Xception expect 299x299; resize internally.
    if name in {"inceptionv3", "xception"}:
        x = layers.Resizing(299, 299)(x)

    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs), backbone

def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

def callbacks_for(model_name):
    folder = MODEL_DIR / model_name
    folder.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            folder / "best_model.keras",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
        ),
    ]

for model_name in MODELS_TO_TRAIN:
    print("\n" + "=" * 72)
    print(f"TRAINING: {model_name}")
    print("=" * 72)

    model, backbone = build_model(model_name)
    compile_model(model, LEARNING_RATE)

    history_initial = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks_for(model_name),
    )

    # Fine-tune only the final N backbone layers.
    if backbone is not None:
        backbone.trainable = True
        for layer in backbone.layers[:-FINE_TUNE_LAYERS]:
            layer.trainable = False

        compile_model(model, FINE_TUNE_LR)

        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=FINE_TUNE_EPOCHS,
            callbacks=callbacks_for(model_name),
        )
    else:
        history_ft = None

    folder = MODEL_DIR / model_name
    model.save(folder / "final_model.keras")

    with open(folder / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # Save simple training curves.
    import matplotlib.pyplot as plt

    def plot_history(histories, filename):
        acc, val_acc, loss, val_loss = [], [], [], []
        for h in histories:
            acc += h.history.get("accuracy", [])
            val_acc += h.history.get("val_accuracy", [])
            loss += h.history.get("loss", [])
            val_loss += h.history.get("val_loss", [])

        plt.figure(figsize=(8, 5))
        plt.plot(acc, label="Train Accuracy")
        plt.plot(val_acc, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{model_name} Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "plots" / f"{model_name}_accuracy.png")
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(loss, label="Train Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{model_name} Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "plots" / f"{model_name}_loss.png")
        plt.close()

    plot_history(
        [history_initial] + ([history_ft] if history_ft else []),
        model_name,
    )

    print(f"Completed: {model_name}")

print("\nAll selected models completed.")
