import tensorflow as tf

# Load trained model (.h5)
model = tf.keras.models.load_model(r"D:\Agribot Project\Fruit\models\efficientnetb0\best_model.keras")
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save file
with open("efficientnet_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ EfficientNetB0 converted to TFLite")