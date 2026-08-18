import io
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Cross-Origin Request များကို လက်ခံရန်

# -------------------------------------------------------------
# 1. TFLite Model ကို TensorFlow Lite Interpreter ဖြင့် Load လုပ်ခြင်း
# -------------------------------------------------------------
MODEL_PATH = "efficientnet_model.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Input နှင့် Output Tensor Details ယူခြင်း
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class Label များ (သင့် Model ရဲ့ Label များအတိုင်း လိုအပ်ပါက ပြင်ပါ)
CLASSES = ["Loam_Soil", "Sandy_Soil"]


# -------------------------------------------------------------
# 2. Flask Routes
# -------------------------------------------------------------
@app.route("/")
def index():
  return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
  try:
    if "file" not in request.files:
      return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
      return jsonify({"error": "No file selected"}), 400

    # Image Preprocessing
    image = Image.open(file.stream).convert("RGB")

    # EfficientNet Input Size သတ်မှတ်ခြင်း (224x224)
    image = image.resize((224, 224))

    # Numpy Array အဖြစ် ပြောင်းလဲပြီး Batch Dimension ထည့်ခြင်း (Shape: [1, 224, 224, 3])
    input_data = np.expand_dims(image, axis=0).astype(np.float32)

    # Normalization (0 မှ 1 အတွင်း ပြောင်းခြင်း)
    input_data = input_data / 255.0

    # Model သို့ Input Data ထည့်သွင်း၍ Predict လုပ်ခြင်း
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    # Result ရယူခြင်း
    output_data = interpreter.get_tensor(output_details[0]["index"])

    # Output processing (Classification Result)
    predicted_index = int(np.argmax(output_data[0]))
    confidence = float(np.max(output_data[0])) * 100

    return jsonify({
        "prediction": CLASSES[predicted_index],
        "confidence": round(confidence, 2),
    })

  except Exception as e:
    return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000, debug=True)