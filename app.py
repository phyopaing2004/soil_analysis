import io
from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite  # သို့မဟုတ် import tensorflow as tf

app = Flask(__name__)

# TFLite Model ကို Load လုပ်ခြင်း
interpreter = tflite.Interpreter(model_path="efficientnet_model.tflite")
interpreter.allocate_tensors()

# Input နဲ့ Output details ယူခြင်း
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Image Preprocessing (သင့် Model Input Size ပေါ်မူတည်ပြီး ပြင်ဆင်ပါ)
    image = Image.open(file.stream).convert("RGB")
    image = image.resize((224, 224))  # Model ရဲ့ input size အတိုင်း ပြင်ပါ
    input_data = np.expand_dims(image, axis=0).astype(np.float32)

    # Normalization လိုအပ်ပါက ပြုလုပ်ရန် (ဥပမာ - / 255.0)
    input_data = input_data / 255.0

    # Prediction ပြုလုပ်ခြင်း
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])

    # Output ကို Process လုပ်ခြင်း
    result = float(output_data[0][0])  # Output shape ပေါ်မူတည်ပြီး ပြင်ပါ

    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(debug=True)