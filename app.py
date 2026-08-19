import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path

app = Flask(__name__)
CORS(app)

# ၁။ app.py ရှိသော Folder လမ်းကြောင်းအတိုင်း tflite ဖိုင်ကို လှမ်းခေါ်ခြင်း
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "efficientnet_model.tflite"

interpreter = tf.lite.Interpreter(model_path=str(model_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# အက္ခရာစဉ် (Alphabetical Order) အရ 0: Loam_Soil, 1: Sandy ဖြစ်ပါသည်
labels = ["Loam_Soil", "Sandy"]

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "message": "Agribot Soil Classifier API is running successfully!"
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    
    # Model Input Size အလိုက် Resize လုပ်ခြင်း
    input_shape = input_details[0]['shape']
    img = img.resize((input_shape[2], input_shape[1]))
    
    input_data = np.expand_dims(img, axis=0).astype(np.float32)
    
    # Prediction
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    max_idx = int(np.argmax(output_data))
    confidence = float(np.max(output_data))
    
    return jsonify({
        'class': labels[max_idx],
        'confidence': f"{confidence * 100:.2f}%"
    })

if __name__ == '__main__':
    # Render ၏ Dynamic Port Allocation အတွက် စီစဉ်ပေးခြင်း
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)