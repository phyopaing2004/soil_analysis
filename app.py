import json
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path

from config import *  # config.py ထဲက IMG_SIZE ကို သုံးပါမည်

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# app.py ရှိသော Folder လမ်းကြောင်းကို ယူခြင်း
BASE_DIR = Path(__file__).resolve().parent

# Local ရော Render မှာပါ လမ်းကြောင်း မလွဲအောင် app.py နားက models ကို တိုက်ရိုက်ညွှန်းခြင်း
MODEL_NAME = "efficientnetb0"
model_path = BASE_DIR / "models" / MODEL_NAME / "best_model.keras"
class_path = BASE_DIR / "models" / MODEL_NAME / "class_names.json"

if not model_path.exists():
    raise SystemExit(f"Model not found at: {model_path}")

# Keras Model Direct Loading
model = tf.keras.models.load_model(model_path)

with open(class_path, encoding="utf-8") as f:
    class_names = json.load(f)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "online", "message": "Soil Analysis API is Ready"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    
    # 07_predict.py ပါ ပုံစံအတိုင်း Exact Image Processing ပြုလုပ်ခြင်း
    img = Image.open(file.stream).convert('RGB')
    
    # IMG_SIZE အတိုင်း Resize လုပ်ခြင်း
    img = img.resize((IMG_SIZE[1], IMG_SIZE[0]))
    
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    
    # Prediction ယူခြင်း
    probs = model.predict(arr, verbose=0)[0]
    
    max_idx = int(np.argmax(probs))
    confidence = float(probs[max_idx])
    
    return jsonify({
        'class': class_names[max_idx],
        'confidence': f"{confidence * 100:.2f}%"
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)