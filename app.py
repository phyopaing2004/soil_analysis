import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # HTML ဘက်ကနေ Request လက်ခံနိုင်ရန်

# 1. Model Load လုပ်ခြင်း
model_path = "efficientnet_model.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = ["Sandy", "Loam_Soil"] # သင့် Class အစဉ်အတိုင်း ပြင်ပါ

# --- Root Route ဖြည့်စွက်ချက် ---
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "message": "Agribot Fruit Classifier API is running successfully!"
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    
    # EfficientNet Input Size (ဥပမာ 224x224)
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
    app.run(port=5000, debug=True)