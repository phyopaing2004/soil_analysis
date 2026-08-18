import json
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 1. Model Load ပြုလုပ်ခြင်း
MODEL_PATH = "efficientnet_model.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. Class Names ကို JSON ဖိုင်မှ တိုက်ရိုက်ဖတ်ပါ (CMD ထဲကအတိုင်း Index မမှားစေရန်)
# (အကယ်၍ class_names.json မရှိပါက CMD ထဲက အစဉ်အတိုင်း labels = ["Sandy", "Loam_Soil"] စစ်ကြည့်ပါ)
try:
    with open("class_names.json", encoding="utf-8") as f:
        labels = json.load(f)
except:
    labels = ["Sandy", "Loam_Soil"]  # CMD ထဲက Index 0 က Sandy ဖြစ်နေပါက ဤအတိုင်းထားပါ


@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "online", "message": "Soil API is running!"})


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')

    # Input Size ပြောင်းခြင်း
    target_height = input_details[0]['shape'][1]
    target_width = input_details[0]['shape'][2]
    img = img.resize((target_width, target_height))

    # Array ပြောင်းခြင်း
    input_data = np.array(img, dtype=np.float32)

    # *** အရေးကြီးသည်: EfficientNet Preprocessing / Pixel Scaling ***
    # EfficientNet တော်တော်များများအတွက် [0, 255] ကို float32 ထားရင် ရသလို [0, 1] Scaling လုပ်ရတာလည်း ရှိပါတယ်။
    # အကယ်၍ အဖြေလွဲနေပါက / 255.0 ကို ဖြုတ်/ထည့် စမ်းသပ်ကြည့်ပါ:
    input_data = input_data / 255.0

    input_data = np.expand_dims(input_data, axis=0)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Softmax output မဟုတ်သေးပါက Softmax ပြန်တွက်ခြင်း (Probability Scaling)
    exp_scores = np.exp(output_data - np.max(output_data))
    probs = exp_scores / np.sum(exp_scores)

    max_idx = int(np.argmax(probs))
    confidence = float(probs[max_idx])

    return jsonify({
        'class': labels[max_idx],
        'confidence': f"{confidence * 100:.2f}%"
    })


if __name__ == '__main__':
    app.run(port=5000, debug=True)