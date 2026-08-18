import os
import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image

# Import tflite_runtime (lightweight alternative to full TensorFlow)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

app = Flask(__name__)

MODEL_PATH = "efficientnet_model.tflite"

# Load TFLite Model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_type = input_details[0]['dtype']

@app.route('/')
def home():
    """Serves the frontend page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API Endpoint for model inference."""
    try:
        # Case 1: Image Upload Input
        if 'file' in request.files:
            file = request.files['file']
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
            
            # Resize image according to model input dimensions
            target_h, target_w = input_shape[1], input_shape[2]
            image = image.resize((target_w, target_h))
            
            input_data = np.array(image, dtype=np.float32)
            
            # Normalize pixel values if model expects 0.0 - 1.0 range
            if input_type == np.float32:
                input_data = input_data / 255.0
                
            input_data = np.expand_dims(input_data, axis=0)

        # Case 2: JSON Numerical Array Input
        elif request.is_json:
            payload = request.get_json()
            input_data = np.array(payload['data'], dtype=input_type)
            if len(input_data.shape) == len(input_shape) - 1:
                input_data = np.expand_dims(input_data, axis=0)

        else:
            return jsonify({'error': 'No file or valid JSON data provided'}), 400

        # Perform Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        return jsonify({
            'status': 'success',
            'predictions': output_data.tolist()
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)