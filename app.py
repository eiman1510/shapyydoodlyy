from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image, ImageOps
import io
import base64
import os
import joblib
import logging
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

# Global variables for model and features
model = None
img_size = 64
target_features = 1568

def load_model():
    """Load the model if not already loaded"""
    global model
    try:
        if model is None:
            model_path = os.path.join(os.path.dirname(__file__), 'shapes_model.joblib')
            logger.info(f"Loading model from {model_path}")
            model = joblib.load(model_path)
            logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def preprocess_image(image_data):
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Invert colors (we trained on black shapes on white background)
        image = ImageOps.invert(image)
        
        # Resize to match model input size
        image = image.resize((img_size, img_size))
        
        # Convert to numpy array and normalize
        img_array = np.array(image) / 255.0
        
        # Downsample to match the expected number of features
        downsample_size = int(np.sqrt(target_features/4))
        image = Image.fromarray((img_array * 255).astype(np.uint8))
        image = image.resize((downsample_size * 2, downsample_size * 2))
        
        # Convert back to numpy and normalize
        img_array = np.array(image) / 255.0
        
        # Create blocks of 2x2 pixels and calculate their average
        blocks = img_array.reshape(downsample_size, 2, downsample_size, 2).mean(axis=(1, 3))
        
        # Flatten and ensure we have exactly target_features
        features = blocks.reshape(-1)
        if len(features) < target_features:
            features = np.pad(features, (0, target_features - len(features)))
        elif len(features) > target_features:
            features = features[:target_features]
            
        return features
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

@app.before_first_request
def initialize():
    """Initialize the model before first request"""
    load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure model is loaded
        if model is None:
            load_model()
            
        # Get image data from request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Preprocess image and extract features
        features = preprocess_image(image_data)
        
        # Make prediction
        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]
        
        # Get confidence scores for all classes
        confidence_scores = {
            class_name: float(prob)
            for class_name, prob in zip(model.classes_, probabilities)
        }
        
        logger.info(f"Successful prediction: {prediction}")
        return jsonify({
            'prediction': prediction,
            'confidence_scores': confidence_scores
        })
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Verify model is loaded
        if model is None:
            load_model()
        return jsonify({
            'status': 'healthy',
            'model_loaded': model is not None
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 