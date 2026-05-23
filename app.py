from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import numpy as np
import io

# Initialize Flask Application
application = Flask(__name__)

# Load Trained Deep Learning Model
trained_model = tf.keras.models.load_model('best_model.h5')

# Load Class Labels
with open('labels.txt', 'r') as label_file:
    class_labels = [label.strip() for label in label_file.readlines()]

# Frontend HTML Template - Modern Design
WEB_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>✨ AI Vision Studio | Smart Image Classifier</title>
    
    <link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,600;14..32,700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            position: relative;
            overflow-x: hidden;
        }

        /* Animated Background */
        body::before {
            content: '';
            position: absolute;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
            background-size: 50px 50px;
            animation: float 20s linear infinite;
            pointer-events: none;
        }

        @keyframes float {
            0% { transform: translate(0, 0); }
            100% { transform: translate(-50px, -50px); }
        }

        /* Main Container */
        .container {
            position: relative;
            z-index: 1;
            width: 100%;
            max-width: 600px;
        }

        /* Glassmorphism Card */
        .glass-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 32px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 30px 70px rgba(0, 0, 0, 0.4);
        }

        /* Header Section */
        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .icon-wrapper {
            display: inline-block;
            background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 50%;
            padding: 20px;
            margin-bottom: 20px;
            animation: pulse 2s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }

        .icon-wrapper i {
            font-size: 48px;
            color: white;
        }

        h1 {
            font-size: 28px;
            color: #1a1a2e;
            margin-bottom: 10px;
            font-weight: 700;
        }

        .subtitle {
            color: #666;
            font-size: 14px;
        }

        /* Upload Area */
        .upload-area {
            border: 2px dashed #ddd;
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #f8f9fa;
            margin-bottom: 20px;
        }

        .upload-area:hover {
            border-color: #667eea;
            background: #f0f1ff;
        }

        .upload-area.drag-over {
            border-color: #667eea;
            background: #e8eaff;
            transform: scale(0.98);
        }

        .upload-icon {
            font-size: 48px;
            color: #667eea;
            margin-bottom: 15px;
        }

        .upload-text {
            color: #666;
            font-size: 14px;
        }

        .upload-text strong {
            color: #667eea;
        }

        input[type="file"] {
            display: none;
        }

        /* Preview Section */
        .preview-section {
            margin: 20px 0;
            display: none;
        }

        .image-preview {
            position: relative;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
        }

        .image-preview img {
            width: 100%;
            height: auto;
            display: block;
        }

        .remove-image {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            border: none;
            border-radius: 50%;
            width: 35px;
            height: 35px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .remove-image:hover {
            background: #dc2626;
            transform: scale(1.1);
        }

        /* Predict Button */
        .predict-btn {
            width: 100%;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 16px;
            border-radius: 50px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }

        .predict-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        }

        .predict-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        /* Loading Animation */
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }

        .spinner {
            width: 50px;
            height: 50px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Result Section */
        .result-section {
            margin-top: 20px;
            display: none;
            animation: slideIn 0.5s ease;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .result-card {
            background: linear-gradient(135deg, #667eea15, #764ba215);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(102, 126, 234, 0.3);
        }

        .prediction-label {
            font-size: 24px;
            font-weight: 700;
            color: #1a1a2e;
            margin-bottom: 10px;
            text-align: center;
        }

        .confidence-bar {
            background: #e0e0e0;
            border-radius: 50px;
            overflow: hidden;
            height: 30px;
            margin: 15px 0;
        }

        .confidence-fill {
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 12px;
            font-weight: 600;
            transition: width 1s ease;
            border-radius: 50px;
        }

        .confidence-text {
            text-align: center;
            color: #666;
            font-size: 14px;
            margin-top: 10px;
        }

        /* Error Style */
        .error {
            background: linear-gradient(135deg, #f5656515, #ed64a615);
            color: #c53030;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
        }

        /* Footer */
        .footer {
            text-align: center;
            margin-top: 30px;
            color: rgba(255, 255, 255, 0.8);
            font-size: 12px;
        }

        /* Responsive */
        @media (max-width: 640px) {
            .glass-card {
                padding: 25px;
            }
            
            h1 {
                font-size: 24px;
            }
            
            .prediction-label {
                font-size: 20px;
            }
        }
    </style>
</head>

<body>
    <div class="container">
        <div class="glass-card">
            <div class="header">
                <div class="icon-wrapper">
                    <i class="fas fa-brain"></i>
                </div>
                <h1>AI Vision Studio</h1>
                <p class="subtitle">Advanced Image Recognition Powered by Deep Learning</p>
            </div>

            <div class="upload-area" id="uploadArea">
                <i class="fas fa-cloud-upload-alt upload-icon"></i>
                <p class="upload-text">
                    <strong>Click to upload</strong> or drag and drop<br>
                    <small>PNG, JPG, JPEG up to 10MB</small>
                </p>
                <input type="file" id="fileInput" accept="image/*">
            </div>

            <div class="preview-section" id="previewSection">
                <div class="image-preview">
                    <img id="previewImage" alt="Preview">
                    <button class="remove-image" id="removeImage">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </div>

            <button class="predict-btn" id="predictBtn" disabled>
                <i class="fas fa-magic"></i>
                <span>Analyze Image</span>
            </button>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 10px; color: #666;">Analyzing with AI...</p>
            </div>

            <div class="result-section" id="resultSection">
                <div class="result-card" id="resultCard"></div>
            </div>
        </div>
        <div class="footer">
            <i class="fas fa-shield-alt"></i> Powered by TensorFlow Deep Learning
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const previewSection = document.getElementById('previewSection');
        const previewImage = document.getElementById('previewImage');
        const removeImage = document.getElementById('removeImage');
        const predictBtn = document.getElementById('predictBtn');
        const loading = document.getElementById('loading');
        const resultSection = document.getElementById('resultSection');
        const resultCard = document.getElementById('resultCard');

        let currentFile = null;

        // Upload area click
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('drag-over');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                handleFile(file);
            }
        });

        // File input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files[0]) {
                handleFile(e.target.files[0]);
            }
        });

        // Remove image
        removeImage.addEventListener('click', () => {
            currentFile = null;
            previewSection.style.display = 'none';
            uploadArea.style.display = 'block';
            predictBtn.disabled = true;
            resultSection.style.display = 'none';
            fileInput.value = '';
        });

        function handleFile(file) {
            currentFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                previewSection.style.display = 'block';
                uploadArea.style.display = 'none';
                predictBtn.disabled = false;
                resultSection.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }

        // Predict
        predictBtn.addEventListener('click', async () => {
            if (!currentFile) return;

            predictBtn.disabled = true;
            loading.style.display = 'block';
            resultSection.style.display = 'none';

            const formData = new FormData();
            formData.append('file', currentFile);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.prediction) {
                    const confidencePercent = (data.confidence * 100).toFixed(2);
                    
                    resultCard.innerHTML = `
                        <div style="text-align: center; margin-bottom: 15px;">
                            <i class="fas fa-robot" style="font-size: 40px; color: #667eea;"></i>
                        </div>
                        <div class="prediction-label">
                            ${data.prediction}
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: 0%;">
                                ${confidencePercent}%
                            </div>
                        </div>
                        <div class="confidence-text">
                            <i class="fas fa-chart-line"></i> Confidence Score: ${confidencePercent}%
                        </div>
                        <div style="text-align: center; margin-top: 15px; font-size: 12px; color: #999;">
                            <i class="fas fa-check-circle" style="color: #10b981;"></i> AI Analysis Complete
                        </div>
                    `;
                    
                    // Animate confidence bar
                    setTimeout(() => {
                        const fill = document.querySelector('.confidence-fill');
                        if (fill) {
                            fill.style.width = `${confidencePercent}%`;
                        }
                    }, 100);
                } else {
                    resultCard.innerHTML = `
                        <div class="error">
                            <i class="fas fa-exclamation-triangle"></i>
                            <strong>Error:</strong> ${data.error}
                        </div>
                    `;
                }
                
                resultSection.style.display = 'block';
            } catch (error) {
                resultCard.innerHTML = `
                    <div class="error">
                        <i class="fas fa-exclamation-triangle"></i>
                        <strong>Error:</strong> Failed to analyze image. Please try again.
                    </div>
                `;
                resultSection.style.display = 'block';
            } finally {
                loading.style.display = 'none';
                predictBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

# Home Route
@application.route('/', methods=['GET'])
def homepage():
    return render_template_string(WEB_TEMPLATE)

# Prediction Route
@application.route('/predict', methods=['POST'])
def generate_prediction():
    # Validate Uploaded File
    if 'file' not in request.files:
        return jsonify({
            'error': 'No image file found in the request'
        }), 400

    uploaded_image = request.files['file']
    
    if uploaded_image.filename == '':
        return jsonify({
            'error': 'No image selected'
        }), 400

    try:
        # Image Preprocessing
        processed_image = tf.keras.utils.load_img(
            io.BytesIO(uploaded_image.read()),
            target_size=(150, 150)
        )

        image_array = tf.keras.utils.img_to_array(processed_image)
        normalized_image = image_array / 255.0
        input_tensor = np.expand_dims(normalized_image, axis=0)

        # Model Prediction
        prediction_result = trained_model.predict(input_tensor)
        predicted_index = np.argmax(prediction_result)
        predicted_label = class_labels[predicted_index]
        prediction_confidence = float(np.max(prediction_result))

        # JSON Response
        return jsonify({
            'status': 'success',
            'prediction': predicted_label,
            'confidence': round(prediction_confidence, 4)
        })

    except Exception as error:
        return jsonify({
            'error': f'Prediction Error: {str(error)}'
        }), 500

# Run Flask Server
if __name__ == '__main__':
    application.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )