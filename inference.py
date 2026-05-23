import os
import numpy as np
import tensorflow as tf

# Load the model and labels you just trained
model = tf.keras.models.load_model('best_model.h5')
with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

PRED_DIR = 'dataset/seg_pred' # Path to your prediction folder

# Grab the first 4 image files from the folder
sample_images = [os.path.join(PRED_DIR, img) for img in os.listdir(PRED_DIR)[:4]]

print("\n--- Batch Inference Results for 4 Samples ---")
for img_path in sample_images:
    # Load and preprocess image matching the training configuration
    img = tf.keras.utils.load_img(img_path, target_size=(150, 150))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class = labels[np.argmax(predictions)]
    confidence = np.max(predictions)
    
    print(f"Image: {os.path.basename(img_path)} | Predicted: {predicted_class} | Confidence: {confidence*100:.2f}%")