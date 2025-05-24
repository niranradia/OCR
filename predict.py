import numpy as np
import tensorflow as tf
import cv2

model = tf.keras.models.load_model('model.keras')

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Image not found or invalid format: {image_path}")

    img_resized = cv2.resize(img, (28, 28))
    img_normalized = img_resized / 255.0

    img_ready = np.expand_dims(img_normalized, axis=(0, -1))
    
    return img_ready

def predict_image(image_path, top_k):
    preprocessed_image = preprocess_image(image_path)

    predictions = model.predict(preprocessed_image).flatten()

    top_indices = np.argsort(predictions)[-top_k:][::-1]

    print("Top predictions:")
    values = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    for idx in top_indices:
        print(f"Character: {values[idx]}, Confidence: {predictions[idx]:.4f}")

if __name__ == "__main__":
    predict_image('K.png', 10)
