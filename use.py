import numpy as np
import tensorflow as tf
import cv2

model = tf.keras.models.load_model('model.keras')

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Image not found or invalid format: {image_path}")

    img_resized = cv2.resize(img, (28, 28))

    if np.mean(img_resized) > 127:
        img_resized = 255 - img_resized

    img_normalized = img_resized / 255.0
    img_ready = np.expand_dims(img_normalized, axis=(0, -1))
    
    return img_ready

def predict_image(image_path):
    preprocessed_image = preprocess_image(image_path)

    predictions = model.predict(preprocessed_image)

    predicted_class = np.argmax(predictions)

    return predicted_class

image_path = 'K.png'

predicted_value = predict_image(image_path)

values = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

print(f"Predicted value: {values[predicted_value]}")
