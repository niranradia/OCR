import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import cv2
import tensorflow_datasets as tfds
import array_record
from tqdm import tqdm  

trainX, trainy = [], []
testX, testy = [], []

try:
    trainX = np.load('trainX.npy')
    trainy = np.load('trainy.npy')
    testX = np.load('testX.npy')
    testy = np.load('testy.npy')
    print("Preprocessed data loaded from disk.")
except:
    print("Download Data First")
    exit()

print(f"trainX shape: {trainX.shape}, trainy shape: {trainy.shape}")
print(f"testX shape: {testX.shape}, testy shape: {testy.shape}")

unique_classes = np.unique(trainy)

num_classes = len(unique_classes)

print(f"Number of unique classes in trainy: {num_classes}")
print(f"Unique classes: {unique_classes}")

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr  
    else:
        return lr * 0.9 

lr_scheduler = LearningRateScheduler(lr_schedule)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

trainy_one_hot = tf.keras.utils.to_categorical(trainy, num_classes=62)
testy_one_hot = tf.keras.utils.to_categorical(testy, num_classes=62)

train_datagen = ImageDataGenerator(
    rescale=1./255,  
    rotation_range=15,  
    zoom_range=0.1, 
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    horizontal_flip=True,
    fill_mode='nearest' 
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_size = int(0.8 * len(trainX))

train_generator = train_datagen.flow(trainX[:train_size], trainy_one_hot[:train_size], batch_size=64)
validation_generator = test_datagen.flow(trainX[train_size:], trainy_one_hot[train_size:], batch_size=64)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=15,
    callbacks=[lr_scheduler, early_stopping]
)

test_loss, test_acc = model.evaluate(testX, testy_one_hot)
print(f"Test accuracy: {test_acc:.4f}")

model.save('model.keras')
