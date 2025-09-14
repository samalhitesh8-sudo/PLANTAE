import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split

# Define constants
DATASET_PATH = '../data/dataset/'
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10

# Load dataset labels
labels_df = pd.read_csv('../data/dataset_labels.csv')  # Ensure this CSV maps image filenames to labels

# Prepare image file paths and labels
file_paths = []
labels = []

for index, row in labels_df.iterrows():
    filename = row['filename']
    label = row['label']
    file_paths.append(os.path.join(DATASET_PATH, filename))
    labels.append(label)

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Split data
train_paths, val_paths, train_labels, val_labels = train_test_split(
    file_paths, labels_encoded, test_size=0.2, stratify=labels_encoded, random_state=42)

# Data generator
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2)
val_datagen = ImageDataGenerator(rescale=1./255)

def data_generator(file_paths, labels, datagen):
    while True:
        for i in range(0, len(file_paths), BATCH_SIZE):
            batch_paths = file_paths[i:i+BATCH_SIZE]
            batch_labels = labels[i:i+BATCH_SIZE]
            images = []
            for path in batch_paths:
                img = tf.keras.preprocessing.image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
            images = np.array(images)
            batch_labels = np.array(batch_labels)
            yield datagen.flow(images, batch_labels, batch_size=BATCH_SIZE).__next__()

# Build model (Using transfer learning with VGG16)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
steps_per_epoch = len(train_paths) // BATCH_SIZE
validation_steps = len(val_paths) // BATCH_SIZE

model.fit(
    data_generator(train_paths, train_labels, train_datagen),
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=data_generator(val_paths, val_labels, val_datagen),
    validation_steps=validation_steps
)

# Save model
model.save('../model/saved_model.h5')
print("Model trained and saved.")
