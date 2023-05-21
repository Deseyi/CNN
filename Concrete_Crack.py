# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 20:52:44 2023

@author: desey
"""

import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Define the image size and number of channels
IMG_SIZE = (224, 224)
IMG_CHANNELS = 3

# Define the directories for the positive and negative images
positive_dir = "C:/Users/desey/Documents/MACHINE_LEARN/FinalProject/POSITIVE"
negative_dir = "C:/Users/desey/Documents/MACHINE_LEARN/FinalProject/NEGATIVE"

# Define a function to read and preprocess an image
def preprocess_image(file_path):
    # Load the image using PIL
    image = Image.open(file_path)
    
    # Resize the image
    image = image.resize(IMG_SIZE)
    
    # Convert the image to a numpy array
    image_array = np.array(image)
    
    # Normalize the image
    image_array = image_array / 255.0
    
    return image_array

# Load and preprocess the positive images
positive_images = []
for file_name in os.listdir(positive_dir):
    if file_name.endswith(".jpg") or file_name.endswith(".png"):
        file_path = os.path.join(positive_dir, file_name)
        image_array = preprocess_image(file_path)
        positive_images.append(image_array)
        
positive_images = np.array(positive_images)

# Load and preprocess the negative images
negative_images = []
for file_name in os.listdir(negative_dir):
    if file_name.endswith(".jpg") or file_name.endswith(".png"):
        file_path = os.path.join(negative_dir, file_name)
        image_array = preprocess_image(file_path)
        negative_images.append(image_array)
        
negative_images = np.array(negative_images)

# Combine the positive and negative images into a single array
X = np.concatenate((positive_images, negative_images), axis=0)

# Create the target labels (1 for positive images, 0 for negative images)
y = np.concatenate((np.ones(len(positive_images)), np.zeros(len(negative_images))))

# Shuffle the data
permutation = np.random.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Print the shapes of the data splits
print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)
print("Testing data shape:", X_test.shape)

# Define the model architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test, y_test)
print("Testing set accuracy:", accuracy)



# Make predictions on the testing set
y_pred = model.predict(X_test)
y_pred_classes = np.round(y_pred)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Define the class names
class_names = ['Negative', 'Positive']

# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()