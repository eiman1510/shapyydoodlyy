import tensorflow as tf
import kagglehub
import os
import joblib
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Download dataset and set paths
base_path = kagglehub.dataset_download("smeschke/four-shapes")
dataset_path = os.path.join(base_path, "shapes")

# Define model architecture
img_size = 64
model = tf.keras.Sequential([
    tf.keras.Input(shape=(img_size, img_size, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data preprocessing
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

# Train model
model.fit(train_generator, epochs=10)

# Save TensorFlow model
model.save('shapes_model.h5')
print("TensorFlow model saved as shapes_model.h5")

# Create and save scikit-learn model for deployment
# Extract features and labels from the generator
features = []
labels = []
for i in range(len(train_generator)):
    x, y = train_generator[i]
    features.append(x.reshape(x.shape[0], -1))
    labels.append(np.argmax(y, axis=1))
    
features = np.concatenate(features)
labels = np.concatenate(labels)

# Train a simple decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(features, labels)

# Save scikit-learn model
joblib.dump(clf, 'shapes_model.joblib')
print("Scikit-learn model saved as shapes_model.joblib") 