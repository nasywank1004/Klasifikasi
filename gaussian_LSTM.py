
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import numpy as np

from tensorflow.keras.optimizers import Adam

import numpy as np

from sklearn.metrics import classification_report, confusion_matrix

def preprocess_input_grayscale(img):
    return np.repeat(img, 3, axis=-1)

# Menggunakan ImageDataGenerator untuk data hasil Gaussian Filtering
train_datagen = ImageDataGenerator(
    # preprocessing_function=preprocess_input_grayscale,
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    shear_range=0.2,
    fill_mode='wrap',
)

# Gunakan data dari 'filtered_train' untuk training
train_generator = train_datagen.flow_from_directory(
    'gaussian_train',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
)

val_datagen = ImageDataGenerator(
    # preprocessing_function=preprocess_input_grayscale,
    rescale=1./255
)

# Gunakan data dari 'filtered_test' untuk validasi
val_generator = val_datagen.flow_from_directory(
    'gaussian_test',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

num_classes = len(train_generator.class_indices)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(224,224,3)),
    tf.keras.layers.Reshape((224, 224 * 3)),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(64),
    # tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax') #Output
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'precision', 'recall'])

model.summary()

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=15,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    verbose=1
)

# Mendapatkan nilai akurasi untuk setiap epoch dari history
train_accuracy = history.history['accuracy']  # Akurasi pada data training
val_accuracy = history.history['val_accuracy']  # Akurasi pada data validasi

# Mencetak akurasi di setiap epoch
print("Training Accuracy per Epoch:")
for epoch, acc in enumerate(train_accuracy, 1):
    print(f"Epoch {epoch}: {acc:.4f}")

print("\nValidation Accuracy per Epoch:")
for epoch, val_acc in enumerate(val_accuracy, 1):
    print(f"Epoch {epoch}: {val_acc:.4f}")

model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy', 'precision', 'recall'])

fine_tune_history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    verbose=1
)

# Mendapatkan nilai akurasi untuk setiap epoch dari history
train_accuracy = fine_tune_history.history['accuracy']  # Akurasi pada data training
val_accuracy = fine_tune_history.history['val_accuracy']  # Akurasi pada data validasi
# Mencetak akurasi di setiap epoch
print("Training Accuracy per Epoch:")
for epoch, acc in enumerate(train_accuracy, 1):
    print(f"Epoch {epoch}: {acc:.4f}")

print("\nValidation Accuracy per Epoch:")
for epoch, val_acc in enumerate(val_accuracy, 1):
    print(f"Epoch {epoch}: {val_acc:.4f}")

test_images, test_labels = next(val_generator)

test_images = test_images[:10]
test_labels = test_labels[:10]

predictions = model.predict(test_images)

predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

# Tampilkan hasil prediksi untuk 10 data
for i in range(10):
    print(f"Predicted: {predicted_classes[i]}, True: {true_classes[i]}")

y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get true labels from the test generator
y_true = val_generator.classes  # True class labels

# Get class names from the generator
class_names = list(val_generator.class_indices.keys())

# Generate the classification report
report = classification_report(y_true, y_pred_classes, target_names=class_names)
print("Classification Report: \n", report)

# Optional: Confusion matrix for additional insights
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:\n", conf_matrix)