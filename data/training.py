import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. Configuration
# ==========================================
# Point this to where your split rep folders are
DATA_DIR = 'excercise_data_release/Exported_Exercises'  # Change this to your actual path

# 50Hz data * 3 seconds max = 150 timesteps. 
# Any rep shorter than 3s will be padded with zeros. Any rep longer will be truncated.
MAX_TIMESTEPS = 150 
NUM_FEATURES = 6 # Accel X,Y,Z + Gyro X,Y,Z

# ==========================================
# 2. Data Loading & Padding
# ==========================================
print("Loading and padding data...")
X_data = []
y_labels = []
label_map = {}

# Iterate through each exercise folder
for label_idx, folder_name in enumerate(sorted(os.listdir(DATA_DIR))):
    folder_path = os.path.join(DATA_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue
        
    label_map[label_idx] = folder_name
    print(f"Loading '{folder_name}' as Class {label_idx}...")
    
    # Grab all rep CSVs in this folder
    for file in glob.glob(os.path.join(folder_path, '*.csv')):
        df = pd.read_csv(file)
        
        # Extract the 6 sensor columns
        # (Assuming columns: time_s, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
        sensor_data = df.iloc[:, 1:7].values 
        
        # Pad or Truncate to fixed length
        length = len(sensor_data)
        if length < MAX_TIMESTEPS:
            # Pad with zeros at the end
            padding = np.zeros((MAX_TIMESTEPS - length, NUM_FEATURES))
            sensor_data = np.vstack((sensor_data, padding))
        else:
            # Truncate if too long
            sensor_data = sensor_data[:MAX_TIMESTEPS, :]
            
        X_data.append(sensor_data)
        y_labels.append(label_idx)

# ... (End of Step 2 data loading) ...
X = np.array(X_data, dtype=np.float32)

# ---> NEW LINE: Expand dimensions for Conv2D <---
X = np.expand_dims(X, axis=2) 

y = np.array(y_labels, dtype=np.int32)
NUM_CLASSES = len(label_map)
y = to_categorical(y, num_classes=NUM_CLASSES)

print(f"\nFinal Input Shape: {X.shape} (Samples, Height, Width, Channels)")
print(f"Classes Found: {label_map}")

# Split into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ==========================================
# 3. Define the Tiny 2D-CNN (Acting as a 1D-CNN)
# ==========================================
model = Sequential([
    # First Convolutional Block
    Conv2D(filters=16, kernel_size=(3, 1), activation='relu', input_shape=(MAX_TIMESTEPS, 1, NUM_FEATURES)),
    MaxPooling2D(pool_size=(2, 1)),
    
    # Second Convolutional Block
    Conv2D(filters=32, kernel_size=(3, 1), activation='relu'),
    MaxPooling2D(pool_size=(2, 1)),
    Dropout(0.2),
    
    # Flatten to dense layers
    Flatten(),
    Dense(32, activation='relu'),
    Dense(NUM_CLASSES, activation='sigmoid')
])

model.summary()

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# ==========================================
# 4. Train the Model
# ==========================================
print("\nStarting Training...")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# ==========================================
# 5. TinyML Magic: TFLite Quantization
# ==========================================
print("\nConverting and Quantizing model for Microcontroller...")

# To do INT8 quantization, TFLite needs a small sample of data to calibrate the weights
def representative_data_gen():
    for i in range(100):
        # Yield a single batch of shape (1, 150, 6)
        yield [X_train[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.float32 depending on MCU pipeline
converter.inference_output_type = tf.int8 # or tf.float32

tflite_quant_model = converter.convert()

# Save the TFLite file
tflite_filename = "exercise_cnn_quantized.tflite"
with open(tflite_filename, "wb") as f:
    f.write(tflite_quant_model)

print(f"\nSuccess! Saved quantized model to {tflite_filename}")
print(f"Model size: {os.path.getsize(tflite_filename) / 1024:.2f} KB")

# ==========================================
# 6. Verification: Classification Report & Confusion Matrix
# ==========================================

print("Evaluating the original Keras Model on the Test Set...")

# 1. Get the model's predictions
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# ---> NEW: Convert y_test back from One-Hot to standard integers <---
y_test_classes = np.argmax(y_test, axis=1)

# 2. Re-create our class names list
class_names = [label_map[i] for i in range(NUM_CLASSES)]

# 3. Print the Text Report (Use y_test_classes here!)
print("\n=== Classification Report ===")
print(classification_report(y_test_classes, y_pred_classes, target_names=class_names))

# 4. Generate the Confusion Matrix (Use y_test_classes here too!)
cm = confusion_matrix(y_test_classes, y_pred_classes)

# 5. Plot it beautifully using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)

plt.title('Exercise Recognition Confusion Matrix')
plt.ylabel('Actual Exercise (Ground Truth)')
plt.xlabel('Predicted Exercise (What the AI guessed)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()