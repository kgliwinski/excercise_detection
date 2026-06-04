import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. Configuration
# ==========================================
TFLITE_MODEL_PATH = "exercise_cnn_quantized.tflite"
DATA_DIR = "extracted_reps"

# MUST match the C++ EXERCISE_NAMES array exactly!
# CLASS_NAMES = [
#     "Bicep Curl", "Dip", "Lateral Raise", 
#     "Overhead Triceps Ext", "Rest", "Two arm dumbbell Curl"
# ]
CLASS_NAMES = [
    "Rest", "Bicep Curl"
]

# ==========================================
# 2. Load TFLite Model & Quantization Params
# ==========================================
print(f"Loading TFLite model: {TFLITE_MODEL_PATH}")
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_scale, input_zero_point = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']

print(f"Input Quantization - Scale: {input_scale}, Zero Point: {input_zero_point}")
print(f"Output Quantization - Scale: {output_scale}, Zero Point: {output_zero_point}")

# ==========================================
# 3. Process the Extracted Arduino Data
# ==========================================
print("\nProcessing extracted Arduino reps...")
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

if not csv_files:
    print(f"No CSV files found in {DATA_DIR}/! Please run the extraction script first.")
    exit()

mcu_predictions = []
pc_predictions = []

for file in csv_files:
    # 1. Parse the MCU's prediction from the filename
    # New format: validation_001_logs_rep_001_Rest.csv -> "Rest"
    # or validation_002_logs_rep_001_Lateral_Raise.csv -> "Lateral Raise"
    filename = os.path.basename(file)
    stem = filename.replace('.csv', '')
    parts = stem.split('_')
    
    # Try to find a valid label by checking suffixes from longest to shortest
    mcu_label = None
    for i in range(len(parts), 0, -1):
        candidate = '_'.join(parts[i-1:]).replace('_', ' ')
        if candidate in CLASS_NAMES:
            mcu_label = candidate
            break
    
    if mcu_label is None:
        print(f"Warning: Could not parse label from {filename}. Skipping.")
        continue

    # 2. Load the CSV data
    df = pd.read_csv(file)
    
    # Drop the time_s column if it exists, grab the 6 features
    if 'time_s' in df.columns:
        sensor_data = df.drop(columns=['time_s']).values
    else:
        sensor_data = df.values
        
    # Ensure it's exactly 150 samples
    if len(sensor_data) != 150:
        print(f"Warning: {filename} has {len(sensor_data)} samples instead of 150. Skipping.")
        continue

    # 3. Format exactly like the Keras input: (1, 150, 1, 6)
    sensor_data = np.float32(sensor_data)
    sensor_data = np.expand_dims(sensor_data, axis=0)
    sensor_data = np.expand_dims(sensor_data, axis=2)
    
    # 4. Manual INT8 Quantization (Mimicking C++ exactly)
    # C++ equivalent: input_data[offset] = (val / input_scale) + input_zero_point
    quantized_input = np.round((sensor_data / input_scale) + input_zero_point)
    quantized_input = np.clip(quantized_input, -128, 127).astype(np.int8)

    # 5. Run TFLite Inference
    interpreter.set_tensor(input_details[0]['index'], quantized_input)
    interpreter.invoke()
    quantized_output = interpreter.get_tensor(output_details[0]['index'])[0]

    # 6. Dequantize output & find best class
    float_output = (quantized_output - output_zero_point) * output_scale
    pc_best_class_idx = np.argmax(float_output)
    pc_label = CLASS_NAMES[pc_best_class_idx]
    
    # Store for comparison
    mcu_predictions.append(mcu_label)
    pc_predictions.append(pc_label)

# ==========================================
# 4. Generate Verification Reports
# ==========================================
print("\n=== PC vs MCU Inference Report ===")
print("If these metrics are 100%, your C++ code mathematically matches Python perfectly!\n")

print(classification_report(mcu_predictions, pc_predictions, labels=CLASS_NAMES, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(mcu_predictions, pc_predictions, labels=CLASS_NAMES)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=CLASS_NAMES, 
            yticklabels=CLASS_NAMES)

plt.title('Sim-to-Real Verification Matrix\n(Do the PC and MCU agree?)')
plt.ylabel('MCU Prediction (What the Arduino guessed)')
plt.xlabel('PC Prediction (What TFLite guessed via Python)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()