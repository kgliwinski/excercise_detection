import os

tflite_path = "data/exercise_cnn_quantized.tflite"
header_path = "fw/include/model.h"

with open(tflite_path, 'rb') as f:
    tflite_data = f.read()

hex_lines = [', '.join([f'0x{b:02x}' for b in tflite_data[i:i+12]]) 
             for i in range(0, len(tflite_data), 12)]
hex_string = ',\n  '.join(hex_lines)

with open(header_path, 'w') as f:
    f.write(f"// Auto-generated TFLite Model\n")
    f.write(f"const unsigned char model_tflite[] = {{\n  {hex_string}\n}};\n")
    f.write(f"const unsigned int model_tflite_len = {len(tflite_data)};\n")

print(f"Successfully created {header_path}!")