import os
import pandas as pd
import numpy as np
import io

def parse_validation_logs(dir_path, file_name, output_folder="extracted_reps"):
    """
    Parses the intermediate Arduino Serial Monitor log file, extracts 150-sample windows,
    injects a 50Hz synthetic timestamp, and saves them to CSV.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_path = os.path.join(dir_path, file_name)
    # get the file name without extension for naming the output CSVs
    file_name = os.path.splitext(file_name)[0]

    with open(file_path, 'r') as file:
        lines = file.readlines()

    is_recording = False
    current_csv_data = []
    detection_count = 0

    for line in lines:
        line = line.strip()

        # 1. Detect the start of a window evaluation
        if "[SYSTEM] Repetition Valley Detected" in line:
            is_recording = True
            current_csv_data = []
            continue
            
        # 2. Capture headers
        if is_recording and line.startswith("accel_x_g,"):
            current_csv_data.append(line)
            continue

        # 3. Detect the AI's prediction and save the file
        if "[AI] Detected:" in line and is_recording:
            is_recording = False
            
            label_part = line.split("Detected:")[1].split("(confidence")[0].strip()
            safe_label = label_part.replace(" ", "_").replace("/", "_")
            
            if len(current_csv_data) > 1:
                detection_count += 1
                
                # Load the collected lines into a pandas DataFrame
                csv_string = "\n".join(current_csv_data)
                df = pd.read_csv(io.StringIO(csv_string))
                
                # ==========================================
                # INJECT SYNTHETIC 50HZ TIMESTAMP
                # ==========================================
                # Create an array from 0 to (length * 0.02) in steps of 0.02
                time_array = np.arange(0, len(df)) * 0.02
                
                # Insert it as the very first column (index 0)
                df.insert(0, 'time_s', time_array)
                
                # Format the time_s column to string with exactly 6 decimal places
                df['time_s'] = df['time_s'].map('{:.6f}'.format)
                # ==========================================
                
                # Save to CSV
                filename = f"{file_name}_rep_{detection_count:03d}_{safe_label}.csv"
                filepath = os.path.join(output_folder, filename)
                df.to_csv(filepath, index=False)
                
                print(f"[{detection_count}] Saved {filename} (Shape: {df.shape}) - Predicted: {label_part}")
            continue

        # 4. If we are recording and it's a data line (contains commas), save it
        if is_recording and "," in line:
            current_csv_data.append(line)

    print(f"\nExtraction complete! Found {detection_count} complete windows.")
    print(f"Files saved in: {os.path.abspath(output_folder)}")

if __name__ == "__main__":
    # Point this to your Arduino log file
    INPUT_LOG_DIR = "validation"
    input_log_files = [f for f in os.listdir(INPUT_LOG_DIR) if f.endswith(".csv")]
    if not input_log_files:
        print(f"Error: No .csv log files found in {INPUT_LOG_DIR}. Please place your Arduino logs there.")
    
    for log_file in input_log_files:
        print(f"\nProcessing log file: {log_file}")
        parse_validation_logs(INPUT_LOG_DIR, log_file)
