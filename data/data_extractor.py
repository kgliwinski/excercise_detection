import os
import numpy as np
import scipy.io as sio
import pandas as pd
import re

# ==========================================
# 1. Configuration and Environment
# ==========================================
data_base_dir = 'excercise_data_release'
output_base_dir = os.path.join(data_base_dir, 'Exported_Exercises')

# This file has *only* data during exercises, already separated out by exercise.
data_file_single_activity = os.path.join(data_base_dir, 'exercise_data.50.0000_singleonly.mat')

# The EXACT names of the exercises you want
target_exercises = [
    'Bicep Curl', 
    'Dip', 
    'Two-arm Dumbbell Curl (both arms, not alternating)',
    'Overhead Triceps Extension',
    'Lateral Raise',
    'Rest'
]

# --- Digital Twin Constants ---
TARGET_SAMPLES = 150
VEDBA_THRESHOLD = 0.08
EMA_ALPHA = 0.1

# ==========================================
# 2. Digital Twin: VeDBA Calculator
# ==========================================
def calculate_vedba(accel_data, alpha):
    """
    Perfectly mimics the C++ Exponential Moving Average (EMA) gravity filter.
    accel_data is expected to be an (N, 3) numpy array (X, Y, Z).
    """
    vedba = np.zeros(len(accel_data))
    grav_x, grav_y, grav_z = 0.0, 0.0, 1.0 # Assuming 1G resting on Z
    
    for i in range(len(accel_data)):
        ax, ay, az = accel_data[i, 0], accel_data[i, 1], accel_data[i, 2]
        
        # 1. Update Gravity Low-Pass Filter (EMA)
        grav_x = (alpha * ax) + ((1.0 - alpha) * grav_x)
        grav_y = (alpha * ay) + ((1.0 - alpha) * grav_y)
        grav_z = (alpha * az) + ((1.0 - alpha) * grav_z)
        
        # 2. Isolate pure kinetic movement
        dba_x = ax - grav_x
        dba_y = ay - grav_y
        dba_z = az - grav_z
        
        # 3. Calculate 3D Magnitude
        vedba[i] = np.sqrt((dba_x**2) + (dba_y**2) + (dba_z**2))
        
    return vedba

# ==========================================
# 3. Load the Dataset
# ==========================================
print("Loading dataset...")
try:
    data_single = sio.loadmat(data_file_single_activity, squeeze_me=True, struct_as_record=False)
    subject_data = data_single['subject_data']
    activities_array = data_single['exerciseConstants'].activities
    all_activities = [str(act) for act in activities_array]
except NotImplementedError:
    import mat73
    data_single = mat73.loadmat(data_file_single_activity)
    subject_data = data_single['subject_data']
    all_activities = data_single['exerciseConstants']['activities']

n_participants = subject_data.shape[0]

# ==========================================
# 4. Process and Save Repetitions
# ==========================================
print(f"\nExtracting and splitting data into {output_base_dir}...\n")

total_reps_saved = 0
csv_header = "time_s,accel_x_g,accel_y_g,accel_z_g,gyro_x_dps,gyro_y_dps,gyro_z_dps"

for exercise_name in target_exercises:
    if exercise_name not in all_activities:
        print(f"  [!] Warning: '{exercise_name}' not found. Skipping.")
        continue
        
    ex_idx = all_activities.index(exercise_name)
    
    # Create clean directory name for the exercise
    clean_ex_name = re.sub(r'[^a-zA-Z0-9]', '_', exercise_name)
    clean_ex_name = re.sub(r'_+', '_', clean_ex_name).strip('_')
    ex_dir = os.path.join(output_base_dir, clean_ex_name)
    os.makedirs(ex_dir, exist_ok=True)
    
    exercise_rep_count = 0
    
    # Loop through every subject
    for subj_idx in range(n_participants):
        recs = subject_data[subj_idx, ex_idx]
        
        if isinstance(recs, float) and np.isnan(recs):
            continue
            
        if not isinstance(recs, (list, np.ndarray)):
            recs = [recs]
            
        for rec_num, rec in enumerate(recs, start=1):
            accel = rec.data.accelDataMatrix
            gyro = rec.data.gyroDataMatrix
            
            min_len = min(accel.shape[0], gyro.shape[0])
            
            combined_data = np.column_stack((
                accel[:min_len, 0],    # Time
                accel[:min_len, 1:4],  # Accel X, Y, Z
                gyro[:min_len, 1:4]    # Gyro X, Y, Z
            ))
            
            rep_end_indices = []

            # =========================================================
            # STRATEGY A: THE REST CLASS (Bypass Watchdog)
            # =========================================================
            if clean_ex_name.lower() == 'rest':
                # Just chop the resting data into sequential 150-sample chunks
                for i in range(TARGET_SAMPLES, len(combined_data), TARGET_SAMPLES):
                    rep_end_indices.append(i)
                    
            # =========================================================
            # STRATEGY B: ACTIVE EXERCISES (Digital Twin Watchdog)
            # =========================================================
            else:
                vedba_array = calculate_vedba(combined_data[:, 1:4], EMA_ALPHA)
                state = 'RESTING'
                
                for i, vedba_val in enumerate(vedba_array):
                    if state == 'RESTING':
                        if vedba_val > VEDBA_THRESHOLD:
                            state = 'ACTIVE'
                            
                    elif state == 'ACTIVE':
                        if vedba_val < VEDBA_THRESHOLD:
                            state = 'RESTING'
                            # Ensure we have a full 150-sample history to grab
                            if i >= TARGET_SAMPLES:
                                # Debounce: Prevent micro-twitches at the bottom from triggering 5 reps
                                if not rep_end_indices or (i - rep_end_indices[-1] > TARGET_SAMPLES):
                                    rep_end_indices.append(i)

            # =========================================================
            # EXTRACT & SAVE
            # =========================================================
            for i, end_idx in enumerate(rep_end_indices):
                start_idx = end_idx - TARGET_SAMPLES
                
                # Grab the exact 150-sample window
                rep_data = combined_data[start_idx:end_idx, :].copy()
                
                # Synthetic time generation: Reset to 0.0s, step by 0.02s (50Hz)
                rep_data[:, 0] = np.arange(0, TARGET_SAMPLES) * 0.02
                
                # Save the individual repetition
                filename = f"subject_{subj_idx+1:03d}_rec_{rec_num:02d}_rep_{i+1:02d}.csv"
                filepath = os.path.join(ex_dir, filename)
                
                np.savetxt(filepath, rep_data, delimiter=',', header=csv_header, comments='', fmt='%.6f')
                
                total_reps_saved += 1
                exercise_rep_count += 1
                
    print(f"  -> Extracted {exercise_rep_count} hardware-aligned reps for '{exercise_name}'")

print("\n=== Pipeline Complete ===")
print(f"Total fixed-length (150-sample) repetition files saved: {total_reps_saved}")