import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import re
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# ==========================================
# Constants and environment
# ==========================================

data_base_dir = 'excercise_data_release'
output_base_dir = os.path.join(data_base_dir, 'Exported_Exercises')

# This file has *only* data during exercises, already separated out by exercise.
data_file_single_activity = os.path.join(data_base_dir, 'exercise_data.50.0000_singleonly.mat')

# This file has data from complete exercise sessions, including non-exercise time.
data_file_multi_activity = os.path.join(data_base_dir, 'exercise_data.50.0000_multionly.mat')

# Put the EXACT names of the 5 exercises you want here (without the "04 | " numbers)
# Replace these examples with the actual strings printed from your header script
target_exercises = [
    'Bicep Curl', 
    'Dip', 
    'Two-arm Dumbbell Curl (both arms, not alternating)',
    'Overhead Triceps Extension',
    'Lateral Raise'
]

# ==========================================
# 2. Load the Dataset
# ==========================================
print("Loading dataset...")
try:
    data_single = sio.loadmat(data_file_single_activity, squeeze_me=True, struct_as_record=False)
    subject_data = data_single['subject_data']
    
    # Extract the activities list
    activities_array = data_single['exerciseConstants'].activities
    all_activities = [str(act) for act in activities_array]
    
except NotImplementedError:
    # Fallback for massive v7.3 files
    import mat73
    data_single = mat73.loadmat(data_file_single_activity)
    subject_data = data_single['subject_data']
    all_activities = data_single['exerciseConstants']['activities']

n_participants = subject_data.shape[0]

def print_all_excercises():
    print("Available exercises in the dataset:")
    for idx, act in enumerate(all_activities):
        print(f"  {idx+1:02d} | {act}")

# ==========================================
# 3. Process and Save Repetitions
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
        
        # Skip if subject didn't do this exercise
        if isinstance(recs, float) and np.isnan(recs):
            continue
            
        if not isinstance(recs, (list, np.ndarray)):
            recs = [recs]
            
        # Extract each recording (visit) for this subject
        for rec_num, rec in enumerate(recs, start=1):
            accel = rec.data.accelDataMatrix
            gyro = rec.data.gyroDataMatrix
            
            min_len = min(accel.shape[0], gyro.shape[0])
            
            # Stack the core data: [Time, Accel X-Y-Z, Gyro X-Y-Z]
            combined_data = np.column_stack((
                accel[:min_len, 0],    
                accel[:min_len, 1:4],  
                gyro[:min_len, 1:4]    
            ))
            
            # --- UPGRADED PEAK DETECTION (Repetition Splitting) ---
            # 1. Calculate raw magnitude
            accel_mag = np.sqrt(combined_data[:, 1]**2 + combined_data[:, 2]**2 + combined_data[:, 3]**2)
            
            # 2. Smooth the signal to remove twitchy sensor noise (sigma=3 works well for 50Hz)
            smoothed_mag = gaussian_filter1d(accel_mag, sigma=3)
            
            # 3. Invert to find valleys
            inverted_mag = smoothed_mag * -1
            
            # 4. Smarter Peak Finding
            # - distance=50: reps are still min 1 second apart
            # - prominence=0.15: Requires a more distinct valley
            # - height=-1.2: The magnitude MUST drop back down near 1.0g (gravity) to count as a rest.
            #   (Because it's inverted, we look for values > -1.2)
            peaks, _ = find_peaks(inverted_mag, distance=50, prominence=0.15, height=-1.2)
            
            # If no clear reps are found, save the whole recording as "rep 1"
            if len(peaks) < 2:
                # Zero out the time
                combined_data[:, 0] = combined_data[:, 0] - combined_data[0, 0]
                
                filename = f"subject_{subj_idx+1:03d}_rec_{rec_num:02d}_rep_01_fallback.csv"
                filepath = os.path.join(ex_dir, filename)
                np.savetxt(filepath, combined_data, delimiter=',', header=csv_header, comments='', fmt='%.6f')
                
                total_reps_saved += 1
                exercise_rep_count += 1
                continue
                
            # If peaks are found, slice into individual reps
            for i in range(len(peaks) - 1):
                start_idx = peaks[i]
                end_idx = peaks[i+1]
                
                # Copy the slice so we don't modify the original array
                rep_data = combined_data[start_idx:end_idx, :].copy()
                
                # Reset time to start at 0.0 seconds
                rep_data[:, 0] = rep_data[:, 0] - rep_data[0, 0]
                
                # Save the individual repetition
                rep_num = i + 1
                filename = f"subject_{subj_idx+1:03d}_rec_{rec_num:02d}_rep_{rep_num:02d}.csv"
                filepath = os.path.join(ex_dir, filename)
                
                np.savetxt(filepath, rep_data, delimiter=',', header=csv_header, comments='', fmt='%.6f')
                
                total_reps_saved += 1
                exercise_rep_count += 1
                
    print(f"  -> Extracted {exercise_rep_count} individual reps for '{exercise_name}'")

print("\n=== Pipeline Complete ===")
print(f"Total individual repetition files saved: {total_reps_saved}")