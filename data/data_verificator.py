import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Configuration
# ==========================================
# Point this to a folder we generated in the last step
exercise_dir = 'excercise_data_release/Exported_Exercises/Bicep_Curl'  # Change this to the exercise you want to verify

# Let's verify Subject 1's first recording. 
# We use glob to find all files matching that specific pattern.
file_pattern = os.path.join(exercise_dir, 'subject_002_rec_01_rep_*.csv')
rep_files = sorted(glob.glob(file_pattern))

if not rep_files:
    print("No files found! Check your path.")
    exit(1)
else:
    print(f"Found {len(rep_files)} repetitions to verify.")

# ==========================================
# Plotting
# ==========================================
plt.figure(figsize=(10, 6))

for file in rep_files:
    # Load the split CSV
    df = pd.read_csv(file)
    
    # Calculate magnitude to get the overall movement shape
    mag = np.sqrt(df['accel_x_g']**2 + df['accel_y_g']**2 + df['accel_z_g']**2)
    
    # Get just the rep number for the label (e.g., "rep_01")
    rep_name = os.path.basename(file).split('.')[0].split('_')[-2:]
    label = f"{rep_name[0]}_{rep_name[1]}"
    
    # Plot Time vs. Magnitude
    plt.plot(df['time_s'], mag, alpha=0.5)

plt.title("Split Verification: Overlaid Overhead Triceps Extension Repetitions")
plt.xlabel("Time (seconds) - All normalized to 0.0s")
plt.ylabel("Accelerometer Magnitude (g)")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()