import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import glob
import os

# ==========================================
# 1. Configuration
# ==========================================
# Point this to one of the CSVs we generated in the last step
# Automatically find the first matching exported CSV for Bicep_Curl
pattern = os.path.join('excercise_data_release', 'Exported_Exercises', 'Bicep_Curl', 'Subj001_Bicep_Curl_Rec01.csv')
matches = glob.glob(pattern)
if not matches:
    raise FileNotFoundError(f"No CSV files found for Bicep_Curl under {pattern}")

input_csv = matches[0]
output_prefix = os.path.splitext(input_csv)[0]

# ==========================================
# 2. Load Data & Calculate Magnitude
# ==========================================
df = pd.read_csv(input_csv)

# Normalize column names to lowercase so the script works regardless of header capitalization
df.columns = [c.strip().lower() for c in df.columns]

# Calculate the overall magnitude of the accelerometer (gravity + movement)
# This removes the problem of watch orientation
df['accel_mag'] = np.sqrt(df['accel_x_g']**2 + df['accel_y_g']**2 + df['accel_z_g']**2)

# ==========================================
# 3. Peak Detection (Finding the reps)
# ==========================================
# To find the end of a rep, we actually want to find the "valleys" (where movement stops).
# We can do this by finding the peaks of the INVERTED magnitude.
inverted_mag = df['accel_mag'] * -1

# Find peaks. 
# distance=50 means reps must be at least 1 second apart (since data is 50Hz)
# prominence=0.1 ensures we only grab actual reps, not tiny twitches
peaks, properties = find_peaks(inverted_mag, distance=50, prominence=0.1)

print(f"Detected {len(peaks)-1} complete repetitions based on {len(peaks)} boundaries.")

# ==========================================
# 4. Plot to Verify
# ==========================================
plt.figure(figsize=(12, 5))
plt.plot(df['time_s'], df['accel_mag'], label='Accelerometer Magnitude')

# Draw red lines where we are chopping the data
for peak in peaks:
    plt.axvline(x=df['time_s'].iloc[peak], color='red', linestyle='--', alpha=0.7)

plt.title("Repetition Slicing via Peak Detection")
plt.xlabel("Time (seconds)")
plt.ylabel("Magnitude (g)")
plt.legend()
plt.show()

# ==========================================
# 5. Extract and Save Individual Reps
# ==========================================
# Loop through the peak boundaries and slice the dataframe
for i in range(len(peaks) - 1):
    start_idx = peaks[i]
    end_idx = peaks[i+1]
    
    # Extract just this repetition
    rep_df = df.iloc[start_idx:end_idx].copy()
    
    # Optional: Reset the time to start at 0.0 for every single rep 
    # (Highly recommended for AI model training!)
    rep_df['time_s'] = rep_df['time_s'] - rep_df['time_s'].iloc[0]
    
    # Save to a new file
    output_file = f"{output_prefix}_rep_{i+1:02d}.csv"
    rep_df.drop(columns=['accel_mag']).to_csv(output_file, index=False)
    
print("Individual repetition files saved successfully!")