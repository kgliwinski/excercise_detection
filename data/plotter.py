import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# we just want a method to plot the raw data of a single repetition (from filename)
def plot_repetition_from_csv(filename):
    df = pd.read_csv(filename)
    
    # Normalize column names to lowercase so the script works regardless of header capitalization
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Calculate the overall magnitude of the accelerometer (gravity + movement)
    # This removes the problem of watch orientation
    df['accel_mag'] = np.sqrt(df['accel_x_g']**2 + df['accel_y_g']**2 + df['accel_z_g']**2)
    
    fig, ax_acc = plt.subplots(figsize=(12, 5))
    ax_gyro = ax_acc.twinx()

    # Accelerometer traces on left axis (g)
    acc_lines = []
    acc_lines.append(ax_acc.plot(df['time_s'], df['accel_mag'], label='Accelerometer Magnitude')[0])
    acc_lines.append(ax_acc.plot(df['time_s'], df['accel_x_g'], label='Accel X', alpha=0.5)[0])
    acc_lines.append(ax_acc.plot(df['time_s'], df['accel_y_g'], label='Accel Y', alpha=0.5)[0])
    acc_lines.append(ax_acc.plot(df['time_s'], df['accel_z_g'], label='Accel Z', alpha=0.5)[0])

    # Gyroscope traces on right axis (dps)
    gyro_lines = []
    gyro_lines.append(ax_gyro.plot(df['time_s'], df['gyro_x_dps'], label='Gyro X', alpha=0.6, linestyle='--')[0])
    gyro_lines.append(ax_gyro.plot(df['time_s'], df['gyro_y_dps'], label='Gyro Y', alpha=0.6, linestyle='--')[0])
    gyro_lines.append(ax_gyro.plot(df['time_s'], df['gyro_z_dps'], label='Gyro Z', alpha=0.6, linestyle='--')[0])

    # Independent y-limits based on highest absolute values in each sensor group
    acc_limit = np.nanmax(np.abs(df[['accel_mag', 'accel_x_g', 'accel_y_g', 'accel_z_g']].to_numpy()))
    gyro_limit = np.nanmax(np.abs(df[['gyro_x_dps', 'gyro_y_dps', 'gyro_z_dps']].to_numpy()))
    acc_limit = acc_limit * 1.1 if np.isfinite(acc_limit) and acc_limit > 0 else 1.0
    gyro_limit = gyro_limit * 1.1 if np.isfinite(gyro_limit) and gyro_limit > 0 else 1.0

    ax_acc.set_ylim(-acc_limit, acc_limit)
    ax_gyro.set_ylim(-gyro_limit, gyro_limit)

    ax_acc.set_title(f"Raw Data Plot for {filename}")
    ax_acc.set_xlabel("Time (seconds)")
    ax_acc.set_ylabel("Acceleration (g)")
    ax_gyro.set_ylabel("Angular Velocity (dps)")

    # Single legend containing lines from both axes
    lines = acc_lines + gyro_lines
    labels = [line.get_label() for line in lines]
    ax_acc.legend(lines, labels, loc='best')

    fig.tight_layout()
    plt.show()

output_base_dir = 'excercise_data_release/Exported_Exercises/'
for dir in os.listdir(output_base_dir):
    dir_path = os.path.join(output_base_dir, dir)
    if os.path.isdir(dir_path):
        for file in os.listdir(dir_path):
            if file.endswith('.csv'):
                plot_repetition_from_csv(os.path.join(dir_path, file))
                break  # just plot the first one for now