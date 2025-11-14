"""
Analyze why yaw climbs at startup.
"""

import numpy as np
import matplotlib.pyplot as plt

# Load raw IMU data
print("Loading raw IMU data...")
data = []
with open('real_imu_data.csv', 'r') as f:
    for line in f:
        if line.startswith('#') or not line.strip():
            continue
        values = [float(x) for x in line.strip().split(',')]
        data.append(values)

data = np.array(data)
timestamps = data[:, 0] / 1e6  # Convert to seconds
accel = data[:, 1:4]
gyro = data[:, 4:7]
mag = data[:, 7:10]

# Load processed orientation
print("Loading processed orientation...")
orient_data = np.loadtxt('real_imu_data_orientations.csv', delimiter=',', skiprows=1)
orient_timestamps = orient_data[:, 0]
euler = orient_data[:, 1:4]

# Focus on first 5 seconds
mask = timestamps < (timestamps[0] + 5)
t = timestamps[mask] - timestamps[0]

mask_orient = orient_timestamps < (orient_timestamps[0] + 5)
t_orient = orient_timestamps[mask_orient] - orient_timestamps[0]

# Create diagnostic plot
fig, axes = plt.subplots(5, 1, figsize=(14, 12))

# 1. Magnetometer readings
axes[0].plot(t, mag[mask, 0], 'r-', label='mx', alpha=0.7)
axes[0].plot(t, mag[mask, 1], 'g-', label='my', alpha=0.7)
axes[0].plot(t, mag[mask, 2], 'b-', label='mz', alpha=0.7)
axes[0].set_ylabel('Magnetometer (calibrated)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_title('First 5 Seconds of IMU Data - Yaw Startup Analysis')

# 2. Gyroscope readings
axes[1].plot(t, gyro[mask, 0], 'r-', label='gx', alpha=0.7)
axes[1].plot(t, gyro[mask, 1], 'g-', label='gy', alpha=0.7)
axes[1].plot(t, gyro[mask, 2], 'b-', label='gz', alpha=0.7)
axes[1].set_ylabel('Gyro (rad/s)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Accelerometer readings
axes[2].plot(t, accel[mask, 0], 'r-', label='ax', alpha=0.7)
axes[2].plot(t, accel[mask, 1], 'g-', label='ay', alpha=0.7)
axes[2].plot(t, accel[mask, 2], 'b-', label='az', alpha=0.7)
axes[2].set_ylabel('Accel (m/s²)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# 4. Euler angles
axes[3].plot(t_orient, euler[mask_orient, 0], 'r-', label='Roll', alpha=0.7)
axes[3].plot(t_orient, euler[mask_orient, 1], 'g-', label='Pitch', alpha=0.7)
axes[3].plot(t_orient, euler[mask_orient, 2], 'b-', label='Yaw', linewidth=2)
axes[3].set_ylabel('Angle (degrees)')
axes[3].legend()
axes[3].grid(True, alpha=0.3)
axes[3].axhline(y=0, color='k', linestyle='--', alpha=0.3)

# 5. Magnetometer magnitude
mag_norm = np.linalg.norm(mag[mask], axis=1)
axes[4].plot(t, mag_norm, 'k-', label='|mag|', linewidth=2)
axes[4].set_ylabel('Mag magnitude')
axes[4].set_xlabel('Time (seconds)')
axes[4].legend()
axes[4].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('startup_analysis.png', dpi=150)
print("\nAnalysis saved to startup_analysis.png")

# Print statistics
print("\n" + "="*60)
print("STARTUP ANALYSIS")
print("="*60)

# Check for IMU motion in first 2 seconds
mask_2s = timestamps < (timestamps[0] + 2)
gyro_2s = gyro[mask_2s]
accel_2s = accel[mask_2s]

print(f"\nFirst 2 seconds statistics:")
print(f"  Gyro range: x=[{gyro_2s[:,0].min():.3f}, {gyro_2s[:,0].max():.3f}] rad/s")
print(f"             y=[{gyro_2s[:,1].min():.3f}, {gyro_2s[:,1].max():.3f}] rad/s")
print(f"             z=[{gyro_2s[:,2].min():.3f}, {gyro_2s[:,2].max():.3f}] rad/s")
print(f"  Max gyro magnitude: {np.linalg.norm(gyro_2s, axis=1).max():.3f} rad/s")
print(f"  Accel std: x={accel_2s[:,0].std():.3f}, y={accel_2s[:,1].std():.3f}, z={accel_2s[:,2].std():.3f}")

# Check if yaw change matches gyro integration
mask_orient_2s = orient_timestamps < (orient_timestamps[0] + 2)
yaw_change = euler[mask_orient_2s, 2][-1] - euler[mask_orient_2s, 2][0]
print(f"\n  Yaw change in first 2s: {yaw_change:.1f}°")

# Integrate gyro Z to estimate expected yaw change
dt = np.diff(timestamps[mask_2s]).mean()
gyro_z_integrated = np.cumsum(gyro_2s[:, 2]) * dt * (180/np.pi)  # Convert to degrees
print(f"  Gyro-Z integrated: {gyro_z_integrated[-1]:.1f}° (if starting from 0)")
print(f"\n  -> Gyro shows motion: {'YES - IMU was rotating!' if abs(gyro_z_integrated[-1]) > 10 else 'NO - stationary'}")

print("\nConclusion:")
if abs(gyro_z_integrated[-1]) > 10:
    print("  The yaw climb is REAL MOTION - the IMU was physically rotating during startup.")
    print("  This is NOT a filter artifact, but actual rotation captured by the gyroscope.")
else:
    print("  The yaw climb appears to be a convergence artifact during filter initialization.")
    print("  The magnetometer reference may have taken time to stabilize.")
